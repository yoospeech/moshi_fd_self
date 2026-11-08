"""Train user-audio conditioned assistant response with Moshi interleaving."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from .model import (
    FullDuplexConfig,
    MoshiFullDuplexModel,
    masked_multistream_loss,
)
from .dataset import DailyTalkDataset, collate_dailytalk
from .distributed import TrainingObjective, setup, barrier

from moshi.models import loaders as moshi_loaders


MODEL_REPOSITORIES = {
    "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
    "LFM3-1.7B": "LiquidAI/LFM3-1.7B",
}


def acoustic_prompt_mask(
    batch_size: int, text_only_fraction: float, *, device: torch.device | str
) -> torch.Tensor:
    """Select acoustic-conditioned and text-only samples within one batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= text_only_fraction <= 1.0:
        raise ValueError("text_only_fraction must be between 0 and 1")
    text_only_count = min(round(batch_size * text_only_fraction), batch_size - 1)
    mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    if text_only_count:
        text_only_indices = torch.randperm(batch_size, device=device)[:text_only_count]
        mask[text_only_indices] = False
    return mask


def resolve_model_source(model: str, download_root: str) -> tuple[Path, str | None]:
    """Return the local model path and an optional repository to download."""
    requested = Path(model)
    if requested.exists():
        return requested, None
    basename = model.rstrip("/").rsplit("/", 1)[-1]
    local_path = Path(download_root) / basename
    if local_path.exists():
        return local_path, None
    if requested.parent.name == "models" or "/" not in model:
        repository = MODEL_REPOSITORIES.get(basename)
        if repository is None:
            raise FileNotFoundError(
                f"unknown local model alias {model!r}; pass an existing path or a Hugging Face repo ID"
            )
    else:
        repository = model
    return local_path, repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--model", default="models/SmolLM2-135M")
    parser.add_argument(
        "--model-download-root",
        default="models",
        help="download Hugging Face temporal models under this directory",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow custom Hugging Face model code; enabled automatically for LFM3",
    )
    parser.add_argument("--mimi-checkpoint")
    parser.add_argument("--mimi-repo", default=moshi_loaders.DEFAULT_REPO)
    parser.add_argument("--mimi-device", help="e.g. cuda:1; defaults to the training device")
    parser.add_argument("--output", default="runs/full_duplex")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--text-only-fraction", type=float, default=0.5,
        help="fraction of each training batch with audio removed (default: 0.5)",
    )
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--temporal-learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-codebooks", type=int, default=4)
    parser.add_argument("--audio-card", type=int, default=2048)
    parser.add_argument("--context", type=int, default=3000)
    parser.add_argument(
        "--assistant-text-delay-steps", type=int, default=0,
        help="must remain 0 for assistant inner-monologue timing",
    )
    parser.add_argument("--acoustic-delay", type=int, default=1)
    parser.add_argument("--depformer-dim", type=int, default=1024)
    parser.add_argument("--depformer-heads", type=int, default=16)
    parser.add_argument("--depformer-layers", type=int, default=6)
    parser.add_argument("--depformer-ffn-dim", type=int, default=4096)
    parser.add_argument(
        "--audio-loss-weight", type=float, default=0.1,
        help="assistant feedback-audio loss weight; 0 disables assistant audio prediction",
    )
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--semantic-loss-weight", type=float, default=100.0)
    parser.add_argument(
        "--text-padding-weight",
        type=float,
        default=0.5,
        help="PAD loss weight relative to non-PAD text tokens (default: 0.5)",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--valid-every", type=int, default=500)
    parser.add_argument("--valid-batches", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=200.0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--min-words-per-second", type=float, default=0.5)
    parser.add_argument("--max-words-per-second", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=4)
    parser.add_argument("--print-model", action="store_true", help="print the full PyTorch module tree")
    return parser.parse_args()


def save_checkpoint(model: MoshiFullDuplexModel, optimizer: torch.optim.Optimizer, step: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        output / f"checkpoint_{step:08d}.pt",
    )
    (output / "config.json").write_text(json.dumps(model.checkpoint_metadata(), indent=2), encoding="utf-8")


def print_model_summary(
    model: MoshiFullDuplexModel,
    config: FullDuplexConfig,
    mimi: torch.nn.Module | None,
    *,
    full: bool = False,
) -> None:
    groups = {"temporal": 0, "streams_and_text_head": 0}
    for name, parameter in model.named_parameters():
        if name.startswith("transformer."):
            groups["temporal"] += parameter.numel()
        else:
            groups["streams_and_text_head"] += parameter.numel()

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    print("model structure:")
    print(
        f"  temporal: {config.temporal_model} hidden={model.dim} "
        f"context={config.context}"
    )
    print(
        f"  inputs: text=1 assistant_audio={config.num_codebooks} user_audio={config.num_codebooks} card={config.audio_card} "
        f"delays={config.delays}"
    )
    objective = "assistant text"
    if config.audio_loss_weight > 0:
        objective += " + assistant feedback audio Depth Transformer"
    print(f"  objective: {objective}")
    for name, count in groups.items():
        print(f"  parameters/{name}: {count / 1e6:.2f}M")
    print(f"  parameters/total: {total / 1e6:.2f}M (trainable {trainable / 1e6:.2f}M)")
    print(f"  parameter storage: {parameter_bytes / 2**20:.1f} MiB ({model.dtype})")
    if mimi is not None:
        mimi_total = sum(parameter.numel() for parameter in mimi.parameters())
        mimi_bytes = sum(parameter.numel() * parameter.element_size() for parameter in mimi.parameters())
        mimi_device = next(mimi.parameters()).device
        print(
            f"  mimi/frozen: {mimi_total / 1e6:.2f}M, "
            f"{mimi_bytes / 2**20:.1f} MiB, device={mimi_device}"
        )
    if full:
        print("full model:")
        print(model)


@torch.no_grad()
def online_streams(
    batch: dict[str, object],
    mimi: torch.nn.Module,
    device: torch.device,
    config: FullDuplexConfig,
    acoustic_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build Moshi streams: assistant text/audio followed by user audio."""
    assistant_text_tokens = batch["assistant_text_tokens"].to(  # type: ignore[union-attr]
        device, non_blocking=True
    )
    audio_frame_lengths = batch["audio_frame_lengths"].tolist()  # type: ignore[union-attr]
    assistant_text_frame_lengths = batch["assistant_text_frame_lengths"].tolist()  # type: ignore[union-attr]
    mimi_device = next(mimi.parameters()).device
    waveforms = batch["waveforms"].to(mimi_device, non_blocking=True)  # type: ignore[union-attr]
    assistant_waveforms = batch["assistant_waveforms"].to(  # type: ignore[union-attr]
        mimi_device, non_blocking=True
    )
    combined_codes = mimi.encode(torch.cat([waveforms, assistant_waveforms], dim=0)).to(device)
    audio_codes, assistant_codes = combined_codes.chunk(2, dim=0)
    if audio_codes.shape[1] != config.num_codebooks:
        raise RuntimeError(
            f"Mimi returned {audio_codes.shape[1]} codebooks, expected {config.num_codebooks}"
        )

    tail = max(config.delays)
    total_frames = max(
        audio_codes.shape[-1], max(assistant_text_frame_lengths)
    ) + tail
    streams = torch.full(
        (audio_codes.shape[0], 1 + 2 * config.num_codebooks, total_frames),
        -1,
        dtype=torch.long,
        device=device,
    )
    if acoustic_mask is None:
        acoustic_mask = torch.ones(audio_codes.shape[0], dtype=torch.bool, device=device)
    else:
        acoustic_mask = acoustic_mask.to(device=device, dtype=torch.bool)
        if acoustic_mask.shape != (audio_codes.shape[0],):
            raise ValueError("acoustic_mask must have one boolean value per sample")
    for index, (audio_frames, assistant_text_frames) in enumerate(
        zip(
            audio_frame_lengths,
            assistant_text_frame_lengths,
            strict=True,
        )
    ):
        valid_audio_frames = min(audio_frames, audio_codes.shape[-1])
        streams[index, 0, :assistant_text_frames] = assistant_text_tokens[
            index, :assistant_text_frames
        ]
        streams[
            index, 0, assistant_text_frames : assistant_text_frames + tail
        ] = config.text_padding_id
        if acoustic_mask[index]:
            valid_assistant_frames = min(audio_frames, assistant_codes.shape[-1])
            streams[index, 1 : 1 + config.num_codebooks, :valid_assistant_frames] = assistant_codes[
                index, :, :valid_assistant_frames
            ]
            streams[index, 1 + config.num_codebooks :, :valid_audio_frames] = audio_codes[
                index, :, :valid_audio_frames
            ]
    return streams


@torch.no_grad()
def evaluate(
    model: MoshiFullDuplexModel,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    text_padding_id: int,
    max_batches: int,
    mimi: torch.nn.Module,
    config: FullDuplexConfig,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        streams = online_streams(batch, mimi, device, config)
        autocast = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast:
            text_logits, text_mask, audio_logits, audio_mask = model.forward_training(
                streams, predict_audio=config.audio_loss_weight > 0
            )
            _, metrics = masked_multistream_loss(
                text_logits, text_mask,
                streams,
                text_padding_id=text_padding_id,
                text_padding_weight=config.text_padding_weight,
                audio_logits=audio_logits,
                audio_mask=audio_mask,
                audio_loss_weight=config.audio_loss_weight,
                text_loss_weight=config.text_loss_weight,
                semantic_loss_weight=config.semantic_loss_weight,
            )
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value.item()
        count += 1
    model.train()
    if count == 0:
        raise RuntimeError("validation loader produced no batches")
    return {name: value / count for name, value in totals.items()}


def main() -> None:
    try:
        run_training()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_training() -> None:
    args = parse_args()
    if not 0.0 <= args.text_only_fraction <= 1.0:
        raise ValueError("text-only-fraction must be between 0 and 1")
    if args.text_only_fraction > 0 and args.batch_size < 2:
        raise ValueError("batch-size must be at least 2 when mixing text-only samples")
    for name in ("batch_size", "grad_accum", "steps", "smoke_steps", "log_every", "save_every", "valid_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    rank, world_size, device = setup()
    primary = rank == 0
    if world_size > 1 and args.mimi_device is not None:
        raise ValueError("DDP assigns Mimi to each rank's local device; omit --mimi-device")
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    model_path, model_repository = resolve_model_source(
        args.model, args.model_download_root
    )
    if model_repository is not None:
        if primary:
            print(f"downloading {model_repository} to {model_path}", flush=True)
            snapshot_download(repo_id=model_repository, local_dir=model_path)
        barrier()
        if not model_path.exists():
            raise FileNotFoundError(f"downloaded model directory not found: {model_path}")
    model_name = str(model_path)

    trust_remote_code = (
        args.trust_remote_code
        if args.trust_remote_code is not None
        else "lfm3" in model_name.lower()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    base_text_card = len(tokenizer)
    # PAD and EPAD are shared vocabulary rows; timelines and heads are separate.
    config = FullDuplexConfig(
        temporal_model=model_name,
        trust_remote_code=trust_remote_code,
        text_card=base_text_card + 2,
        text_padding_id=base_text_card,
        text_epad_id=base_text_card + 1,
        audio_card=args.audio_card,
        num_codebooks=args.num_codebooks,
        context=args.context,
        assistant_text_delay_steps=args.assistant_text_delay_steps,
        acoustic_delay_steps=args.acoustic_delay,
        depformer_dim=args.depformer_dim,
        depformer_heads=args.depformer_heads,
        depformer_layers=args.depformer_layers,
        depformer_ffn_dim=args.depformer_ffn_dim,
        audio_loss_weight=args.audio_loss_weight,
        text_loss_weight=args.text_loss_weight,
        semantic_loss_weight=args.semantic_loss_weight,
        text_padding_weight=args.text_padding_weight,
    )
    model = MoshiFullDuplexModel(config, dtype=dtype, device=device)
    model.train()

    mimi = None
    if not args.smoke_test:
        mimi_checkpoint = args.mimi_checkpoint
        if mimi_checkpoint is None:
            mimi_checkpoint = str(moshi_loaders.hf_get(moshi_loaders.MIMI_NAME, args.mimi_repo))
        mimi_device = torch.device(args.mimi_device) if args.mimi_device else device
        mimi = moshi_loaders.get_mimi(
            mimi_checkpoint,
            device=mimi_device,
            num_codebooks=config.num_codebooks,
        )
        mimi.eval()
        for parameter in mimi.parameters():
            parameter.requires_grad_(False)

    if primary:
        print_model_summary(model, config, mimi, full=args.print_model)
        print(f"world_size={world_size} effective_batch={world_size * args.batch_size * args.grad_accum}")
    print(f"rank={rank} training_device={device} mimi_device={device if world_size > 1 else args.mimi_device or device}", flush=True)

    tensorboard_dir = Path(args.tensorboard_dir or Path(args.output) / "tensorboard")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise SystemExit(
            "TensorBoard logging requires the 'tensorboard' package: pip install tensorboard"
        ) from error
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if primary else None

    objective = TrainingObjective(
        model,
        config.text_padding_id,
        config.text_padding_weight,
        config.text_loss_weight,
        config.audio_loss_weight,
        config.semantic_loss_weight,
    )
    if world_size > 1:
        # The backbone's own token embedding is bypassed by inputs_embeds.
        objective = DistributedDataParallel(
            objective, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    torch.manual_seed(args.seed + rank)
    train_sampler = None

    temporal_parameters = list(model.transformer.parameters())
    temporal_ids = {id(parameter) for parameter in temporal_parameters}
    new_parameters = [parameter for parameter in model.parameters() if id(parameter) not in temporal_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": temporal_parameters, "lr": args.temporal_learning_rate},
            {"params": new_parameters, "lr": args.learning_rate},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    if args.smoke_test:
        # Valid DSM controls plus random Mimi codes. This checks the complete
        # audio-conditioned text loss and backward path without a dataset.
        smoke_frames = max(64, max(config.delays) + 8)
        text = torch.randint(
            0, config.text_card, (args.batch_size, 1, smoke_frames), device=device
        )
        audio = torch.randint(
            0, args.audio_card, (args.batch_size, args.num_codebooks, smoke_frames), device=device
        )
        assistant_audio_targets = torch.randint(
            0, args.audio_card, (args.batch_size, args.num_codebooks, smoke_frames), device=device
        )
        streams = torch.cat([text, assistant_audio_targets, audio], dim=1)
        batches = [{"streams": streams}]
        valid_loader = None
        target_steps = args.smoke_steps
    else:
        if args.manifest is None:
            raise SystemExit("--manifest is required unless --smoke-test is used")
        dataset = DailyTalkDataset(
            args.manifest,
            tokenizer,
            sample_rate=24000,
            frame_rate=config.frame_rate,
            text_padding_id=config.text_padding_id,
            text_epad_id=config.text_epad_id,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_words=args.min_words,
            min_words_per_second=args.min_words_per_second,
            max_words_per_second=args.max_words_per_second,
        )
        print(
            f"dataset filter: original={dataset.original_size} kept={len(dataset)} "
            f"hours={dataset.filtered_hours:.2f} rejected={dataset.rejected}"
        )
        if not 0.0 < args.valid_ratio < 1.0:
            raise SystemExit("--valid-ratio must be between 0 and 1")
        valid_size = max(1, round(len(dataset) * args.valid_ratio))
        train_size = len(dataset) - valid_size
        if train_size < 1:
            raise SystemExit("validation split leaves no training samples")
        split_generator = torch.Generator().manual_seed(args.seed)
        train_dataset, valid_dataset = random_split(
            dataset,
            [train_size, valid_size],
            generator=split_generator,
        )
        output_dir = Path(args.output)
        if primary:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "split.json").write_text(
                json.dumps(
                    {
                        "seed": args.seed,
                        "valid_ratio": args.valid_ratio,
                        "train_indices": train_dataset.indices,
                        "valid_indices": valid_dataset.indices,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if world_size > 1:
            if train_size < world_size:
                raise ValueError("training split must contain at least one sample per rank")
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank,
                seed=args.seed, shuffle=True, drop_last=True,
            )
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=device.type == "cuda",
            collate_fn=collate_dailytalk,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
            collate_fn=collate_dailytalk,
        )
        print(f"dataset split: train={train_size} valid={valid_size} seed={args.seed}")
        batches = loader
        target_steps = args.steps

    optimizer.zero_grad(set_to_none=True)
    step = 0
    epoch = 0
    while step < target_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in batches:
            if args.smoke_test:
                streams = batch["streams"].to(device, non_blocking=True).clone()
                acoustic_mask = acoustic_prompt_mask(
                    streams.shape[0], args.text_only_fraction, device=device
                )
                streams[~acoustic_mask, 1:] = -1
            else:
                assert mimi is not None
                # Keep a fixed ratio with random positions inside each batch:
                # by default, batch size 4 means 2 audio-conditioned and 2
                # text-only examples. Text-only examples retain their text
                # targets but have no user-audio prompt or audio loss targets.
                batch_count = batch["waveforms"].shape[0]  # type: ignore[union-attr]
                acoustic_mask = acoustic_prompt_mask(
                    batch_count, args.text_only_fraction, device=device
                )
                streams = online_streams(
                    batch, mimi, device, config, acoustic_mask
                )
            autocast = (
                torch.autocast(device_type="cuda", dtype=dtype)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            sync = (step + 1) % args.grad_accum == 0 or step + 1 == target_steps
            accumulation_size = min(args.grad_accum, target_steps - (step // args.grad_accum) * args.grad_accum)
            sync_context = objective.no_sync() if world_size > 1 and not sync else contextlib.nullcontext()
            with sync_context:
                with autocast:
                    loss, metrics = objective(streams)
                    scaled_loss = loss / accumulation_size
                scaled_loss.backward()

            if sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == 1:
                if world_size > 1:
                    for value in metrics.values():
                        dist.all_reduce(value)
                        value.div_(world_size)
            if primary and (step % args.log_every == 0 or step == 1):
                for name, value in metrics.items():
                    writer.add_scalar(name, value.item(), step)
                writer.add_scalar("optimizer/temporal_lr", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("optimizer/new_modules_lr", optimizer.param_groups[1]["lr"], step)
                audio_log = (
                    f"assistant_audio={metrics['loss/assistant_audio'].item():.4f} "
                    if "loss/assistant_audio" in metrics else ""
                )
                print(
                    f"step={step} loss={metrics['loss/total'].item():.4f} "
                    f"assistant_text={metrics['loss/assistant'].item():.4f} "
                    f"{audio_log}"
                    f"assistant_np_accuracy={metrics['assistant/non_pad_accuracy'].item():.3f} "
                    f"assistant_all_accuracy={metrics['assistant/accuracy'].item():.3f}",
                    flush=True,
                )
            if primary and not args.smoke_test and step % args.save_every == 0:
                save_checkpoint(model, optimizer, step, Path(args.output))
            validate = valid_loader is not None and step % args.valid_every == 0
            if validate:
                barrier()
            if primary and validate:
                valid_metrics = evaluate(
                    model,
                    valid_loader,
                    device,
                    dtype,
                    config.text_padding_id,
                    args.valid_batches,
                    mimi,
                    config,
                )
                for name, value in valid_metrics.items():
                    writer.add_scalar(f"valid/{name}", value, step)
                audio_log = (
                    f"audio_acc={valid_metrics['assistant_audio/accuracy']:.3f}"
                    if "assistant_audio/accuracy" in valid_metrics else ""
                )
                print(
                    f"valid step={step} loss={valid_metrics['loss/total']:.4f} "
                    f"assistant_np_accuracy={valid_metrics['assistant/non_pad_accuracy']:.3f} "
                    f"assistant_all_accuracy={valid_metrics['assistant/accuracy']:.3f} "
                    f"{audio_log}",
                    flush=True,
                )
            if validate:
                barrier()
            if step >= target_steps:
                break
        epoch += 1

    barrier()
    if primary and not args.smoke_test:
        save_checkpoint(model, optimizer, step, Path(args.output))
    if args.smoke_test and world_size > 1:
        for parameter in model.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            if not torch.equal(reference, parameter):
                raise RuntimeError("DDP smoke test: model parameters differ across ranks")
        if primary:
            print("DDP smoke test: all model parameters synchronized", flush=True)
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
