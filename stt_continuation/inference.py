"""Reference inference with forced user audio and assistant text continuation."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import fields
from pathlib import Path

import sphn
import torch
from transformers import AutoTokenizer

from .model import STTContinuationConfig, SmolLM2MoshiSTTContinuation
from moshi.models import loaders as moshi_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="defaults to config.json beside checkpoint")
    parser.add_argument("--audio", required=True, help="stereo or mono user-audio input")
    parser.add_argument("--user-channel", type=int, default=1)
    parser.add_argument("--model", help="override temporal_model in config")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the checkpoint setting for custom Hugging Face model code",
    )
    parser.add_argument("--mimi-checkpoint")
    parser.add_argument("--mimi-repo", default=moshi_loaders.DEFAULT_REPO)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="language-model device; auto uses CUDA when available (default: auto)",
    )
    parser.add_argument("--mimi-device")
    parser.add_argument("--tail-seconds", type=float, default=10.0)
    parser.add_argument("--text-temperature", type=float, default=0.7)
    parser.add_argument("--text-top-k", type=int, default=25)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=250)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--output", default="inference_text.json")
    parser.add_argument(
        "--audio-output",
        help="assistant WAV path; defaults to the JSON output with a .wav suffix",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> STTContinuationConfig:
    checkpoint = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else checkpoint.parent / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("text_context_mode") != "moshi":
        raise ValueError(
            "Checkpoint is not a Moshi-style assistant-text checkpoint."
        )
    names = {field.name for field in fields(STTContinuationConfig)}
    values = {key: value for key, value in raw.items() if key in names}
    if args.model:
        values["temporal_model"] = args.model
    trust_remote_code = getattr(args, "trust_remote_code", None)
    if trust_remote_code is not None:
        values["trust_remote_code"] = trust_remote_code
    elif args.model and "lfm3" in args.model.lower():
        values["trust_remote_code"] = True
    config = STTContinuationConfig(**values)
    expected_delays = [0] + ([0] + [1] * (config.num_codebooks - 1)) * 2
    if config.delays != expected_delays:
        raise ValueError(
            "Checkpoint delay pattern is incompatible with Moshi-style inference: "
            f"expected {expected_delays}, got {config.delays}"
        )
    return config


def sample_text(logits: torch.Tensor, *, temperature: float, top_k: int, greedy: bool) -> int:
    if greedy or temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    scores = logits.float() / temperature
    if 0 < top_k < scores.shape[-1]:
        values, indices = torch.topk(scores, top_k, dim=-1)
        sampled = torch.multinomial(torch.softmax(values, dim=-1), 1)
        return int(indices.gather(-1, sampled).item())
    return int(torch.multinomial(torch.softmax(scores, dim=-1), 1).item())


def load_user_waveform(path: Path, channel: int, sample_rate: int) -> torch.Tensor:
    wav, _ = sphn.read(path, sample_rate=sample_rate)
    if wav.ndim == 1:
        if channel != 0:
            print(f"warning: mono input; using channel 0 instead of requested {channel}")
        selected = wav
    else:
        if channel >= wav.shape[0]:
            raise ValueError(f"{path}: missing channel {channel}; shape={wav.shape}")
        selected = wav[channel]
    return torch.from_numpy(selected).float().clamp(-1.0, 1.0)


@torch.inference_mode()
def force_audio_sample_text(
    model: SmolLM2MoshiSTTContinuation,
    audio_codes: torch.Tensor,
    config: STTContinuationConfig,
    *,
    temperature: float,
    top_k: int,
    greedy: bool,
) -> list[int]:
    """Force user audio and sample the assistant continuation text."""
    if audio_codes.shape[0] != 1:
        raise ValueError("reference inference currently supports batch size 1")
    if audio_codes.shape[1] != config.num_codebooks:
        raise ValueError(
            f"Mimi returned {audio_codes.shape[1]} codebooks; expected {config.num_codebooks}"
        )

    device = model.device
    audio_codes = audio_codes.to(device)
    initial = model._get_initial_token()
    pattern = initial
    generated = [[]]
    total_pattern_steps = audio_codes.shape[-1] + max(config.delays)

    for step in range(total_pattern_steps):
        _, text_logits = model.forward_text(pattern)
        frame = torch.empty(
            (1, model.num_codebooks, 1), dtype=torch.long, device=device
        )

        for channel, delay in enumerate(config.delays[: model.num_text_streams]):
            if step < delay:
                frame[:, channel, 0] = model.text_initial_token_id
            elif step - delay >= audio_codes.shape[-1]:
                frame[:, channel, 0] = config.text_padding_id
            else:
                token = sample_text(
                    text_logits[0, channel, -1],
                    temperature=temperature,
                    top_k=top_k,
                    greedy=greedy,
                )
                frame[:, channel, 0] = token
                generated[channel].append(token)

        # This text-only reference path does not synthesize assistant audio;
        # its feedback stream remains absent. User audio is force-fed solely
        # into the user-conditioning stream.
        for codebook in range(config.num_codebooks):
            delay = config.delays[codebook + model.audio_offset]
            source = step - delay
            assistant_index = codebook + model.audio_offset
            frame[:, assistant_index, 0] = (
                model.initial_token_id if source < 0 else model.zero_token_id
            )
            user_index = codebook + model.user_audio_offset
            if 0 <= source < audio_codes.shape[-1]:
                frame[:, user_index, 0] = audio_codes[:, codebook, source]
            elif source < 0:
                frame[:, user_index, 0] = model.initial_token_id
            else:
                # Match the absent-audio tail used by online_streams in training.
                frame[:, user_index, 0] = model.zero_token_id

        pattern = torch.cat([pattern, frame], dim=-1)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == total_pattern_steps:
            print(f"inference frame {step + 1}/{total_pattern_steps}", flush=True)
    return generated[0]


@torch.inference_mode()
def generate_assistant(
    model: SmolLM2MoshiSTTContinuation,
    user_audio_codes: torch.Tensor,
    config: STTContinuationConfig,
    *,
    text_temperature: float,
    text_top_k: int,
    audio_temperature: float,
    audio_top_k: int,
    greedy: bool,
) -> tuple[list[int], torch.Tensor]:
    """Offline causal generation of assistant text and Mimi audio."""
    if user_audio_codes.shape[:2] != (1, config.num_codebooks):
        raise ValueError(
            "user_audio_codes must have shape "
            f"[1, {config.num_codebooks}, frames]"
        )
    device = model.device
    user_audio_codes = user_audio_codes.to(device)
    frames = user_audio_codes.shape[-1]
    tail = max(config.delays)
    streams = torch.full(
        (1, model.num_codebooks, frames + tail),
        -1,
        dtype=torch.long,
        device=device,
    )
    streams[:, 0] = config.text_padding_id
    streams[:, model.user_audio_offset :, :frames] = user_audio_codes
    generated_text: list[int] = []

    for frame_index in range(frames):
        text_logits, _, _, _ = model.forward_training(streams, predict_audio=False)
        text_token = sample_text(
            text_logits[0, 0, frame_index],
            temperature=text_temperature,
            top_k=text_top_k,
            greedy=greedy,
        )
        streams[0, 0, frame_index] = text_token
        generated_text.append(text_token)

        # Depth is autoregressive over codebooks at the current audio frame.
        # Re-running the offline forward after each sampled codebook supplies
        # the just-generated q(k-1) token to the q(k) Depth input.
        for codebook in range(config.num_codebooks):
            _, _, audio_logits, _ = model.forward_training(streams)
            assert audio_logits is not None
            audio_token = sample_text(
                audio_logits[0, codebook, frame_index],
                temperature=audio_temperature,
                top_k=audio_top_k,
                greedy=greedy,
            )
            streams[0, model.audio_offset + codebook, frame_index] = audio_token

        if (
            frame_index == 0
            or (frame_index + 1) % 25 == 0
            or frame_index + 1 == frames
        ):
            print(
                f"generated assistant frame {frame_index + 1}/{frames}",
                flush=True,
            )

    assistant_codes = streams[
        :, model.audio_offset : model.user_audio_offset, :frames
    ]
    return generated_text, assistant_codes


def render_tokens(
    token_ids: list[int], tokenizer: AutoTokenizer, config: STTContinuationConfig
) -> tuple[str, list[dict[str, object]]]:
    special = {
        config.text_padding_id: "PAD",
        config.text_epad_id: "EPAD",
    }
    events: list[dict[str, object]] = []
    lexical: list[int] = []

    for frame, token_id in enumerate(token_ids):
        if token_id in special:
            label = special[token_id]
        elif 0 <= token_id < len(tokenizer):
            label = tokenizer.decode([token_id], skip_special_tokens=False)
            lexical.append(token_id)
        else:
            label = f"UNKNOWN_{token_id}"
        events.append({"frame": frame, "time": frame / config.frame_rate, "id": token_id, "token": label})
    return tokenizer.decode(lexical, skip_special_tokens=False), events


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = load_config(args)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")
    device = torch.device(device_name)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    print(f"loading model on {device} with dtype={dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.temporal_model,
        trust_remote_code=config.trust_remote_code,
    )
    model = SmolLM2MoshiSTTContinuation(config, dtype=dtype, device=device)
    print(f"memory-mapping checkpoint {args.checkpoint}", flush=True)
    package = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    print("loading model weights from checkpoint", flush=True)
    model.load_state_dict(package["model"], strict=True)
    checkpoint_step = package.get("step")
    del package
    gc.collect()
    model.eval()
    print("model checkpoint loaded", flush=True)

    mimi_checkpoint = args.mimi_checkpoint
    if mimi_checkpoint is None:
        mimi_checkpoint = str(moshi_loaders.hf_get(moshi_loaders.MIMI_NAME, args.mimi_repo))
    mimi_device = torch.device(args.mimi_device) if args.mimi_device else device
    mimi = moshi_loaders.get_mimi(
        mimi_checkpoint, device=mimi_device, num_codebooks=config.num_codebooks
    )
    mimi.eval()

    waveform = load_user_waveform(Path(args.audio), args.user_channel, mimi.sample_rate)
    tail_samples = max(0, round(args.tail_seconds * mimi.sample_rate))
    if tail_samples:
        waveform = torch.cat([waveform, torch.zeros(tail_samples)])
    audio_codes = mimi.encode(waveform[None, None].to(mimi_device))
    continuation_token_ids, assistant_codes = generate_assistant(
        model,
        audio_codes,
        config,
        text_temperature=args.text_temperature,
        text_top_k=args.text_top_k,
        audio_temperature=args.audio_temperature,
        audio_top_k=args.audio_top_k,
        greedy=args.greedy,
    )
    assistant_waveform = mimi.decode(assistant_codes.to(mimi_device))
    continuation_text, continuation_events = render_tokens(
        continuation_token_ids, tokenizer, config
    )
    payload = {
        "audio": str(Path(args.audio)),
        "user_channel": args.user_channel,
        "checkpoint_step": checkpoint_step,
        "stream_order": [
            "assistant_text",
            "assistant_semantic",
            *[f"assistant_acoustic_{index}" for index in range(1, config.num_codebooks)],
            "user_semantic",
            *[f"user_acoustic_{index}" for index in range(1, config.num_codebooks)],
        ],
        "delay_pattern": config.delays,
        "decoding": {
            "greedy": args.greedy,
            "temperature": args.text_temperature,
            "top_k": args.text_top_k,
            "audio_temperature": args.audio_temperature,
            "audio_top_k": args.audio_top_k,
            "seed": args.seed,
        },
        "continuation_text": continuation_text,
        "transcript": {
            "assistant_continuation": continuation_text,
        },
        "turns": [{
                "speaker": "assistant",
                "channel": "continuation",
                "text": continuation_text,
                "start_time": next(
                    (event["time"] for event in continuation_events if event["token"] not in {"PAD", "EPAD"}),
                    None,
                ),
                "end_time": next(
                    (event["time"] for event in reversed(continuation_events) if event["token"] not in {"PAD", "EPAD"}),
                    None,
                ),
            }
        ],
        "channels": {
            "continuation": {"text": continuation_text, "events": continuation_events},
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    audio_output = Path(args.audio_output) if args.audio_output else output.with_suffix(".wav")
    audio_output.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(
        audio_output,
        assistant_waveform[0, 0].float().clamp(-1, 1).cpu().numpy(),
        mimi.sample_rate,
    )
    print("\nASSISTANT CONTINUATION")
    print(continuation_text or "(empty)")
    print(f"saved {output}")
    print(f"saved {audio_output}")


if __name__ == "__main__":
    main()
