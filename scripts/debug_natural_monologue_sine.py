"""Visualize NaturalMonologueDataset sine conditioning on one Behavior-SD item."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer

from full_duplex.natural_monologue_dataset import (
    NaturalMonologueDataset,
    write_natural_monologue_debug,
)
from full_duplex.tokenizer import add_control_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", default="models/LFM3-1.7B")
    parser.add_argument("--output-prefix", type=Path, default=Path("/tmp/natural_sine_debug"))
    parser.add_argument("--assistant-channel", type=int, default=0)
    parser.add_argument("--user-channel", type=int, default=1)
    parser.add_argument("--wait-token-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--spk-token-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--hear-token-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sine-frequency", type=float, default=440.0)
    parser.add_argument("--sine-amplitude", type=float, default=0.5)
    parser.add_argument("--mimi-checkpoint", type=Path)
    parser.add_argument("--num-codebooks", type=int, default=4)
    return parser.parse_args()


def behavior_sd_entry(metadata_path: Path, assistant_channel: int, user_channel: int):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audio_path = metadata_path.with_suffix(".wav")
    if not audio_path.is_file():
        raise FileNotFoundError(f"same-stem WAV not found: {audio_path}")
    info = sf.info(audio_path)
    monologues = [
        {"start": float(item["start_time"]), "text": item["tts_text"]}
        for item in metadata["utterances"]
        if int(item["speaker_idx"]) == assistant_channel
    ]
    if not monologues:
        raise ValueError(f"{metadata_path}: no utterances for speaker_idx={assistant_channel}")
    return {
        "id": metadata_path.stem,
        "path": str(audio_path),
        "duration": info.frames / info.samplerate,
        "language": metadata.get("language", "unknown"),
        "user_channel": user_channel,
        "assistant_channel": assistant_channel,
        "assistant_texts": monologues,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    control_ids = add_control_tokens(tokenizer)
    explicit_ids = {
        "wait": args.wait_token_id,
        "spk": args.spk_token_id,
        "hear": args.hear_token_id,
    }
    for name, value in explicit_ids.items():
        if value is not None and value != control_ids[name]:
            raise ValueError(f"{name} ID {value} does not match tokenizer ID {control_ids[name]}")
    entry = behavior_sd_entry(args.metadata, args.assistant_channel, args.user_channel)
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "sample.jsonl"
        manifest.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
        dataset = NaturalMonologueDataset(
            manifest,
            tokenizer,
            wait_token_id=control_ids["wait"],
            spk_token_id=control_ids["spk"],
            hear_token_id=control_ids["hear"],
            sample_rate=24_000,
            frame_rate=12.5,
            min_duration=0,
            min_text_tokens=0,
            enable_sine_conditioning=True,
            sine_frequency=args.sine_frequency,
            sine_amplitude=args.sine_amplitude,
        )
        item = dataset[0]
        stats = write_natural_monologue_debug(item, args.output_prefix, dataset)
        stats["audio_samples"] = item["assistant_waveform"].numel()
        stats["text_frames"] = item["assistant_text_frame_length"]
        stats["audio_frames"] = item["audio_frame_length"]
        if args.mimi_checkpoint:
            moshi_source = (
                Path(__file__).resolve().parent.parent / "thirdparty" / "moshi" / "moshi"
            )
            if str(moshi_source) not in sys.path:
                sys.path.insert(0, str(moshi_source))
            from moshi.models import loaders as moshi_loaders

            mimi = moshi_loaders.get_mimi(
                args.mimi_checkpoint, device="cpu", num_codebooks=args.num_codebooks
            )
            pair = torch.stack([
                item["original_assistant_waveform"],
                item["assistant_waveform"],
            ])[:, None]
            with torch.inference_mode():
                codes = mimi.encode(pair)
            stats["mimi_code_shape"] = list(codes.shape)
            stats["mimi_changed_tokens"] = int((codes[0] != codes[1]).sum())
            stats["mimi_changed_frames"] = int((codes[0] != codes[1]).any(dim=0).sum())
        stats_path = args.output_prefix.with_suffix(".json")
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
