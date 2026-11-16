#!/usr/bin/env python3
"""Export word-aligned mono-continuation samples as stereo WAV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from full_duplex.dataset import MonoContinuationDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=45.0)
    parser.add_argument("--min-prefix-seconds", type=float, default=1.0)
    parser.add_argument("--min-suffix-seconds", type=float, default=1.0)
    parser.add_argument("--min-gap-seconds", type=float, default=0.1)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_items < 1:
        parser.error("--max-items must be positive")

    dataset = MonoContinuationDataset(
        args.manifest,
        None,  # Audio-only export: training-tokenizer collision filtering is not applicable.
        sample_rate=args.sample_rate,
        text_padding_id=0,
        text_epad_id=1,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_prefix_seconds=args.min_prefix_seconds,
        min_suffix_seconds=args.min_suffix_seconds,
        gap_seconds=(args.min_gap_seconds, args.max_gap_seconds),
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(args.max_items, len(dataset))):
        user, assistant, alignments = dataset.render_stereo(index)
        entry = dataset.entries[index]
        stem = f"{index:06d}_{Path(entry['path']).stem}"
        # soundfile expects samples x channels; project convention is
        # channel 0=assistant and channel 1=user.
        stereo = torch.stack([assistant, user], dim=1).numpy()
        sf.write(args.output_dir / f"{stem}.wav", stereo, args.sample_rate, subtype="PCM_16")
        sidecar = {
            "source_path": str(entry["path"]),
            "split_seconds": entry["split_seconds"],
            "gap_seconds": entry["gap_seconds"],
            "assistant_channel": 0,
            "user_channel": 1,
            "alignments": alignments,
        }
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {stem}.wav")


if __name__ == "__main__":
    main()
