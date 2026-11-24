#!/usr/bin/env python3
"""Export word-aligned mono post-training samples as stereo WAV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from full_duplex.dataset import MonoPostTrainingDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=45.0)
    args = parser.parse_args()
    if args.max_items < 1:
        parser.error("--max-items must be positive")

    dataset = MonoPostTrainingDataset(
        args.manifest,
        None,  # Audio-only export: training-tokenizer collision filtering is not applicable.
        sample_rate=args.sample_rate,
        text_padding_id=0,
        text_epad_id=1,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
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
