#!/usr/bin/env python3
"""Create a deterministic, duration-capped DuplexChat training manifest.

DuplexChat publishes clip metadata, not the podcast audio itself.  Use the
result of this script as the input to an episode downloader so it only fetches
the selected clips' source episodes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable


MANIFEST_NAMES = {
    "en": "duplexchat_manifest_en.jsonl.gz",
    "ja": "duplexchat_manifest_ja.jsonl.gz",
}


def stable_fraction(seed: int, language: str, line: str) -> float:
    """Return a repeatable pseudo-random value in [0, 1) for a row."""
    digest = hashlib.blake2b(
        f"{seed}:{language}:".encode() + line.encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def rows(path: Path) -> Iterable[str]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        yield from source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-hours", type=float, default=1000.0)
    parser.add_argument("--languages", nargs="+", choices=sorted(MANIFEST_NAMES),
                        default=["en", "ja"])
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument(
        "--oversample-factor", type=float, default=1.25,
        help="Candidate rate above the requested duration; must be at least 1.",
    )
    args = parser.parse_args()
    if args.target_hours <= 0 or args.oversample_factor < 1:
        parser.error("--target-hours must be positive and --oversample-factor >= 1")

    counts_path = args.source_dir / "manifest_counts.json"
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    available_hours = {language: float(counts[language]["hours"]) for language in args.languages}
    total_available = sum(available_hours.values())
    if args.target_hours > total_available:
        parser.error(f"requested {args.target_hours:g} hours, but only {total_available:.2f} are available")

    # Preserve the language balance of the selected source manifests.  The
    # hash filter makes selection independent of input ordering and repeatable.
    quotas = {
        language: args.target_hours * available_hours[language] / total_available
        for language in args.languages
    }
    rates = {
        language: min(1.0, args.oversample_factor * quotas[language] / available_hours[language])
        for language in args.languages
    }
    selected_hours = dict.fromkeys(args.languages, 0.0)
    selected_rows = dict.fromkeys(args.languages, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Do not expose a partial manifest if this long streaming pass is stopped.
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False,
        prefix=args.output.name + ".", suffix=".tmp",
    ) as output:
        temporary_output = Path(output.name)
        for language in args.languages:
            source_path = args.source_dir / MANIFEST_NAMES[language]
            for line in rows(source_path):
                if stable_fraction(args.seed, language, line) >= rates[language]:
                    continue
                row = json.loads(line)
                duration_hours = float(row["duration_sec"]) / 3600
                # Never exceed the requested duration.  A small remainder may
                # be unavoidable because a complete dialogue clip is atomic.
                if selected_hours[language] + duration_hours > quotas[language]:
                    continue
                output.write(line)
                selected_hours[language] += duration_hours
                selected_rows[language] += 1

    summary = {
        "target_hours": args.target_hours,
        "selected_hours": sum(selected_hours.values()),
        "seed": args.seed,
        "languages": {
            language: {"rows": selected_rows[language], "hours": selected_hours[language]}
            for language in args.languages
        },
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    temporary_output.replace(args.output)
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
