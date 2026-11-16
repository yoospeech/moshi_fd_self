"""Build a duration manifest from local LibriTTS WAV headers (no audio copy)."""

import argparse
import json
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="one or more WAV directories separated by commas")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    parts = args.root.split(",")
    if any(not part.strip() for part in parts):
        parser.error("folder list contains an empty path")
    roots = list(dict.fromkeys(Path(part.strip()).expanduser().resolve() for part in parts))
    for root in roots:
        if not root.is_dir():
            parser.error(f"not a directory: {root}")
    # Resolve paths and retain their first source root to avoid duplicates
    # when directories repeat or a parent and child are both supplied.
    sources = {}
    for root in roots:
        for path in sorted(root.rglob("*.wav")):
            sources.setdefault(path.resolve(), (root, path.relative_to(root).parts[0]))
    paths = sorted(sources)
    if not paths:
        parser.error("no WAV files found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    totals = {}
    with temporary.open("x", encoding="utf-8") as handle:
        for path in paths:
            info = sf.info(path)
            if info.channels != 1 or info.frames <= 0:
                raise ValueError(f"expected nonempty mono WAV: {path}")
            duration = info.frames / info.samplerate
            row = {"path": str(path), "duration": duration}
            sidecar = path.with_suffix(".json")
            if sidecar.is_file():
                row["alignment_path"] = str(sidecar)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            root, source_split = sources[path]
            split = source_split if len(roots) == 1 else str(root)
            stats = totals.setdefault(split, {"files": 0, "seconds": 0.0})
            stats["files"] += 1
            stats["seconds"] += duration
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "splits": totals}, indent=2))


if __name__ == "__main__":
    main()
