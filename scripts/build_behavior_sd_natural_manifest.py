#!/usr/bin/env python3
"""Build a NaturalMonologueDataset manifest from Behavior-SD JSON/WAV pairs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import soundfile as sf


def assistant_texts(metadata: dict, assistant_channel: int) -> list[dict[str, object]]:
    """Return all main utterances and listener backchannels on one channel."""
    texts: list[dict[str, object]] = []
    for utterance in metadata.get("utterances", []):
        speaker = int(utterance["speaker_idx"])
        if speaker == assistant_channel:
            text = str(utterance.get("tts_text", "")).strip()
            if text:
                texts.append({"start": float(utterance["start_time"]), "text": text})
        else:
            # A backchannel nested under speaker N is spoken by the listener.
            for backchannel in utterance.get("backchannels", []):
                text = str(backchannel.get("tts_text", "")).strip()
                if text:
                    texts.append({
                        "start": float(backchannel["start_time"]),
                        "text": text,
                    })
    texts.sort(key=lambda item: float(item["start"]))
    return texts


def user_backchannel_intervals(
    metadata: dict, assistant_channel: int
) -> list[list[float]]:
    """Return user backchannels nested under assistant utterances."""
    return [
        [float(backchannel["start_time"]), float(backchannel["end_time"])]
        for utterance in metadata.get("utterances", [])
        if int(utterance["speaker_idx"]) == assistant_channel
        for backchannel in utterance.get("backchannels", [])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Behavior-SD split directory")
    parser.add_argument("output", type=Path, help="output JSONL path")
    parser.add_argument("--language", default="en")
    parser.add_argument("--assistant-channel", type=int, default=0, choices=(0, 1))
    parser.add_argument("--user-channel", type=int, default=1, choices=(0, 1))
    parser.add_argument("--limit", type=int, default=0, help="0 writes every valid pair")
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    if args.assistant_channel == args.user_channel:
        parser.error("assistant and user channels must differ")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if output.exists() and not args.force:
        parser.error(f"output already exists: {output}; pass --force to replace it")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    rejected = 0
    hours = 0.0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent,
        prefix=output.name + ".", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            for audio_path in sorted(root.rglob("*.wav")):
                metadata_path = audio_path.with_suffix(".json")
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if Path(str(metadata["file_name"])).stem != audio_path.stem:
                        raise ValueError("metadata file_name stem does not match audio")
                    utterances = assistant_texts(metadata, args.assistant_channel)
                    if not utterances:
                        raise ValueError("no assistant utterances")
                    info = sf.info(audio_path)
                    if info.channels != 2 or info.frames <= 0:
                        raise ValueError("expected nonempty stereo audio")
                    duration = info.frames / info.samplerate
                except (KeyError, TypeError, ValueError, OSError, RuntimeError, json.JSONDecodeError):
                    rejected += 1
                    continue
                row = {
                    "id": metadata_path.stem,
                    "path": str(audio_path.resolve()),
                    "duration": duration,
                    "language": args.language,
                    "user_channel": args.user_channel,
                    "assistant_channel": args.assistant_channel,
                    "assistant_texts": utterances,
                    "user_backchannel_intervals": user_backchannel_intervals(
                        metadata, args.assistant_channel
                    ),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                hours += duration / 3600
                if args.limit and written >= args.limit:
                    break
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    if not written:
        temporary.unlink(missing_ok=True)
        raise SystemExit("no valid Behavior-SD JSON/WAV pairs found")
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "rows": written,
        "hours": hours,
        "rejected_json": rejected,
    }, indent=2))


if __name__ == "__main__":
    main()
