"""Create stereo speaker-aware Whisper alignments for behavior-sd audio."""
# SPDX-FileCopyrightText: Copyright (c) Kyutai Labs
# SPDX-FileContributor: OpenAI
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sphn
import torch
import torchaudio.functional as audio_functional
import whisper_timestamped as whisper


SAMPLE_RATE = 16_000
transcribe = importlib.import_module("whisper_timestamped.transcribe")
old_get_vad_segments = transcribe.get_vad_segments


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Align each stereo speaker channel independently and merge word timestamps."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("/home/ysw/Documents/DB/behavior-sd/validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: beside each WAV as <stem>.alignment.json).",
    )
    parser.add_argument("--model", default="medium", help="Whisper model name (default: medium).")
    parser.add_argument("--language", default="en", help="Forced Whisper language (default: en).")
    parser.add_argument(
        "--assistant-channel",
        type=int,
        choices=(0, 1),
        default=0,
        help="Channel whose words become the training 'alignments' target (default: 0).",
    )
    parser.add_argument(
        "--vad",
        choices=("auto", "none", "auditok", "silero"),
        default="auto",
        help=(
            "Voice activity detector. Auto matches scripts/annotate.py and uses "
            "auditok for audio longer than 10 seconds (default: auto)."
        ),
    )
    parser.add_argument(
        "--keep-silence-in-segments",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help=(
            "Restore this much silence at VAD segment boundaries, matching "
            "scripts/annotate.py (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device. Auto uses CUDA when available.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="CUDA GPU index, for example 0 or 1.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of parallel file shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="This process's zero-based shard index.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files for testing.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing alignment JSON files.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def atomic_json_dump(value: Any, path: Path) -> None:
    """Write JSON atomically so interrupted jobs do not leave partial output."""

    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_metadata(path: Path) -> dict:
    """Load the source dialog metadata associated with an audio file."""

    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing source metadata: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    speakers = metadata.get("speakers")
    if not isinstance(speakers, list) or len(speakers) != 2:
        raise ValueError(f"Expected two metadata speakers in {metadata_path}, got {speakers!r}")
    return metadata


def transcribe_channel(
        model: Any,
        audio: np.ndarray,
        sample_rate: int,
        channel: int,
        speaker: str,
        language: str,
        vad: str,
        keep_silence_in_segments: float) -> tuple[list[list[Any]], list[dict]]:
    """Transcribe one isolated speaker channel into word and Whisper-segment alignments."""

    model_device = next(model.parameters()).device
    waveform = torch.from_numpy(
        np.asarray(audio[channel], dtype=np.float32)
    )[None].to(model_device)
    waveform = audio_functional.resample(waveform, sample_rate, SAMPLE_RATE)
    samples = waveform.cpu().numpy()[0]
    duration = samples.shape[-1] / SAMPLE_RATE

    def new_get_vad_segments(*args, **kwargs):
        vad_segments = old_get_vad_segments(*args, **kwargs)
        outputs = []
        last_end = 0
        boundary_samples = int(SAMPLE_RATE * keep_silence_in_segments)
        for segment in vad_segments:
            outputs.append({
                "start": max(last_end, segment["start"] - boundary_samples),
                "end": segment["end"] + boundary_samples,
            })
            last_end = outputs[-1]["end"]
        return outputs

    if keep_silence_in_segments:
        transcribe.get_vad_segments = new_get_vad_segments

    selected_vad = "auditok" if vad == "auto" and duration > 10 else None
    if vad not in {"auto", "none"}:
        selected_vad = vad
    result = whisper.transcribe(
        model,
        samples,
        language=language,
        vad=selected_vad,
        best_of=5,
        beam_size=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        verbose=None,
    )

    words = []
    segments = []
    for segment in result.get("segments", []):
        segment_words = []
        for word in segment.get("words", []):
            text = word.get("text", "")
            start = float(word["start"])
            end = float(word["end"])
            words.append([text, [start, end], speaker])
            segment_words.append({"text": text, "start": start, "end": end})
        segments.append({
            "text": segment.get("text", "").strip(),
            "timestamp": [float(segment["start"]), float(segment["end"])],
            "speaker": speaker,
            "channel": channel,
            "words": segment_words,
        })
    return words, segments


def process_one(
        audio_path: Path,
        output_path: Path,
        model: Any,
        language: str,
        vad: str,
        assistant_channel: int,
        keep_silence_in_segments: float) -> None:
    """Align both stereo channels and write one merged annotation."""

    gc.collect()
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.empty_cache()
    metadata = load_metadata(audio_path)
    audio, sample_rate = sphn.read(audio_path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"Expected [2, samples] stereo audio in {audio_path}, got {audio.shape}")
    duration = audio.shape[-1] / sample_rate
    if duration > 3600 * 4:
        raise RuntimeError("File is too long for now.")

    channel_alignments = {}
    segments = []
    for channel, speaker in enumerate(metadata["speakers"]):
        channel_words, channel_segments = transcribe_channel(
            model,
            audio,
            sample_rate,
            channel,
            speaker,
            language,
            vad,
            keep_silence_in_segments,
        )
        channel_alignments[channel] = channel_words
        segments.extend(channel_segments)

    alignments = channel_alignments[assistant_channel]
    all_alignments = [item for words in channel_alignments.values() for item in words]
    alignments.sort(key=lambda item: (item[1][0], item[1][1], item[2]))
    all_alignments.sort(key=lambda item: (item[1][0], item[1][1], item[2]))
    segments.sort(key=lambda item: (item["timestamp"][0], item["timestamp"][1], item["speaker"]))
    sentence_alignments = []
    for utterance in metadata.get("utterances", []):
        speaker = utterance["speaker"]
        sentence_alignments.append([
            utterance["tts_text"],
            [float(utterance["start_time"]), float(utterance["end_time"])],
            speaker,
        ])
        other_speaker = next((name for name in metadata["speakers"] if name != speaker), speaker)
        for backchannel in utterance.get("backchannels", []):
            sentence_alignments.append([
                backchannel["tts_text"],
                [float(backchannel["start_time"]), float(backchannel["end_time"])],
                other_speaker,
            ])
    sentence_alignments.sort(key=lambda item: (item[1][0], item[1][1], item[2]))
    output = {
        "audio": str(audio_path),
        "sample_rate": sample_rate,
        "duration": duration,
        "speakers": metadata["speakers"],
        "assistant_channel": assistant_channel,
        "assistant_speaker": metadata["speakers"][assistant_channel],
        "alignments": alignments,
        "channel_alignments": channel_alignments,
        "all_alignments": all_alignments,
        "sentence_alignments": sentence_alignments,
        "segments": segments,
        "source_utterances": metadata.get("utterances", []),
    }
    atomic_json_dump(output, output_path)


def resolve_device(requested: str, gpu: Optional[int]) -> str:
    """Resolve and validate the inference device."""

    if gpu is not None:
        if requested == "cpu":
            raise ValueError("--gpu cannot be combined with --device cpu")
        if gpu < 0:
            raise ValueError(f"--gpu must be non-negative, got {gpu}")
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu was specified, but torch.cuda.is_available() is false")
        if gpu >= torch.cuda.device_count():
            raise ValueError(f"--gpu {gpu} is unavailable; detected {torch.cuda.device_count()} CUDA devices")
        torch.cuda.set_device(gpu)
        return f"cuda:{gpu}"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
    return requested


def main(args: argparse.Namespace) -> None:
    """Load Whisper once and resumably process every WAV in the input folder."""

    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be at least 1, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )
    device = resolve_device(args.device, args.gpu)
    audio_paths = sorted(args.input_dir.rglob("*.wav"))
    total_files = len(audio_paths)
    audio_paths = audio_paths[args.shard_index::args.num_shards]
    if args.limit is not None:
        audio_paths = audio_paths[:args.limit]
    if not audio_paths:
        raise FileNotFoundError(f"No WAV files found in {args.input_dir}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Loading Whisper {args.model!r} on {device}; shard={args.shard_index}/{args.num_shards}; "
        f"files={len(audio_paths)}/{total_files}"
    )
    model = whisper.load_model(args.model, device=torch.device(device))

    completed = skipped = failed = 0
    for index, audio_path in enumerate(audio_paths, 1):
        if args.output_dir is None:
            output_parent = audio_path.parent
        else:
            output_parent = args.output_dir / audio_path.relative_to(args.input_dir).parent
            output_parent.mkdir(parents=True, exist_ok=True)
        output_path = output_parent / f"{audio_path.stem}.alignment.json"
        error_path = output_parent / f"{audio_path.stem}.alignment.json.err"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            process_one(
                audio_path,
                output_path,
                model,
                args.language,
                args.vad,
                args.assistant_channel,
                args.keep_silence_in_segments,
            )
            error_path.unlink(missing_ok=True)
            completed += 1
            print(f"[{index}/{len(audio_paths)}] wrote {output_path.name}")
        except Exception:
            failed += 1
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{index}/{len(audio_paths)}] failed {audio_path.name}: see {error_path}")
            if "cuda" in error_path.read_text(encoding="utf-8").lower():
                raise
    print(f"Done: completed={completed}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main(parse_args())
