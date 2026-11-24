"""Duplex user audio and aligned assistant text/audio targets."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import sphn
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TextAlignmentCollision(ValueError):
    """Two lexical tokens cannot fit on the same fixed text timeline."""


class RecoverableSampleError(ValueError):
    """A lazily prepared sample is unusable and may be skipped."""


def discover_speech_entries(
    folders: str,
    *,
    max_files: int = 0,
    include_unannotated: bool = False,
    audio_mode: str = "mono",
    user_channel: int = 1,
    assistant_channel: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Discover WAVs from comma-separated ``folder:language`` inputs.

    Mono audio uses ``<stem>.json`` sidecars. Stereo audio uses
    ``<stem>.alignment.json`` so source metadata can remain at ``<stem>.json``.
    Without include_unannotated, only WAVs with alignment sidecars are kept.
    """
    import soundfile as sf

    if audio_mode not in {"mono", "stereo"}:
        raise ValueError(f"unsupported audio mode: {audio_mode!r}")
    if max_files < 0:
        raise ValueError("max_files must be non-negative")
    parts = folders.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError("folder list contains an empty path")
    sources: list[tuple[Path, str]] = []
    for part in parts:
        value = part.strip()
        if ":" not in value:
            raise ValueError(
                f"missing language for folder: {value!r}; use /path/to/data:en"
            )
        folder, language = value.rsplit(":", 1)
        if not folder or not language.strip():
            raise ValueError(f"invalid folder:language input: {value!r}")
        source = (Path(folder).expanduser().resolve(), language.strip())
        if source not in sources:
            sources.append(source)
    if not sources:
        raise ValueError("at least one input folder is required")
    for root, _language in sources:
        if not root.is_dir():
            raise ValueError(f"not a directory: {root}")
    paths: dict[Path, str] = {}
    for root, language in sources:
        for path in root.rglob("*.wav"):
            resolved = path.resolve()
            previous = paths.get(resolved)
            if previous is not None and previous != language:
                raise ValueError(
                    f"conflicting languages for {resolved}: {previous!r} and {language!r}"
                )
            paths[resolved] = language
    entries: list[dict[str, Any]] = []
    stats = {"wav": len(paths), "annotated": 0, "unannotated": 0, "invalid": 0}
    for path, language in sorted(paths.items()):
        sidecar = path.with_suffix(
            ".json" if audio_mode == "mono" else ".alignment.json"
        )
        if not sidecar.is_file():
            stats["unannotated"] += 1
            if not include_unannotated:
                continue
        try:
            info = sf.info(path)
            expected_channels = 1 if audio_mode == "mono" else 2
            if info.channels != expected_channels or info.frames <= 0:
                stats["invalid"] += 1
                continue
            entries.append(
                {
                    "path": str(path),
                    "duration": info.frames / info.samplerate,
                    "alignment_path": str(sidecar),
                    "language": language,
                    "audio_mode": audio_mode,
                    "user_channel": user_channel,
                    "assistant_channel": assistant_channel,
                }
            )
            stats["annotated"] += int(sidecar.is_file())
            if max_files and len(entries) >= max_files:
                break
        except (RuntimeError, OSError):
            stats["invalid"] += 1
    return entries, stats


class StereoDialogueDataset(Dataset[dict[str, Any]]):
    """Read stereo dialogues and assistant word-level alignments.

    Audio tokenization intentionally happens after collation so frozen Mimi can
    run online on the training device.
    """

    def __init__(
        self,
        manifest: str | Path | None,
        tokenizer: PreTrainedTokenizerBase,
        *,
        sample_rate: int = 24000,
        frame_rate: float = 12.5,
        text_padding_id: int,
        text_epad_id: int,
        user_channel: int = 1,
        assistant_channel: int = 0,
        min_duration: float = 5.0,
        max_duration: float = 45.0,
        min_text_tokens: int = 5,
        min_speech_coverage: float = 0.05,
        max_speech_coverage: float = 0.8,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.text_padding_id = text_padding_id
        self.text_epad_id = text_epad_id
        self.user_channel = user_channel
        self.assistant_channel = assistant_channel
        if min_text_tokens < 0:
            raise ValueError("min_text_tokens must be non-negative")
        self.min_text_tokens = min_text_tokens
        with self.manifest.open(encoding="utf-8") as handle:
            raw_entries = [json.loads(line) for line in handle if line.strip()]
        self.original_size = len(raw_entries)
        self.entries = []
        self.rejected: dict[str, int] = {
            "duration": 0,
            "text_tokens": 0,
            "alignment": 0,
            "speech_coverage": 0,
            "text_collision": 0,
        }
        for entry in raw_entries:
            duration = float(entry["duration"])
            if not min_duration <= duration <= max_duration:
                self.rejected["duration"] += 1
                continue
            audio_path = Path(entry["path"])
            if not audio_path.is_absolute():
                audio_path = self.root / audio_path
            # The assistant-side transcript is the response target, i.e.
            # the model's inner-monologue text stream.  The current sidecar
            # schema stores its timestamped words under ``alignments``.
            sidecar = json.loads(audio_path.with_suffix(".json").read_text(encoding="utf-8"))
            inner_monologue_alignments = sidecar["alignments"]
            text_token_count = self._lexical_token_count(inner_monologue_alignments)
            if text_token_count < self.min_text_tokens:
                self.rejected["text_tokens"] += 1
                continue
            intervals = []
            invalid = False
            for _word, timestamps, _speaker in inner_monologue_alignments:
                start, end = map(float, timestamps)
                if start < 0 or end < start or end > duration + 0.25:
                    invalid = True
                    break
                intervals.append((max(0.0, start), min(duration, end)))
            if invalid:
                self.rejected["alignment"] += 1
                continue
            intervals.sort()
            merged: list[list[float]] = []
            for start, end in intervals:
                if not merged or start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            speech_coverage = sum(end - start for start, end in merged) / duration
            if not min_speech_coverage <= speech_coverage <= max_speech_coverage:
                self.rejected["speech_coverage"] += 1
                continue
            try:
                self._aligned_text_stream(
                    inner_monologue_alignments,
                    max(1, math.ceil(duration * self.frame_rate)),
                )
            except TextAlignmentCollision:
                self.rejected["text_collision"] += 1
                continue
            self.entries.append(
                {
                    **entry,
                    "language": entry.get("language", sidecar.get("language", "unknown")),
                    "inner_monologue_alignments": inner_monologue_alignments,
                }
            )
        self.filtered_hours = sum(float(entry["duration"]) for entry in self.entries) / 3600
        if not self.entries:
            raise ValueError(f"empty manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    def _word_tokens(self, word: str) -> list[int]:
        # Preserve the tokenizer's word-boundary form for every aligned word.
        return self.tokenizer.encode(" " + word, add_special_tokens=False)

    def _lexical_token_count(self, alignments: list[Any]) -> int:
        return sum(len(self._word_tokens(str(word))) for word, _, _ in alignments)

    def _aligned_text_stream(
        self, alignments: list[Any], frame_length: int
    ) -> torch.Tensor:
        """Build the assistant response timeline on the audio clock."""
        def render() -> torch.Tensor:
            tokens = [self.text_padding_id] * frame_length
            # Track lexical pieces separately. PAD and EPAD may be replaced by
            # a lexical token; only lexical/lexical overlap is a collision.
            lexical = [False] * frame_length

            def ensure(position: int) -> None:
                if position >= len(tokens):
                    extension = position + 1 - len(tokens)
                    tokens.extend([self.text_padding_id] * extension)
                    lexical.extend([False] * extension)

            ordered = sorted(alignments, key=lambda item: float(item[1][0]))
            for word, timestamps, _speaker in ordered:
                pieces = self._word_tokens(str(word))
                if not pieces:
                    continue
                desired_start = max(0, round(float(timestamps[0]) * self.frame_rate))
                # Moshi's t_i=1 special case: EPAD occupies the first frame and
                # the lexical pieces move one frame to the right.
                word_start = 1 if desired_start == 0 else desired_start
                ensure(word_start + len(pieces) - 1)
                if any(lexical[word_start : word_start + len(pieces)]):
                    raise TextAlignmentCollision(
                        f"assistant-text BPE pieces collide at frame {word_start}"
                    )
                epad_position = word_start - 1
                # Moshi equation (5): insert EPAD at t_i - 1 unless doing so
                # would overwrite a lexical token from the previous word.
                if not lexical[epad_position]:
                    tokens[epad_position] = self.text_epad_id
                for offset, token in enumerate(pieces):
                    position = word_start + offset
                    tokens[position] = token
                    lexical[position] = True
            return torch.tensor(tokens, dtype=torch.long)

        return render()

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        audio_path = Path(entry["path"])
        if not audio_path.is_absolute():
            audio_path = self.root / audio_path
        wav, _ = sphn.read(audio_path, sample_rate=self.sample_rate)
        if wav.ndim == 1:
            raise ValueError(f"{audio_path}: feedback-audio training requires stereo audio")
        user_channel = int(entry.get("user_channel", self.user_channel))
        assistant_channel = int(entry.get("assistant_channel", self.assistant_channel))
        if user_channel == assistant_channel:
            raise ValueError(f"{audio_path}: user and assistant channels must differ")
        for channel in (user_channel, assistant_channel):
            if not 0 <= channel < wav.shape[0]:
                raise ValueError(f"{audio_path}: missing channel {channel}")
        waveform = torch.from_numpy(wav[user_channel]).float().clamp(-1.0, 1.0)
        assistant_waveform = torch.from_numpy(wav[assistant_channel]).float().clamp(-1.0, 1.0)
        samples_per_frame = self.sample_rate / self.frame_rate
        frame_length = max(1, math.ceil(waveform.numel() / samples_per_frame))
        assistant_text_tokens = self._aligned_text_stream(
            entry["inner_monologue_alignments"], frame_length
        )
        return {
            "id": audio_path.stem,
            "language": entry.get("language", "unknown"),
            "waveform": waveform,
            "assistant_waveform": assistant_waveform,
            "assistant_text_tokens": assistant_text_tokens,
            "audio_frame_length": frame_length,
            "assistant_text_frame_length": assistant_text_tokens.numel(),
        }


def collate_duplex(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_samples = max(item["waveform"].numel() for item in batch)
    max_assistant_text_frames = max(item["assistant_text_frame_length"] for item in batch)
    waveforms = torch.zeros(len(batch), 1, max_samples, dtype=torch.float32)
    assistant_waveforms = torch.zeros(len(batch), 1, max_samples, dtype=torch.float32)
    assistant_text_tokens = torch.full(
        (len(batch), max_assistant_text_frames), -1, dtype=torch.long
    )
    dense_text = [item.get("text_only_tokens", item["assistant_text_tokens"]) for item in batch]
    max_text_only_tokens = max(tokens.numel() for tokens in dense_text)
    text_only_tokens = torch.full(
        (len(batch), max_text_only_tokens), -1, dtype=torch.long
    )
    for index, item in enumerate(batch):
        sample_length = item["waveform"].numel()
        waveforms[index, 0, :sample_length] = item["waveform"]
        assistant_waveforms[index, 0, :sample_length] = item["assistant_waveform"]
        assistant_text_tokens[index, : item["assistant_text_frame_length"]] = item[
            "assistant_text_tokens"
        ]
        text_only_tokens[index, : dense_text[index].numel()] = dense_text[index]
    return {
        "waveforms": waveforms,
        "assistant_waveforms": assistant_waveforms,
        "assistant_text_tokens": assistant_text_tokens,
        "text_only_tokens": text_only_tokens,
        "audio_frame_lengths": torch.tensor(
            [item["audio_frame_length"] for item in batch], dtype=torch.long
        ),
        "assistant_text_frame_lengths": torch.tensor(
            [item["assistant_text_frame_length"] for item in batch], dtype=torch.long
        ),
        "user_audio_present": torch.tensor(
            [item.get("user_audio_present", True) for item in batch], dtype=torch.bool
        ),
        "ids": [item["id"] for item in batch],
        "languages": [item.get("language", "unknown") for item in batch],
    }


class MonoPostTrainingDataset(StereoDialogueDataset):
    """Use the shared folder/sidecar pipeline for mono or stereo speech.

    Mono mode trains assistant audio/text on a complete utterance with no user
    audio. Stereo mode preserves the configured user and assistant channels.
    The manifest is JSONL with at least ``path`` and ``duration``. Put word
    alignments inline as ``alignments``, point to a JSON sidecar with
    ``alignment_path``, or place a sidecar next to the WAV (``0.wav`` ->
    ``0.json``). An alignment uses the existing ``[word, [start, end],
    speaker]`` schema. By default, samples without alignment are rejected so
    every lexical target can be placed on the audio timeline.
    """

    def __init__(
        self,
        manifest: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        sample_rate: int = 24000,
        frame_rate: float = 12.5,
        text_padding_id: int,
        text_epad_id: int,
        min_duration: float = 2.0,
        max_duration: float = 45.0,
        min_text_tokens: int = 0,
        entries: list[dict[str, Any]] | None = None,
        folders: str | None = None,
        max_files: int = 0,
        online_alignment: bool = False,
        whisper_model: str = "medium",
        alignment_device: str = "cpu",
        audio_mode: str = "mono",
        user_channel: int = 1,
        assistant_channel: int = 0,
    ) -> None:
        if min_text_tokens < 0:
            raise ValueError("min_text_tokens must be non-negative")
        if min_text_tokens and tokenizer is None:
            raise ValueError("min_text_tokens requires a tokenizer")
        self.min_text_tokens = min_text_tokens
        if sum(value is not None for value in (manifest, entries, folders)) != 1:
            raise ValueError("provide exactly one of manifest, entries, or folders")
        self.manifest = Path(manifest) if manifest is not None else None
        self.root = self.manifest.parent if self.manifest is not None else Path.cwd()
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.text_padding_id = text_padding_id
        self.text_epad_id = text_epad_id
        self.online_alignment = online_alignment
        self.whisper_model = whisper_model
        self.alignment_device = alignment_device
        if audio_mode not in {"mono", "stereo"}:
            raise ValueError(f"unsupported audio mode: {audio_mode!r}")
        if user_channel == assistant_channel:
            raise ValueError("user and assistant channels must differ")
        self.audio_mode = audio_mode
        self.user_channel = user_channel
        self.assistant_channel = assistant_channel
        self._alignment_model: Any | None = None
        self.original_size = 0
        self.entries: list[dict[str, Any]] = []
        self.rejected = {
            "duration": 0,
            "text_tokens": 0,
            "alignment": 0,
            "text_collision": 0,
        }
        timing_start = time.perf_counter()
        self.timing_stats: dict[str, float | int] = {
            "discovery_seconds": 0.0,
            "alignment_model_load_seconds": 0.0,
            "alignment_inference_seconds": 0.0,
            "filter_seconds": 0.0,
            "alignment_extracted": 0,
            "alignment_reused": 0,
            "alignment_failed": 0,
        }
        self.discovery_stats: dict[str, int] | None = None
        discovery_start = time.perf_counter()
        if folders is not None:
            raw_entries, self.discovery_stats = discover_speech_entries(
                folders,
                max_files=max_files,
                include_unannotated=online_alignment,
                audio_mode=audio_mode,
                user_channel=user_channel,
                assistant_channel=assistant_channel,
            )
        elif entries is None:
            assert self.manifest is not None
            with self.manifest.open(encoding="utf-8") as handle:
                raw_entries = [json.loads(line) for line in handle if line.strip()]
        else:
            raw_entries = entries
        self.timing_stats["discovery_seconds"] = time.perf_counter() - discovery_start
        self.original_size = len(raw_entries)
        if online_alignment and folders is None:
            raise ValueError("online alignment requires folders with explicit languages")
        filter_start = time.perf_counter()
        for index, entry in enumerate(raw_entries):
            duration = float(entry["duration"])
            if not min_duration <= duration <= max_duration:
                self.rejected["duration"] += 1
                continue
            try:
                alignments = self._load_alignments(entry)
                if alignments:
                    alignments = self._validate_alignments(alignments, duration)
            except (OSError, ValueError, KeyError, TypeError):
                if online_alignment:
                    alignments = None
                else:
                    self.rejected["alignment"] += 1
                    continue
            if not alignments and online_alignment:
                # Keep unannotated samples and prepare them lazily when the
                # DataLoader first asks for the item.
                self.entries.append({**entry, "duration": duration, "alignments": None})
                continue
            if not alignments:
                self.rejected["alignment"] += 1
                continue
            try:
                # Audio-only exporters pass None; training always supplies its tokenizer.
                self._check_text_tokens(alignments, str(entry["path"]))
                if self.tokenizer is not None:
                    self._aligned_text_stream(
                        alignments, max(1, math.ceil(duration * self.frame_rate))
                    )
            except RecoverableSampleError:
                continue
            except TextAlignmentCollision:
                self.rejected["text_collision"] += 1
                continue
            self.entries.append({**entry, "duration": duration, "alignments": alignments})
        self.timing_stats["filter_seconds"] = time.perf_counter() - filter_start
        self.timing_stats["total_seconds"] = time.perf_counter() - timing_start
        self.filtered_hours = sum(float(entry["duration"]) for entry in self.entries) / 3600
        if not self.entries:
            if self.rejected["alignment"] == self.original_size:
                raise ValueError(
                    f"empty mono post-training dataset ({self.manifest or folders}); "
                    "word-alignment sidecars are missing (run annotate.py first)"
                )
            raise ValueError(f"empty mono post-training dataset: {self.manifest or folders}")

    def _extract_alignment(self, entry: dict[str, Any]) -> list[Any]:
        """Run Whisper for one item on first DataLoader access."""
        from scripts import annotate
        from types import SimpleNamespace

        if self._alignment_model is None:
            model_load_start = time.perf_counter()
            self._alignment_model = annotate.whisper.load_model(
                self.whisper_model, device=self.alignment_device
            )
            self.timing_stats["alignment_model_load_seconds"] += (
                time.perf_counter() - model_load_start
            )
        alignment_start = time.perf_counter()
        try:
            result = annotate.process_one(
                Path(entry["path"]), None, entry["language"], self._alignment_model,
                SimpleNamespace(keep_silence_in_segments=1.0),
                channel=(
                    int(entry.get("assistant_channel", self.assistant_channel))
                    if self.audio_mode == "stereo" else 0
                ),
            )
            alignments = self._validate_alignments(
                result["alignments"], float(entry["duration"])
            )
            if not alignments:
                raise ValueError("Whisper returned no word alignments")
            self.timing_stats["alignment_extracted"] += 1
            return alignments
        except (torch.cuda.OutOfMemoryError, MemoryError):
            raise
        except Exception as error:
            self.timing_stats["alignment_failed"] += 1
            raise RecoverableSampleError(
                f"{entry['path']}: Whisper alignment failed"
            ) from error
        finally:
            self.timing_stats["alignment_inference_seconds"] += (
                time.perf_counter() - alignment_start
            )

    def _prepare_lazy_entry(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        if entry.get("alignments") is not None:
            return entry
        if not self.online_alignment:
            return entry
        alignments = self._extract_alignment(entry)
        self._check_text_tokens(alignments, str(entry["path"]))
        if self.tokenizer is not None:
            try:
                self._aligned_text_stream(
                    alignments,
                    max(1, math.ceil(float(entry["duration"]) * self.frame_rate)),
                )
            except TextAlignmentCollision as error:
                self.rejected["text_collision"] += 1
                raise RecoverableSampleError(f"{entry['path']}: {error}") from error
        entry.update(alignments=alignments)
        return entry

    def _check_text_tokens(self, alignments: list[Any], path: str) -> None:
        if not self.min_text_tokens:
            return
        # Match the word-by-word encoding used by _aligned_text_stream;
        # inserted PAD/EPAD tokens do not count as lexical targets.
        count = self._lexical_token_count(alignments)
        if count < self.min_text_tokens:
            self.rejected["text_tokens"] += 1
            raise RecoverableSampleError(
                f"{path}: mono transcript rejected by text_tokens filter "
                f"(text_tokens={count}, min_text_tokens={self.min_text_tokens})"
            )

    def _dense_text_tokens(self, alignments: list[Any]) -> torch.Tensor:
        text = " ".join(str(word) for word, _timestamps, _speaker in alignments)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if isinstance(tokens, dict):
            tokens = tokens["input_ids"]
        return torch.tensor(tokens, dtype=torch.long)

    def _load_alignments(self, entry: dict[str, Any]) -> list[Any] | None:
        if "alignments" in entry:
            return entry["alignments"]
        alignment_value = entry.get("alignment_path")
        if not alignment_value:
            audio_path = Path(entry["path"])
            if not audio_path.is_absolute():
                audio_path = self.root / audio_path
            alignment_path = audio_path.with_suffix(".json")
            if not alignment_path.exists():
                return None
        else:
            alignment_path = Path(alignment_value)
        if not alignment_path.is_absolute():
            alignment_path = self.root / alignment_path
        payload = json.loads(alignment_path.read_text(encoding="utf-8"))
        assistant_channel = int(entry.get("assistant_channel", self.assistant_channel))
        channel_alignments = payload.get("channel_alignments", {})
        if self.audio_mode == "stereo" and str(assistant_channel) in channel_alignments:
            return channel_alignments[str(assistant_channel)]
        if (
            self.audio_mode == "stereo"
            and payload.get("assistant_channel", assistant_channel) != assistant_channel
            and "all_alignments" in payload
            and "speakers" in payload
        ):
            assistant_speaker = payload["speakers"][assistant_channel]
            return [
                alignment
                for alignment in payload["all_alignments"]
                if alignment[2] == assistant_speaker
            ]
        return payload["alignments"]

    @staticmethod
    def _validate_alignments(alignments: list[Any], duration: float) -> list[Any]:
        result = []
        for alignment in alignments:
            if len(alignment) != 3:
                raise ValueError("invalid alignment")
            word, timestamps, speaker = alignment
            start, end = map(float, timestamps)
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start or end > duration + 0.05:
                raise ValueError("invalid alignment timestamp")
            result.append([str(word), [start, end], speaker])
        return sorted(result, key=lambda item: item[1][0])

    def render_stereo(self, index: int) -> tuple[torch.Tensor, torch.Tensor, list[Any] | None]:
        """Render mono post-training or native stereo dialogue streams."""
        entry = self._prepare_lazy_entry(index)
        audio_path = Path(entry["path"])
        if not audio_path.is_absolute():
            audio_path = self.root / audio_path
        wav, _ = sphn.read(audio_path, sample_rate=self.sample_rate)
        if self.audio_mode == "stereo":
            if wav.ndim != 2:
                raise ValueError(f"{audio_path}: stereo training expects multi-channel audio")
            user_channel = int(entry.get("user_channel", self.user_channel))
            assistant_channel = int(entry.get("assistant_channel", self.assistant_channel))
            if user_channel == assistant_channel:
                raise ValueError(f"{audio_path}: user and assistant channels must differ")
            for channel in (user_channel, assistant_channel):
                if not 0 <= channel < wav.shape[0]:
                    raise ValueError(f"{audio_path}: missing channel {channel}")
            user = torch.from_numpy(wav[user_channel]).float().clamp(-1.0, 1.0)
            assistant = torch.from_numpy(wav[assistant_channel]).float().clamp(-1.0, 1.0)
            return user, assistant, entry["alignments"]
        if wav.ndim == 2 and wav.shape[0] == 1:
            wav = wav[0]
        if wav.ndim != 1:
            raise ValueError(f"{audio_path}: mono post-training expects mono audio")
        source = torch.from_numpy(wav).float().clamp(-1.0, 1.0)
        user = torch.zeros_like(source)
        assistant = source
        alignments = entry["alignments"]
        if alignments is None:
            return user, assistant, None
        return user, assistant, alignments

    def __getitem__(self, index: int) -> dict[str, Any]:
        user, assistant, alignments = self.render_stereo(index)
        frame_length = max(1, math.ceil(user.numel() / (self.sample_rate / self.frame_rate)))
        if alignments is None:
            assistant_text_tokens = torch.full(
                (frame_length,), self.text_padding_id, dtype=torch.long
            )
        else:
            assistant_text_tokens = self._aligned_text_stream(alignments, frame_length)
        text_only_tokens = self._dense_text_tokens(alignments or [])
        return {
            "id": f"{Path(self.entries[index]['path']).stem}-{index}",
            "language": self.entries[index].get("language", "unknown"),
            "waveform": user,
            "assistant_waveform": assistant,
            "assistant_text_tokens": assistant_text_tokens,
            "text_only_tokens": text_only_tokens,
            "audio_frame_length": frame_length,
            "assistant_text_frame_length": assistant_text_tokens.numel(),
            "user_audio_present": self.audio_mode == "stereo",
        }
