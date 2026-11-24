"""VAD-driven stereo data for contiguous natural monologues.

This pipeline deliberately does not consume word timestamps.  Each JSONL
manifest row points at a stereo WAV and contains assistant transcripts.
Assistant text is emitted as contiguous BPE pieces; online per-channel VAD
assigns HEAR, SPK, or WAIT to frames that have no lexical piece.  Whisper and
word alignment are not used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import sphn
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import PreTrainedTokenizerBase


class NaturalMonologueDataset(Dataset[dict[str, Any]]):
    """Load stereo dialogue without word-level alignment.

    Manifest rows use this schema::

        {
          "path": "dialog.wav",
          "duration": 12.4,
          "language": "ko",
          "user_channel": 1,
          "assistant_channel": 0,
          "assistant_texts": ["내일은 맑습니다", "부산도 맑습니다"]
        }

    A single ``assistant_text`` string is also accepted.  With multiple texts,
    entries may be objects containing an optional coarse ``start``.  Otherwise
    texts are paired in order with assistant VAD segments.  Text begins
    ``monologue_lead_frames`` before the corresponding assistant speech onset,
    giving one-frame text-to-audio interleaving by default.
    """

    def __init__(
        self,
        manifest: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        wait_token_id: int,
        spk_token_id: int,
        hear_token_id: int,
        sample_rate: int = 24_000,
        frame_rate: float = 12.5,
        monologue_lead_frames: int = 1,
        user_channel: int = 1,
        assistant_channel: int = 0,
        min_duration: float = 0.0,
        max_duration: float = float("inf"),
        min_text_tokens: int = 1,
        vad_min_speech_duration: float = 0.1,
        vad_min_silence_duration: float = 0.1,
        vad_energy_threshold: float = 50.0,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.tokenizer = tokenizer
        self.wait_token_id = int(wait_token_id)
        self.spk_token_id = int(spk_token_id)
        self.hear_token_id = int(hear_token_id)
        self.sample_rate = int(sample_rate)
        self.frame_rate = float(frame_rate)
        self.monologue_lead_frames = int(monologue_lead_frames)
        self.user_channel = int(user_channel)
        self.assistant_channel = int(assistant_channel)
        self.min_text_tokens = int(min_text_tokens)
        self.vad_min_speech_duration = float(vad_min_speech_duration)
        self.vad_min_silence_duration = float(vad_min_silence_duration)
        self.vad_energy_threshold = float(vad_energy_threshold)

        if len({self.wait_token_id, self.spk_token_id, self.hear_token_id}) != 3:
            raise ValueError("WAIT, SPK, and HEAR token IDs must be distinct")
        if self.sample_rate <= 0 or self.frame_rate <= 0:
            raise ValueError("sample_rate and frame_rate must be positive")
        if self.monologue_lead_frames < 0:
            raise ValueError("monologue_lead_frames must be non-negative")
        if self.user_channel == self.assistant_channel:
            raise ValueError("user and assistant channels must differ")
        if self.min_text_tokens < 0:
            raise ValueError("min_text_tokens must be non-negative")
        if self.vad_min_speech_duration <= 0 or self.vad_min_silence_duration < 0:
            raise ValueError("VAD durations must be non-negative and speech must be positive")

        with self.manifest.open(encoding="utf-8") as handle:
            raw_entries = [json.loads(line) for line in handle if line.strip()]
        self.original_size = len(raw_entries)
        self.rejected = {"duration": 0, "annotation": 0, "text_tokens": 0}
        self.entries: list[dict[str, Any]] = []
        for entry_index, raw in enumerate(raw_entries):
            try:
                entry = self._validate_entry(raw, entry_index)
            except (KeyError, TypeError, ValueError):
                self.rejected["annotation"] += 1
                continue
            duration = float(entry["duration"])
            if not min_duration <= duration <= max_duration:
                self.rejected["duration"] += 1
                continue
            token_count = sum(
                len(self._encode_text(item["text"]))
                for item in entry["monologues"]
            )
            if token_count < self.min_text_tokens:
                self.rejected["text_tokens"] += 1
                continue
            self.entries.append(entry)
        if not self.entries:
            raise ValueError(f"empty natural-monologue manifest: {self.manifest}")
        self.filtered_hours = sum(float(item["duration"]) for item in self.entries) / 3600

    def _validate_entry(self, raw: dict[str, Any], entry_index: int) -> dict[str, Any]:
        path = Path(str(raw["path"]))
        duration = float(raw["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"entry {entry_index}: invalid duration")
        source: Any
        if "assistant_texts" in raw:
            source = raw["assistant_texts"]
        elif "assistant_text" in raw:
            source = [raw["assistant_text"]]
        elif "utterances" in raw:
            source = [
                item for item in raw["utterances"]
                if str(item.get("speaker", "")).lower() == "assistant"
            ]
        else:
            raise ValueError(f"entry {entry_index}: missing assistant transcript")
        if not isinstance(source, list) or not source:
            raise ValueError(f"entry {entry_index}: assistant transcript is empty")
        monologues = []
        for item in source:
            if isinstance(item, str):
                text, start = item.strip(), None
            elif isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                start = float(item["start"]) if item.get("start") is not None else None
            else:
                raise TypeError("assistant texts must be strings or objects")
            if not text:
                raise ValueError("assistant text is empty")
            if start is not None and (not math.isfinite(start) or not 0 <= start < duration):
                raise ValueError(f"invalid monologue start: {start}")
            monologues.append({"text": text, "start": start})
        return {
            **raw,
            "path": str(path),
            "duration": duration,
            "monologues": monologues,
        }

    @staticmethod
    def _validate_interval(start: float, end: float, duration: float) -> None:
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or end > duration + 1e-6
        ):
            raise ValueError(f"invalid interval [{start}, {end}] for duration {duration}")

    def _encode_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def __len__(self) -> int:
        return len(self.entries)

    def _frame_interval(self, start: float, end: float, frame_length: int) -> range:
        # VAD boundaries commonly contain tiny float errors (for example,
        # 0.6000000000000001).  Without the epsilon an exact frame boundary
        # would incorrectly activate one additional HEAR/SPK frame.
        epsilon = 1e-7
        first = max(
            0, min(frame_length, math.floor(start * self.frame_rate + epsilon))
        )
        last = max(
            first, min(frame_length, math.ceil(end * self.frame_rate - epsilon))
        )
        return range(first, last)

    def _vad_intervals(self, waveform: torch.Tensor) -> list[tuple[float, float]]:
        """Run lightweight Auditok VAD without loading Whisper or a GPU model."""
        try:
            import auditok
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "NaturalMonologueDataset requires auditok for online VAD"
            ) from error
        normalized = waveform / max(0.1, float(waveform.abs().max()))
        pcm = (normalized.numpy() * 32767).astype("int16").tobytes()
        duration = waveform.numel() / self.sample_rate
        regions = auditok.split(
            pcm,
            sampling_rate=self.sample_rate,
            channels=1,
            sample_width=2,
            min_dur=self.vad_min_speech_duration,
            max_dur=max(duration, self.vad_min_speech_duration),
            max_silence=min(duration * 0.95, self.vad_min_silence_duration),
            energy_threshold=self.vad_energy_threshold,
            drop_trailing_silence=True,
        )
        intervals = []
        for region in regions:
            meta = getattr(region, "_meta", region)
            intervals.append((float(meta.start), min(duration, float(meta.end))))
        return intervals

    def _text_stream(
        self,
        entry: dict[str, Any],
        frame_length: int,
        user_intervals: list[tuple[float, float]],
        assistant_intervals: list[tuple[float, float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_active = torch.zeros(frame_length, dtype=torch.bool)
        assistant_active = torch.zeros(frame_length, dtype=torch.bool)
        for target, intervals in (
            (user_active, user_intervals),
            (assistant_active, assistant_intervals),
        ):
            for start, end in intervals:
                interval = self._frame_interval(start, end, frame_length)
                target[interval.start:interval.stop] = True

        # HEAR wins during overlap because assistant activity is already
        # represented by the assistant audio stream.
        stream = torch.full((frame_length,), self.wait_token_id, dtype=torch.long)
        stream[assistant_active] = self.spk_token_id
        stream[user_active] = self.hear_token_id
        lexical = torch.zeros(frame_length, dtype=torch.bool)
        dense_text: list[int] = []
        for monologue_index, item in enumerate(entry["monologues"]):
            pieces = self._encode_text(item["text"])
            dense_text.extend(pieces)
            speech_start = item["start"]
            if speech_start is None:
                if monologue_index >= len(assistant_intervals):
                    raise ValueError(
                        f"{entry['path']}: VAD found {len(assistant_intervals)} assistant "
                        f"segments for {len(entry['monologues'])} monologues"
                    )
                speech_start = assistant_intervals[monologue_index][0]
            start = max(
                0,
                math.floor(speech_start * self.frame_rate) - self.monologue_lead_frames,
            )
            required = start + len(pieces)
            if required > stream.numel():
                extension = required - stream.numel()
                stream = torch.cat(
                    [
                        stream,
                        torch.full((extension,), self.wait_token_id, dtype=torch.long),
                    ]
                )
                lexical = torch.cat([lexical, torch.zeros(extension, dtype=torch.bool)])
            if lexical[start:required].any():
                raise ValueError(
                    f"{entry['path']}: contiguous assistant monologues overlap at frame {start}"
                )
            stream[start:required] = torch.tensor(pieces, dtype=torch.long)
            lexical[start:required] = True
        return stream, torch.tensor(dense_text, dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        audio_path = Path(entry["path"])
        if not audio_path.is_absolute():
            audio_path = self.root / audio_path
        wav, _ = sphn.read(audio_path, sample_rate=self.sample_rate)
        if wav.ndim != 2:
            raise ValueError(f"{audio_path}: natural-monologue training requires stereo audio")
        user_channel = int(entry.get("user_channel", self.user_channel))
        assistant_channel = int(entry.get("assistant_channel", self.assistant_channel))
        if user_channel == assistant_channel:
            raise ValueError(f"{audio_path}: user and assistant channels must differ")
        for channel in (user_channel, assistant_channel):
            if not 0 <= channel < wav.shape[0]:
                raise ValueError(f"{audio_path}: missing channel {channel}")
        user = torch.from_numpy(wav[user_channel]).float().clamp(-1.0, 1.0)
        assistant = torch.from_numpy(wav[assistant_channel]).float().clamp(-1.0, 1.0)
        sample_length = min(user.numel(), assistant.numel())
        user, assistant = user[:sample_length], assistant[:sample_length]
        frame_length = max(
            1, math.ceil(sample_length / (self.sample_rate / self.frame_rate))
        )
        user_intervals = self._vad_intervals(user)
        assistant_intervals = self._vad_intervals(assistant)
        text, dense_text = self._text_stream(
            entry, frame_length, user_intervals, assistant_intervals
        )
        return {
            "id": str(entry.get("id", audio_path.stem)),
            "language": str(entry.get("language", "unknown")),
            "waveform": user,
            "assistant_waveform": assistant,
            "assistant_text_tokens": text,
            "text_only_tokens": dense_text,
            "audio_frame_length": frame_length,
            "assistant_text_frame_length": text.numel(),
            "user_audio_present": True,
            "user_vad_intervals": user_intervals,
            "assistant_vad_intervals": assistant_intervals,
        }


def collate_natural_monologue(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad natural-monologue samples into the current training batch schema."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_samples = max(item["waveform"].numel() for item in batch)
    max_text_frames = max(item["assistant_text_frame_length"] for item in batch)
    max_dense_tokens = max(item["text_only_tokens"].numel() for item in batch)
    waveforms = torch.zeros(len(batch), 1, max_samples, dtype=torch.float32)
    assistant_waveforms = torch.zeros_like(waveforms)
    assistant_text_tokens = torch.full(
        (len(batch), max_text_frames), -1, dtype=torch.long
    )
    text_only_tokens = torch.full(
        (len(batch), max_dense_tokens), -1, dtype=torch.long
    )
    for row, item in enumerate(batch):
        samples = item["waveform"].numel()
        text_frames = item["assistant_text_frame_length"]
        dense_tokens = item["text_only_tokens"].numel()
        waveforms[row, 0, :samples] = item["waveform"]
        assistant_waveforms[row, 0, :samples] = item["assistant_waveform"]
        assistant_text_tokens[row, :text_frames] = item["assistant_text_tokens"]
        text_only_tokens[row, :dense_tokens] = item["text_only_tokens"]
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
        "user_audio_present": torch.ones(len(batch), dtype=torch.bool),
        "ids": [item["id"] for item in batch],
        "languages": [item["language"] for item in batch],
    }


def create_natural_monologue_dataloader(
    dataset: NaturalMonologueDataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    sampler: Sampler[int] | Iterable[int] | None = None,
    num_workers: int = 2,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Construct a DataLoader suitable for local or DDP training."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sampler is not None and shuffle:
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_natural_monologue,
    )
