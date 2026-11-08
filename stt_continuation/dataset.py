"""DailyTalk user audio and aligned assistant continuation targets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import sphn
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TextAlignmentCollision(ValueError):
    """Two lexical tokens cannot fit on the same fixed text timeline."""


class DailyTalkDataset(Dataset[dict[str, Any]]):
    """Read target speech and word-level alignments from DailyTalkContiguous.

    Audio tokenization intentionally happens after collation so frozen Mimi can
    run online on the training device.
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
        user_channel: int = 1,
        assistant_channel: int = 0,
        min_duration: float = 5.0,
        max_duration: float = 45.0,
        min_words: int = 5,
        min_words_per_second: float = 0.5,
        max_words_per_second: float = 2.5,
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
        with self.manifest.open(encoding="utf-8") as handle:
            raw_entries = [json.loads(line) for line in handle if line.strip()]
        self.original_size = len(raw_entries)
        self.entries = []
        self.rejected: dict[str, int] = {
            "duration": 0,
            "words": 0,
            "words_per_second": 0,
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
            # The assistant-side transcript is the continuation target, i.e.
            # the model's inner-monologue text stream.  The current sidecar
            # schema stores its timestamped words under ``alignments``.
            inner_monologue_alignments = json.loads(
                audio_path.with_suffix(".json").read_text(encoding="utf-8")
            )["alignments"]
            word_count = len(inner_monologue_alignments)
            if word_count < min_words:
                self.rejected["words"] += 1
                continue
            words_per_second = word_count / duration
            if not min_words_per_second <= words_per_second <= max_words_per_second:
                self.rejected["words_per_second"] += 1
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
                    "inner_monologue_alignments": inner_monologue_alignments,
                }
            )
        self.filtered_hours = sum(float(entry["duration"]) for entry in self.entries) / 3600
        if not self.entries:
            raise ValueError(f"empty manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    def _word_tokens(self, word: str) -> list[int]:
        # Preserve SmolLM2's word-boundary BPE form for every aligned word.
        return self.tokenizer.encode(" " + word, add_special_tokens=False)

    def _aligned_text_stream(
        self, alignments: list[Any], frame_length: int
    ) -> torch.Tensor:
        """Build the assistant continuation timeline on the audio clock."""
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
                        f"continuation BPE pieces collide at frame {word_start}"
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
        continuation_tokens = self._aligned_text_stream(
            entry["inner_monologue_alignments"],
            frame_length,
        )
        return {
            "id": audio_path.stem,
            "waveform": waveform,
            "assistant_waveform": assistant_waveform,
            "continuation_tokens": continuation_tokens,
            "audio_frame_length": frame_length,
            "continuation_frame_length": continuation_tokens.numel(),
        }


def collate_dailytalk(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_samples = max(item["waveform"].numel() for item in batch)
    max_continuation_frames = max(item["continuation_frame_length"] for item in batch)
    waveforms = torch.zeros(len(batch), 1, max_samples, dtype=torch.float32)
    assistant_waveforms = torch.zeros(len(batch), 1, max_samples, dtype=torch.float32)
    continuation_tokens = torch.full(
        (len(batch), max_continuation_frames), -1, dtype=torch.long
    )
    for index, item in enumerate(batch):
        sample_length = item["waveform"].numel()
        waveforms[index, 0, :sample_length] = item["waveform"]
        assistant_waveforms[index, 0, :sample_length] = item["assistant_waveform"]
        continuation_tokens[index, : item["continuation_frame_length"]] = item[
            "continuation_tokens"
        ]
    return {
        "waveforms": waveforms,
        "assistant_waveforms": assistant_waveforms,
        "continuation_tokens": continuation_tokens,
        "audio_frame_lengths": torch.tensor(
            [item["audio_frame_length"] for item in batch], dtype=torch.long
        ),
        "continuation_frame_lengths": torch.tensor(
            [item["continuation_frame_length"] for item in batch], dtype=torch.long
        ),
        "ids": [item["id"] for item in batch],
    }
