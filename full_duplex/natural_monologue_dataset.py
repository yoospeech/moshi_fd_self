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
import re
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import sphn
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import PreTrainedTokenizerBase


_NON_SPEECH_SQUARE = re.compile(
    r"\[(?:[^\[\]]*\s)?(?:laughter|nods|gasps|surprised)\]",
    re.IGNORECASE,
)
_NON_SPEECH_ROUND = re.compile(
    r"\((?:pause|interrupt|interrupted|hesitation|softly)\)",
    re.IGNORECASE,
)
_NON_SPEECH_ANGLE = re.compile(
    r"</?laughter>|<(?:sigh|interrupted)>",
    re.IGNORECASE,
)


def clean_behavior_text(text: str) -> str | None:
    """Remove transcript markup without changing spoken words or timestamps."""
    cleaned = text.strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"</?strong>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = _NON_SPEECH_ANGLE.sub(" ", cleaned)
    cleaned = _NON_SPEECH_SQUARE.sub(" ", cleaned)
    cleaned = _NON_SPEECH_ROUND.sub(" ", cleaned)

    # Markdown emphasis is formatting only; preserve the emphasized words.
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"\*([^*\n]+)\*", r"\1", cleaned)

    # Remaining balanced wrappers contain spoken backchannels such as [yeah].
    cleaned = re.sub(r"\[([^\[\]]+)\]", r"\1", cleaned)
    cleaned = re.sub(r"<([^<>/]+)>", r"\1", cleaned)

    # Leftovers indicate truncated markup or corrupted math exports.
    if any(char in cleaned for char in "[]<>"):
        raise ValueError(f"malformed transcript markup: {text!r}")
    if cleaned.count("(") != cleaned.count(")"):
        raise ValueError(f"unbalanced transcript parentheses: {text!r}")
    if "\\" in cleaned or "|" in cleaned:
        raise ValueError(f"corrupted transcript text: {text!r}")

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned or None


def discover_behavior_sd_entries(
    folders: str,
    *,
    max_files: int = 0,
    assistant_channel: int = 0,
    user_channel: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Discover adjacent Behavior-SD audio/JSON pairs from folder inputs."""
    import soundfile as sf

    if max_files < 0:
        raise ValueError("max_files must be non-negative")
    sources: list[tuple[Path, str]] = []
    for part in folders.split(","):
        value = part.strip()
        if not value or ":" not in value:
            raise ValueError(f"use folder:language input, got {value!r}")
        folder, language = value.rsplit(":", 1)
        root = Path(folder).expanduser().resolve()
        if not root.is_dir() or not language.strip():
            raise ValueError(f"invalid folder:language input: {value!r}")
        source = (root, language.strip())
        if source not in sources:
            sources.append(source)

    entries: list[dict[str, Any]] = []
    stats = {"audio": 0, "paired": 0, "invalid": 0}
    for root, language in sources:
        for audio_path in root.rglob("*.wav"):
            stats["audio"] += 1
            metadata_path = audio_path.with_suffix(".json")
            if not metadata_path.is_file():
                stats["invalid"] += 1
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if Path(str(metadata["file_name"])).stem != audio_path.stem:
                    raise ValueError("metadata file_name stem does not match audio")
                monologues: list[dict[str, Any]] = []
                for utterance in metadata["utterances"]:
                    speaker = int(utterance["speaker_idx"])
                    if speaker == assistant_channel:
                        text = str(utterance.get("tts_text", "")).strip()
                        if text:
                            item = {
                                "start": float(utterance["start_time"]),
                                "text": text,
                            }
                            if utterance.get("end_time") is not None:
                                item["end"] = float(utterance["end_time"])
                            monologues.append(item)
                    else:
                        # Nested backchannels are spoken by the listener.
                        for backchannel in utterance.get("backchannels", []):
                            text = str(backchannel.get("tts_text", "")).strip()
                            if text:
                                item = {
                                    "start": float(backchannel["start_time"]),
                                    "text": text,
                                }
                                if backchannel.get("end_time") is not None:
                                    item["end"] = float(backchannel["end_time"])
                                monologues.append(item)
                if not monologues:
                    raise ValueError("no assistant text")
                monologues.sort(key=lambda item: item["start"])
                # Keep end times so validation can join touching utterances
                # before tokenizing the complete assistant monologue.
                info = sf.info(audio_path)
                if info.channels != 2 or info.frames <= 0:
                    raise ValueError("expected nonempty stereo audio")
            except (KeyError, TypeError, ValueError, OSError, RuntimeError, json.JSONDecodeError):
                stats["invalid"] += 1
                continue
            entries.append({
                "id": metadata_path.stem,
                "path": str(audio_path.resolve()),
                "duration": info.frames / info.samplerate,
                "language": language,
                "user_channel": user_channel,
                "assistant_channel": assistant_channel,
                "assistant_texts": monologues,
            })
            stats["paired"] += 1
            if max_files and len(entries) >= max_files:
                return entries, stats
    return entries, stats


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
    texts are paired in order with assistant VAD segments. Text stays on its
    annotated speech onset; model-level audio delay provides text-to-audio
    interleaving without shifting the text timeline.
    """

    def __init__(
        self,
        manifest: str | Path | None,
        tokenizer: PreTrainedTokenizerBase,
        *,
        wait_token_id: int,
        spk_token_id: int,
        hear_token_id: int,
        sample_rate: int = 24_000,
        frame_rate: float = 12.5,
        user_channel: int = 1,
        assistant_channel: int = 0,
        min_duration: float = 0.0,
        max_duration: float = float("inf"),
        min_text_tokens: int = 1,
        vad_min_speech_duration: float = 0.1,
        vad_min_silence_duration: float = 0.1,
        vad_energy_threshold: float = 50.0,
        enable_sine_conditioning: bool = False,
        sine_frequency: float = 440.0,
        sine_amplitude: float = 0.5,
        audio_delay_steps: int = 1,
        folders: str | None = None,
        max_files: int = 0,
    ) -> None:
        if (manifest is None) == (folders is None):
            raise ValueError("provide exactly one of manifest or folders")
        self.manifest = Path(manifest) if manifest is not None else None
        self.root = self.manifest.parent if self.manifest is not None else Path.cwd()
        self.tokenizer = tokenizer
        self.wait_token_id = int(wait_token_id)
        self.spk_token_id = int(spk_token_id)
        self.hear_token_id = int(hear_token_id)
        self.sample_rate = int(sample_rate)
        self.frame_rate = float(frame_rate)
        self.user_channel = int(user_channel)
        self.assistant_channel = int(assistant_channel)
        self.min_text_tokens = int(min_text_tokens)
        self.vad_min_speech_duration = float(vad_min_speech_duration)
        self.vad_min_silence_duration = float(vad_min_silence_duration)
        self.vad_energy_threshold = float(vad_energy_threshold)
        self.enable_sine_conditioning = bool(enable_sine_conditioning)
        self.sine_frequency = float(sine_frequency)
        self.sine_amplitude = float(sine_amplitude)
        self.audio_delay_steps = int(audio_delay_steps)
        if self.audio_delay_steps < 0:
            raise ValueError("audio_delay_steps must be nonnegative")

        if len({self.wait_token_id, self.spk_token_id, self.hear_token_id}) != 3:
            raise ValueError("WAIT, SPK, and HEAR token IDs must be distinct")
        if self.sample_rate <= 0 or self.frame_rate <= 0:
            raise ValueError("sample_rate and frame_rate must be positive")
        if self.user_channel == self.assistant_channel:
            raise ValueError("user and assistant channels must differ")
        if self.min_text_tokens < 0:
            raise ValueError("min_text_tokens must be non-negative")
        if self.vad_min_speech_duration <= 0 or self.vad_min_silence_duration < 0:
            raise ValueError("VAD durations must be non-negative and speech must be positive")
        if not math.isfinite(self.sine_frequency) or self.sine_frequency <= 0:
            raise ValueError("sine_frequency must be positive and finite")
        if not math.isfinite(self.sine_amplitude) or not 0 <= self.sine_amplitude <= 1:
            raise ValueError("sine_amplitude must be finite and between 0 and 1")

        self.discovery_stats: dict[str, int] | None = None
        if folders is not None:
            raw_entries, self.discovery_stats = discover_behavior_sd_entries(
                folders,
                max_files=max_files,
                assistant_channel=assistant_channel,
                user_channel=user_channel,
            )
        else:
            assert self.manifest is not None
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
            raise ValueError(f"empty natural-monologue input: {self.manifest or folders}")
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
            end = None
            if isinstance(item, str):
                text, start = item.strip(), None
            elif isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                start_value = item.get("start", item.get("start_time"))
                start = float(start_value) if start_value is not None else None
                end_value = item.get("end", item.get("end_time"))
                end = float(end_value) if end_value is not None else None
            else:
                raise TypeError("assistant texts must be strings or objects")
            text = clean_behavior_text(text)
            if text is None:
                continue
            if start is not None and (not math.isfinite(start) or not 0 <= start < duration):
                raise ValueError(f"invalid monologue start: {start}")
            if end is not None and (
                start is None or not math.isfinite(end) or end <= start
            ):
                raise ValueError(f"invalid monologue end: {end}")
            monologue = {"text": text, "start": start}
            if end is not None:
                monologue["end"] = end
            if (
                monologues
                and start is not None
                and monologues[-1].get("end") is not None
                and math.isclose(monologues[-1]["end"], start, rel_tol=0.0, abs_tol=1e-6)
            ):
                # A later sentence now follows the previous text tokens rather
                # than restarting at its own annotated onset.
                previous = monologues[-1]
                previous["text"] += " " + text
                previous.pop("end")
                if end is not None:
                    previous["end"] = end
            else:
                monologues.append(monologue)
        if not monologues:
            raise ValueError("assistant transcript contains no lexical text")
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_active = torch.zeros(frame_length, dtype=torch.bool)
        assistant_active = torch.zeros(frame_length, dtype=torch.bool)
        for target, intervals in (
            (user_active, user_intervals),
            (assistant_active, assistant_intervals),
        ):
            for start, end in intervals:
                interval = self._frame_interval(start, end, frame_length)
                target[interval.start:interval.stop] = True

        # Build the source timeline before the model delays both audio streams.
        stream = torch.full((frame_length,), self.wait_token_id, dtype=torch.long)
        stream[assistant_active] = self.spk_token_id
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
            # Place text on the audio frame containing its annotated onset.
            # Text is never shifted to manufacture text/audio interleaving;
            # the model's delay pattern delays audio instead.
            start = max(0, math.floor(speech_start * self.frame_rate))
            required = start + len(pieces)
            if required > stream.numel():
                if self.enable_sine_conditioning:
                    raise ValueError(
                        f"{entry['path']}: assistant text requires {required} frames "
                        f"but audio has only {frame_length}; sine conditioning requires "
                        "the complete text timeline to fit inside the audio timeline"
                    )
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
        if self.enable_sine_conditioning:
            assistant_content = assistant_active | lexical[:frame_length]
            schedule = self._interleave_schedule(user_active, assistant_content)
            last_content = max(
                (i for i, source in enumerate(schedule) if source is not None),
                default=-1,
            )
            last_user = int(user_active.nonzero()[-1]) if user_active.any() else -1
            # Keep the delayed audio tail and a final WAIT in the text timeline.
            required = max(len(schedule), max(last_content, last_user) + self.audio_delay_steps + 2)
            schedule.extend([None] * (required - len(schedule)))
            interleaved = torch.full(
                (len(schedule),), self.wait_token_id, dtype=torch.long
            )
            output_user_active = torch.zeros(len(schedule), dtype=torch.bool)
            output_user_active[:frame_length] = user_active
            assistant_schedule = torch.full((len(schedule),), -1, dtype=torch.long)
            for output_frame, source_frame in enumerate(schedule):
                if source_frame is not None and assistant_active[source_frame]:
                    assistant_schedule[output_frame] = source_frame
            # Controls describe the audible (delayed) frame. Lexical tokens
            # announce the next audio frame and take precedence over controls.
            for output_frame, source_frame in enumerate(schedule):
                audible_source = output_frame - self.audio_delay_steps
                if audible_source >= 0:
                    if output_user_active[audible_source]:
                        interleaved[output_frame] = self.hear_token_id
                    elif assistant_schedule[audible_source] >= 0:
                        interleaved[output_frame] = self.spk_token_id
                if source_frame is not None and lexical[source_frame]:
                    interleaved[output_frame] = stream[source_frame]
            stream = interleaved
        else:
            # Without pause/resume conditioning, retain the original aligned
            # behavior: lexical pieces take precedence over HEAR.
            lexical_tokens = stream[lexical].clone()
            stream[:frame_length][user_active] = self.hear_token_id
            stream[lexical] = lexical_tokens
            assistant_schedule = torch.arange(frame_length, dtype=torch.long)
            output_user_active = user_active
        return (
            stream,
            torch.tensor(dense_text, dtype=torch.long),
            assistant_schedule,
            output_user_active,
        )

    @staticmethod
    def _interleave_schedule(
        user_active: torch.Tensor, assistant_content: torch.Tensor
    ) -> list[int | None]:
        """Queue only assistant content interrupted by user speech.

        Empty source frames are represented by sine and are never queued. This
        preserves the absolute timeline until a real overlap occurs, then drains
        interrupted assistant frames in order without losing content.
        """
        schedule: list[int | None] = []
        queued: deque[int] = deque()
        for source_frame in range(user_active.numel()):
            has_content = bool(assistant_content[source_frame])
            if bool(user_active[source_frame]):
                if has_content:
                    queued.append(source_frame)
                schedule.append(None)
            elif queued:
                output = queued.popleft()
                if has_content:
                    queued.append(source_frame)
                schedule.append(output)
            elif has_content:
                schedule.append(source_frame)
            else:
                schedule.append(None)
        schedule.extend(queued)
        return schedule

    def _apply_sine_conditioning(
        self, assistant: torch.Tensor, assistant_schedule: torch.Tensor
    ) -> torch.Tensor:
        """Render scheduled assistant content and sine in every empty frame."""
        if not self.enable_sine_conditioning:
            return assistant
        samples_per_frame = int(self.sample_rate / self.frame_rate)
        if samples_per_frame < 1:
            raise ValueError("frame_rate cannot exceed sample_rate")
        source_frames = math.ceil(assistant.numel() / samples_per_frame)
        padded_samples = source_frames * samples_per_frame
        source = torch.nn.functional.pad(
            assistant[:padded_samples],
            (0, max(0, padded_samples - assistant.numel())),
        )
        source = source.view(source_frames, samples_per_frame)
        conditioned = assistant.new_empty(assistant_schedule.numel(), samples_per_frame)
        time = torch.arange(
            conditioned.numel(), dtype=torch.float32, device=assistant.device
        ) / self.sample_rate
        sine = self.sine_amplitude * torch.sin(
            2 * math.pi * self.sine_frequency * time
        ).view_as(conditioned).to(dtype=conditioned.dtype)
        for output_frame, source_frame in enumerate(assistant_schedule.tolist()):
            if source_frame < 0:
                conditioned[output_frame] = sine[output_frame]
            else:
                conditioned[output_frame] = source[source_frame]
                valid_samples = min(samples_per_frame, assistant.numel() - source_frame * samples_per_frame)
                conditioned[output_frame, valid_samples:] = sine[output_frame, valid_samples:]
        return conditioned.flatten()

    def _apply_user_sine(
        self, user: torch.Tensor, user_active: torch.Tensor
    ) -> torch.Tensor:
        """Keep real user speech and represent every no-user frame with sine."""
        if not self.enable_sine_conditioning:
            return user
        samples_per_frame = int(self.sample_rate / self.frame_rate)
        if samples_per_frame < 1:
            raise ValueError("frame_rate cannot exceed sample_rate")
        output_samples = user_active.numel() * samples_per_frame
        padded = torch.nn.functional.pad(
            user[:output_samples], (0, max(0, output_samples - user.numel()))
        )
        time = torch.arange(
            output_samples, dtype=torch.float32, device=user.device
        ) / self.sample_rate
        sine = self.sine_amplitude * torch.sin(
            2 * math.pi * self.sine_frequency * time
        ).to(dtype=user.dtype)
        sample_active = user_active.repeat_interleave(samples_per_frame).to(user.device)
        sample_active &= torch.arange(output_samples, device=user.device) < user.numel()
        return torch.where(sample_active, padded, sine)

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
        text, dense_text, assistant_schedule, output_user_active = self._text_stream(
            entry, frame_length, user_intervals, assistant_intervals
        )
        original_assistant = assistant
        assistant = self._apply_sine_conditioning(assistant, assistant_schedule)
        user = self._apply_user_sine(user, output_user_active)
        if self.enable_sine_conditioning:
            frame_length = text.numel()
        result = {
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
        if self.enable_sine_conditioning:
            result.update(
                original_assistant_waveform=original_assistant,
                sine_mask=assistant_schedule.lt(0),
                sine_frames=int(assistant_schedule.lt(0).sum()),
                sine_duration=float(assistant_schedule.lt(0).sum()) / self.frame_rate,
            )
        return result


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


def write_natural_monologue_debug(
    item: dict[str, Any],
    output_prefix: str | Path,
    dataset: NaturalMonologueDataset,
) -> dict[str, Any]:
    """Write stereo audio, aligned masks/tokens, and sine statistics for one item.

    The WAV contains user audio on channel 0 and the conditioned assistant on
    channel 1. The PNG also includes the unmodified assistant waveform.
    Matplotlib is imported lazily so training has no visualization dependency.
    """
    import matplotlib.pyplot as plt

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    wav_path = prefix.with_suffix(".wav")
    plot_path = prefix.with_suffix(".png")
    stats_path = prefix.with_suffix(".json")
    user = item["waveform"].detach().cpu()
    original = item["original_assistant_waveform"].detach().cpu()
    modified = item["assistant_waveform"].detach().cpu()
    stereo = torch.stack([user, modified]).numpy()
    sphn.write_wav(str(wav_path), stereo, dataset.sample_rate)

    frame_length = int(item["audio_frame_length"])
    user_mask = torch.zeros(frame_length, dtype=torch.bool)
    assistant_mask = torch.zeros(frame_length, dtype=torch.bool)
    for target, key in (
        (user_mask, "user_vad_intervals"),
        (assistant_mask, "assistant_vad_intervals"),
    ):
        for start, end in item[key]:
            interval = dataset._frame_interval(start, end, frame_length)
            target[interval.start:interval.stop] = True
    tokens = item["assistant_text_tokens"][:frame_length].detach().cpu()
    lexical = ~torch.isin(
        tokens,
        torch.tensor(
            [dataset.wait_token_id, dataset.spk_token_id, dataset.hear_token_id]
        ),
    )
    sine_mask = item["sine_mask"].detach().cpu()
    time = torch.arange(user.numel()) / dataset.sample_rate
    frame_time = torch.arange(frame_length) / dataset.frame_rate
    panels = (
        ("User waveform", time, user),
        ("Original assistant", torch.arange(original.numel()) / dataset.sample_rate, original),
        ("Modified assistant", time, modified),
        ("User VAD", frame_time, user_mask),
        ("Assistant VAD", frame_time, assistant_mask),
        ("Assistant BPE", frame_time, lexical),
        ("Sine mask", frame_time, sine_mask),
        ("Final assistant text tokens", frame_time, tokens),
    )
    figure, axes = plt.subplots(len(panels), 1, figsize=(14, 14), sharex=False)
    for axis, (title, x, values) in zip(axes, panels, strict=True):
        axis.plot(x.numpy(), values.numpy(), linewidth=0.8)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    sine_frames = int(sine_mask.sum())
    stats = {
        "frames": frame_length,
        "sine_frames": sine_frames,
        "sine_frame_ratio": sine_frames / frame_length if frame_length else 0.0,
        "sine_duration_seconds": sine_frames / dataset.frame_rate,
        "wav": str(wav_path),
        "plot": str(plot_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


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
