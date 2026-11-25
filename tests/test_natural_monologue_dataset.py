"""Regression checks for contiguous Natural Monologue batches."""

import json
import math
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from full_duplex.natural_monologue_dataset import (
    NaturalMonologueDataset,
    collate_natural_monologue,
    clean_behavior_text,
    create_natural_monologue_dataloader,
    discover_behavior_sd_entries,
)
from full_duplex.model import FullDuplexConfig, MoshiFullDuplexModel, _delay_sequence


class Tokenizer:
    def encode(self, text, **kwargs):
        return {
            "내일은 맑습니다": [1, 2, 3, 4],
            "부산도 맑습니다": [5, 6, 3, 4],
            "짧게": [7],
            "아주 긴 텍스트": list(range(20, 32)),
            "I get that. Maybe, um, taking a break could help?": [30, 31, 32, 33],
            "Yeah, I think you're right.": [40, 41, 42],
            "Yeah, yeah.": list(range(50, 58)),
            "What you said about...": [60, 61, 62],
            "first second third": [21, 22, 23, 24],
            "fourth": [25],
        }[text]


class NaturalMonologueDatasetTests(unittest.TestCase):
    def test_agreed_twelve_frame_interleaving(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), enable_sine_conditioning=True)
            dataset.tokenizer = SimpleNamespace(encode=lambda *a, **kw: [21, 22, 23, 24, 25, 26])
            entry = {"path": "example.wav", "monologues": [{"text": "I think we should leave now", "start": 0.1}]}
            text, dense, schedule, user_active = dataset._text_stream(
                entry, 12, [(0.2, 0.5)], [(0.1, 0.7)]
            )
            self.assertEqual(dense.tolist(), [21, 22, 23, 24, 25, 26])
            # Distinct symbolic codec IDs represent each audible word.
            raw_assistant = torch.tensor([90, 21, 22, 23, 24, 25, 26, 90, 90, 90, 90, 90])
            assistant = torch.tensor([90 if i < 0 else raw_assistant[i] for i in schedule.tolist()])
            user = torch.tensor([90, 90, 31, 32, 33, 90, 90, 90, 90, 90, 90, 90])
            cfg = FullDuplexConfig(num_codebooks=1, sine_audio_token_ids=[90])
            stub = SimpleNamespace(
                full_duplex_config=cfg, audio_offset=1, user_audio_offset=2,
                _get_initial_token=lambda: torch.full((1, 3, 1), 99),
            )
            padding = MoshiFullDuplexModel.delay_padding(stub)
            delayed = _delay_sequence(cfg.delays, torch.stack([text, assistant, user])[None], padding)[0]
            self.assertEqual(delayed[0].tolist(), [10, 21, 11, 12, 12, 22, 23, 24, 25, 26, 11, 10])
            self.assertEqual(delayed[1].tolist(), [90, 90, 21, 90, 90, 90, 22, 23, 24, 25, 26, 90])
            self.assertEqual(delayed[2].tolist(), [90, 90, 90, 31, 32, 33, 90, 90, 90, 90, 90, 90])
            # Exercise waveform rendering too: source frames are moved intact.
            waveform = raw_assistant.float().repeat_interleave(10)
            rendered = dataset._apply_sine_conditioning(waveform, schedule).view(12, 10)
            for frame in (1, 5, 6, 7, 8, 9):
                torch.testing.assert_close(rendered[frame], waveform.view(12, 10)[schedule[frame]])
            self.assertEqual(user_active.nonzero().flatten().tolist(), [2, 3, 4])

    def test_user_only_turn_does_not_shift_future_assistant_onset(self):
        user = torch.tensor([True, True, False, False, False, False])
        assistant = torch.tensor([False, False, False, False, True, False])
        self.assertEqual(NaturalMonologueDataset._interleave_schedule(user, assistant), [None, None, None, None, 4, None])

    def test_repeated_interruptions_preserve_all_assistant_frames(self):
        user = torch.tensor([False, True, True, False, True, False])
        assistant = torch.tensor([True, True, True, True, False, True])
        self.assertEqual(
            NaturalMonologueDataset._interleave_schedule(user, assistant),
            [0, None, None, 1, None, 2, 3, 5],
        )

    def test_user_tail_is_flushed_before_final_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), texts=[{"text": "짧게", "start": 0.1}], enable_sine_conditioning=True)
            text, _, schedule, user = dataset._text_stream(dataset.entries[0], 10, [(0.8, 1.0)], [(0.1, 0.2)])
            self.assertEqual(text[-4:].tolist(), [10, 12, 12, 10])
            self.assertEqual(len(schedule), 12)
            self.assertEqual(user.nonzero().flatten().tolist(), [8, 9])

    def test_behavior_text_cleaner_preserves_speech_and_removes_markup(self):
        self.assertEqual(
            clean_behavior_text("<laughter>yeah</laughter>, [Oh, really?]"),
            "yeah, Oh, really?",
        )
        self.assertEqual(clean_behavior_text("**Yeah.** **(hesitation)**"), "Yeah.")
        self.assertEqual(clean_behavior_text("*Clears throat* hello"),
                         "Clears throat hello")
        self.assertEqual(clean_behavior_text("The roots are (x + 2)(x + 3)."),
                         "The roots are (x + 2)(x + 3).")
        self.assertIsNone(clean_behavior_text("[laughter] <sigh>"))

    def test_behavior_text_cleaner_rejects_corruption(self):
        for text in ("I was worried [", "yeah<!", "formula \\", "uh | okay"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    clean_behavior_text(text)

    def test_discovers_same_stem_behavior_sd_wav_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "0001.wav"
            sf.write(audio_path, np.zeros((100, 2), dtype=np.float32), 100)
            audio_path.with_suffix(".json").write_text(json.dumps({
                "file_name": "0001.flac",
                "utterances": [
                    {
                        "speaker_idx": 0,
                        "start_time": 0.2,
                        "tts_text": "짧게",
                        "backchannels": [],
                    },
                    {
                        "speaker_idx": 1,
                        "start_time": 0.4,
                        "tts_text": "ignored",
                        "backchannels": [{"start_time": 0.5, "tts_text": "짧게"}],
                    },
                ],
            }), encoding="utf-8")
            entries, stats = discover_behavior_sd_entries(f"{root}:ko")
            self.assertEqual(stats, {"audio": 1, "paired": 1, "invalid": 0})
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["path"], str(audio_path.resolve()))
            self.assertEqual(
                entries[0]["assistant_texts"],
                [
                    {"start": 0.2, "text": "짧게"},
                    {"start": 0.5, "text": "짧게"},
                ],
            )

    def test_discovery_preserves_assistant_end_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "0001.wav"
            sf.write(audio_path, np.zeros((400, 2), dtype=np.float32), 100)
            audio_path.with_suffix(".json").write_text(json.dumps({
                "file_name": "0001.flac",
                "utterances": [
                    {
                        "speaker_idx": 0,
                        "start_time": 0.5,
                        "end_time": 1.0,
                        "tts_text": "first",
                        "backchannels": [],
                    },
                    {
                        "speaker_idx": 0,
                        "start_time": 1.0,
                        "end_time": 2.0,
                        "tts_text": "second",
                        "backchannels": [],
                    },
                    {
                        "speaker_idx": 0,
                        "start_time": 2.5,
                        "end_time": 3.0,
                        "tts_text": "third",
                        "backchannels": [],
                    },
                ],
            }), encoding="utf-8")

            entries, stats = discover_behavior_sd_entries(f"{root}:en")

            self.assertEqual(stats, {"audio": 1, "paired": 1, "invalid": 0})
            self.assertEqual(entries[0]["assistant_texts"], [
                {"start": 0.5, "end": 1.0, "text": "first"},
                {"start": 1.0, "end": 2.0, "text": "second"},
                {"start": 2.5, "end": 3.0, "text": "third"},
            ])

    def test_touching_monologues_merge_before_tokenization(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), texts=[
                {"start_time": 0.1, "end_time": 0.2, "text": "first"},
                {"start_time": 0.2000001, "end_time": 0.3, "text": "second"},
                {"start_time": 0.3, "end_time": 0.5, "text": "third"},
                {"start_time": 0.7, "end_time": 0.9, "text": "fourth"},
            ])
            self.assertEqual(dataset.entries[0]["monologues"], [
                {"start": 0.1, "end": 0.5, "text": "first second third"},
                {"start": 0.7, "end": 0.9, "text": "fourth"},
            ])
            dataset.tokenizer = SimpleNamespace(encode=lambda text, **kw: {
                "first second third": [21, 22, 23, 24], "fourth": [25],
            }[text])
            text, dense, _, _ = dataset._text_stream(dataset.entries[0], 10, [], [])
            self.assertEqual(text[1:5].tolist(), [21, 22, 23, 24])
            self.assertEqual(text[7].item(), 25)
            self.assertEqual(dense.tolist(), [21, 22, 23, 24, 25])

    def test_overlapping_time_intervals_are_not_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), texts=[
                {"start": 0.1, "end": 0.5, "text": "짧게"},
                {"start": 0.4, "end": 0.8, "text": "짧게"},
            ])
            self.assertEqual(len(dataset.entries[0]["monologues"]), 2)

    def _dataset(
        self,
        root: Path,
        *,
        texts=None,
        enable_sine_conditioning=False,
    ) -> NaturalMonologueDataset:
        manifest = root / "natural.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "path": "dialog.wav",
                    "duration": 1.0,
                    "language": "ko",
                    "user_channel": 1,
                    "assistant_channel": 0,
                    "assistant_texts": texts or [
                        {"start": 0.1, "text": "내일은 맑습니다"}
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return NaturalMonologueDataset(
            manifest,
            Tokenizer(),
            wait_token_id=10,
            spk_token_id=11,
            hear_token_id=12,
            sample_rate=100,
            frame_rate=10,
            enable_sine_conditioning=enable_sine_conditioning,
        )

    def test_contiguous_bpe_then_hear_spk_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            # Channel 0 is assistant, channel 1 is user.
            audio = np.stack([np.full(100, 0.25), np.full(100, -0.5)])
            with patch(
                "full_duplex.natural_monologue_dataset.sphn.read",
                return_value=(audio, 100),
            ), patch.object(
                dataset,
                "_vad_intervals",
                side_effect=[[(0.4, 0.6)], [(0.1, 0.4), (0.6, 0.8)]],
            ):
                item = dataset[0]
            self.assertEqual(
                item["assistant_text_tokens"].tolist(),
                [10, 1, 2, 3, 4, 12, 11, 11, 10, 10],
            )
            self.assertEqual(item["text_only_tokens"].tolist(), [1, 2, 3, 4])
            self.assertTrue((item["waveform"] < 0).all())
            self.assertTrue((item["assistant_waveform"] > 0).all())

    def test_collate_and_loader_use_training_batch_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            audio = np.zeros((2, 100), dtype=np.float32)
            with patch(
                "full_duplex.natural_monologue_dataset.sphn.read",
                return_value=(audio, 100),
            ), patch.object(
                dataset,
                "_vad_intervals",
                side_effect=[
                    [(0.4, 0.6)], [(0.1, 0.4), (0.6, 0.8)],
                    [(0.4, 0.6)], [(0.1, 0.4), (0.6, 0.8)],
                ],
            ):
                item = dataset[0]
                batch = collate_natural_monologue([item])
                loader = create_natural_monologue_dataloader(
                    dataset, batch_size=1, shuffle=False, num_workers=0
                )
                loaded = next(iter(loader))
            self.assertEqual(batch["assistant_text_tokens"].shape, (1, 10))
            self.assertEqual(loaded["assistant_text_tokens"].tolist(),
                             [item["assistant_text_tokens"].tolist()])
            self.assertEqual(loaded["ids"], ["dialog"])

    def test_word_alignment_is_not_required(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            self.assertEqual(len(dataset), 1)
            self.assertNotIn("alignments", dataset.entries[0])

    def test_vad_float_boundary_does_not_add_a_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            interval = dataset._frame_interval(0.30000000000000004,
                                               0.6000000000000001, 10)
            self.assertEqual(list(interval), [3, 4, 5])

    def test_user_speech_pauses_text_under_sine_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(
                Path(directory), enable_sine_conditioning=True
            )
            entry = dataset.entries[0]
            text, _dense, schedule, user_active = dataset._text_stream(
                entry,
                frame_length=10,
                user_intervals=[(0.0, 0.5)],
                assistant_intervals=[(0.3, 0.6)],
            )
            self.assertEqual(
                text.tolist(),
                [10, 12, 12, 12, 12, 1, 2, 3, 4, 11, 11, 10],
            )
            self.assertEqual(
                schedule.tolist(),
                [-1, -1, -1, -1, -1, -1, -1, 3, 4, 5, -1, -1],
            )
            self.assertEqual(text[5:9].tolist(), [1, 2, 3, 4])
            self.assertEqual(user_active.tolist(), [True] * 5 + [False] * 7)

    def test_sine_pauses_audio_and_marks_user_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(
                Path(directory), enable_sine_conditioning=True
            )
            audio = np.stack([np.full(95, 0.25), np.full(95, -0.5)])
            with patch(
                "full_duplex.natural_monologue_dataset.sphn.read",
                return_value=(audio, 100),
            ), patch.object(
                dataset,
                "_vad_intervals",
                side_effect=[[(0.0, 0.5)], [(0.3, 0.6)]],
            ):
                item = dataset[0]
            self.assertEqual(item["waveform"].numel(), 120)
            self.assertEqual(item["assistant_waveform"].numel(), 120)
            self.assertEqual(item["assistant_text_tokens"].numel(), 12)
            # User speech is preserved for five frames; every no-user frame,
            # including the extended tail, is represented by sine.
            torch.testing.assert_close(item["waveform"][:50], torch.full((50,), -0.5))
            self.assertFalse((item["waveform"][50:] == 0).all())
            # Only real assistant audio is resumed; original inactive frames
            # become sine, including lexical-only frames preceding speech.
            torch.testing.assert_close(
                item["assistant_waveform"][70:100],
                item["original_assistant_waveform"][30:60],
            )
            sine = dataset.sine_amplitude * torch.sin(
                2 * math.pi * dataset.sine_frequency * torch.arange(120) / 100
            )
            torch.testing.assert_close(item["assistant_waveform"][:70], sine[:70], atol=2e-4, rtol=1e-4)
            torch.testing.assert_close(item["assistant_waveform"][100:], sine[100:], atol=2e-4, rtol=1e-4)

    def test_disabled_conditioning_is_bit_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            assistant = torch.linspace(-0.4, 0.4, 95)
            mask = torch.tensor([True] * 9 + [False])
            self.assertIs(dataset._apply_sine_conditioning(assistant, mask), assistant)

            audio = np.stack([
                np.full((95,), -0.5, dtype=np.float32),
                np.full((95,), 0.25, dtype=np.float32),
            ])
            with patch(
                "full_duplex.natural_monologue_dataset.sphn.read",
                return_value=(audio, 100),
            ), patch.object(dataset, "_vad_intervals", side_effect=[[], []]):
                item = dataset[0]
            self.assertNotIn("sine_mask", item)
            self.assertNotIn("original_assistant_waveform", item)
            self.assertNotIn("sine_frames", item)
            self.assertNotIn("sine_duration", item)

    def test_inactive_assistant_audio_is_sine_even_with_bpe(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(
                Path(directory), enable_sine_conditioning=True
            )
            _text, _dense, schedule, _user = dataset._text_stream(
                dataset.entries[0], 10, [], [(0.3, 0.6)]
            )
            self.assertEqual(schedule.tolist(), [-1, -1, -1, 3, 4, 5, -1, -1, -1, -1])

    def test_multiple_monologues_use_annotated_onset_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(
                Path(directory),
                texts=[
                    {"start": 0.1, "text": "짧게"},
                    {"start": 0.6, "text": "짧게"},
                ],
                enable_sine_conditioning=True,
            )
            text, dense, schedule, _user = dataset._text_stream(
                dataset.entries[0], 10, [(0.0, 0.8)], [(0.1, 0.2), (0.6, 0.7)]
            )
            self.assertEqual(dense.tolist(), [7, 7])
            self.assertEqual(text.tolist(), [10] + [12] * 7 + [7, 7, 11, 10])
            self.assertEqual(schedule.tolist(), [-1] * 8 + [1, 6, -1, -1])

    def test_behavior_sd_text_uses_exact_annotated_audio_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "natural.jsonl"
            manifest.write_text(json.dumps({
                "path": "dialog.wav",
                "duration": 31.0,
                "assistant_texts": [
                    {"start": 23.49968253968254,
                     "text": "I get that. Maybe, um, taking a break could help?"},
                    {"start": 28.733242630385487,
                     "text": "Yeah, I think you're right."},
                ],
            }) + "\n")
            dataset = NaturalMonologueDataset(
                manifest, Tokenizer(), wait_token_id=10, spk_token_id=11,
                hear_token_id=12, frame_rate=12.5, min_text_tokens=0,
            )
            text, _dense, _schedule, _user = dataset._text_stream(
                dataset.entries[0], 388, [], []
            )
            first_frame = math.floor(23.49968253968254 * 12.5)
            second_frame = math.floor(28.733242630385487 * 12.5)
            self.assertEqual(first_frame, 293)
            self.assertEqual(second_frame, 359)
            self.assertEqual(text[first_frame:first_frame + 4].tolist(), [30, 31, 32, 33])
            self.assertEqual(text[second_frame:second_frame + 3].tolist(), [40, 41, 42])

    def test_overlapping_bpe_timelines_are_rejected_without_moving_onsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "natural.jsonl"
            manifest.write_text(json.dumps({
                "path": "dialog.wav",
                "duration": 62.0,
                "assistant_texts": [
                    {"start": 60.52, "text": "Yeah, yeah."},
                    {"start": 61.06, "text": "What you said about..."},
                ],
            }) + "\n")
            dataset = NaturalMonologueDataset(
                manifest, Tokenizer(), wait_token_id=10, spk_token_id=11,
                hear_token_id=12, frame_rate=12.5, min_text_tokens=0,
            )
            self.assertEqual(math.floor(60.52 * 12.5), 756)
            self.assertEqual(math.floor(61.06 * 12.5), 763)
            with self.assertRaisesRegex(ValueError, "overlap at frame 763"):
                dataset._text_stream(dataset.entries[0], 775, [], [])

    def test_text_longer_than_audio_fails_when_sine_is_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(
                Path(directory),
                texts=[{"start": 0.5, "text": "아주 긴 텍스트"}],
                enable_sine_conditioning=True,
            )
            with self.assertRaisesRegex(ValueError, "complete text timeline"):
                dataset._text_stream(
                    dataset.entries[0], 10, [(0.0, 1.0)], [(0.5, 0.8)]
                )


if __name__ == "__main__":
    unittest.main()
