"""Regression checks for contiguous Natural Monologue batches."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from full_duplex.natural_monologue_dataset import (
    NaturalMonologueDataset,
    collate_natural_monologue,
    create_natural_monologue_dataloader,
    discover_behavior_sd_entries,
)


class Tokenizer:
    def encode(self, text, **kwargs):
        return {
            "내일은 맑습니다": [1, 2, 3, 4],
            "부산도 맑습니다": [5, 6, 3, 4],
            "짧게": [7],
            "아주 긴 텍스트": list(range(20, 32)),
        }[text]


class NaturalMonologueDatasetTests(unittest.TestCase):
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

    def test_discovery_merges_touching_assistant_utterances(self):
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
                {"start": 0.5, "text": "first second"},
                {"start": 2.5, "text": "third"},
            ])

    def _dataset(
        self,
        root: Path,
        *,
        texts=None,
        enable_sine_conditioning=False,
        monologue_lead_frames=1,
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
            monologue_lead_frames=monologue_lead_frames,
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
                [1, 2, 3, 4, 12, 12, 11, 11, 10, 10],
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

    def test_sine_mask_covers_only_hear_overwritten_by_bpe_during_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(
                Path(directory), enable_sine_conditioning=True
            )
            entry = dataset.entries[0]
            text, _dense, sine_mask = dataset._text_stream(
                entry,
                frame_length=10,
                user_intervals=[(0.0, 0.5)],
                assistant_intervals=[(0.3, 0.6)],
            )
            self.assertEqual(text.tolist(), [1, 2, 3, 4, 12, 11, 10, 10, 10, 10])
            # Frames 0-2: HEAR overwritten by BPE while assistant is silent.
            self.assertEqual(
                sine_mask.tolist(),
                [True, True, True, False, False, False, False, False, False, False],
            )
            # BPE+assistant speech, HEAR-only, WAIT and SPK are all excluded.
            self.assertFalse(sine_mask[3])
            self.assertFalse(sine_mask[4])
            self.assertFalse(sine_mask[5])
            self.assertFalse(sine_mask[6:].any())

    def test_sine_is_inserted_without_changing_user_or_lengths(self):
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
            torch.testing.assert_close(
                item["waveform"], torch.full((95,), -0.5)
            )
            self.assertEqual(item["assistant_waveform"].numel(), 95)
            self.assertEqual(item["assistant_text_tokens"].numel(), 10)
            # Selected frames changed, actual assistant speech and incomplete
            # final frame remain bit-identical to the source.
            self.assertFalse(torch.equal(
                item["assistant_waveform"][:30],
                item["original_assistant_waveform"][:30],
            ))
            torch.testing.assert_close(
                item["assistant_waveform"][30:],
                item["original_assistant_waveform"][30:],
            )

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

    def test_user_silence_produces_no_sine_for_bpe(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(
                Path(directory), enable_sine_conditioning=True
            )
            _text, _dense, sine_mask = dataset._text_stream(
                dataset.entries[0], 10, [], [(0.3, 0.6)]
            )
            self.assertFalse(sine_mask.any())

    def test_lead_zero_and_multiple_monologues(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(
                Path(directory),
                texts=[
                    {"start": 0.1, "text": "짧게"},
                    {"start": 0.6, "text": "짧게"},
                ],
                enable_sine_conditioning=True,
                monologue_lead_frames=0,
            )
            text, dense, sine_mask = dataset._text_stream(
                dataset.entries[0], 10, [(0.0, 0.8)], [(0.1, 0.2), (0.6, 0.7)]
            )
            self.assertEqual(dense.tolist(), [7, 7])
            self.assertEqual(text[1].item(), 7)
            self.assertEqual(text[6].item(), 7)
            self.assertFalse(sine_mask.any())  # Both BPE frames contain assistant speech.

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
