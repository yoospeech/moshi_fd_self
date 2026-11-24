"""Regression checks for contiguous Natural Monologue batches."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from full_duplex.natural_monologue_dataset import (
    NaturalMonologueDataset,
    collate_natural_monologue,
    create_natural_monologue_dataloader,
)


class Tokenizer:
    def encode(self, text, **kwargs):
        return {
            "내일은 맑습니다": [1, 2, 3, 4],
            "부산도 맑습니다": [5, 6, 3, 4],
        }[text]


class NaturalMonologueDatasetTests(unittest.TestCase):
    def _dataset(self, root: Path) -> NaturalMonologueDataset:
        manifest = root / "natural.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "path": "dialog.wav",
                    "duration": 1.0,
                    "language": "ko",
                    "user_channel": 1,
                    "assistant_channel": 0,
                    "assistant_texts": [
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


if __name__ == "__main__":
    unittest.main()
