"""Regression checks for folder-based mono post-training."""
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from full_duplex.dataset import MonoContinuationDataset, collate_duplex


class Tokenizer:
    def encode(self, text, **kwargs):
        return [2, 3, 4] if text == ' multi' else [5]


class MonoPipelineTests(unittest.TestCase):
    def test_bad_annotations_and_collision_are_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / 'bad.json'
            broken.write_text('{')
            entries = [
                {'path': 'good.wav', 'alignments': [['prefix', [0, 2], 'S'], ['end', [2, 4], 'S']]},
                {'path': 'collision.wav', 'alignments': [['prefix', [0, 2], 'S'], ['multi', [2, 2.08], 'S'], ['end', [2.08, 4], 'S']]},
                {'path': 'empty.wav', 'alignments': []},
                {'path': str(Path(directory) / 'missing.wav')},
                {'path': 'broken.wav', 'alignment_path': str(broken)},
            ]
            for entry in entries:
                entry['duration'] = 4
            with patch.object(MonoContinuationDataset, '_split_time', return_value=2):
                ds = MonoContinuationDataset(None, Tokenizer(), entries=entries,
                    text_padding_id=10, text_epad_id=11, gap_seconds=(0, 0))
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds.rejected['alignment'], 3)
            self.assertEqual(ds.rejected['text_collision'], 1)

    def test_online_extraction_language_and_no_json_writes(self):
        annotation = types.ModuleType('scripts.annotate')
        annotation.whisper = types.SimpleNamespace(load_model=Mock(return_value=object()))
        annotation.process_one = Mock(return_value={'alignments': [
            ['prefix', [0, 2], 'S'], ['end', [2, 4], 'S']]})
        with tempfile.TemporaryDirectory() as directory:
            sf.write(Path(directory) / 'sample.wav', np.zeros(96000), 24000)
            with patch.dict('sys.modules', {'scripts.annotate': annotation}):
                ds = MonoContinuationDataset(None, Tokenizer(), folders=directory + ':ko',
                    online_alignment=True, text_padding_id=10, text_epad_id=11)
                self.assertFalse(annotation.process_one.called)
                batch = collate_duplex([ds[0]])
                ds[0]
            self.assertEqual(annotation.process_one.call_args.args[2], 'ko')
            self.assertIsNone(annotation.process_one.call_args.args[1])
            self.assertEqual(annotation.whisper.load_model.call_count, 1)
            self.assertEqual(annotation.process_one.call_count, 1)
            self.assertFalse(list(Path(directory).glob('*.json')))
            self.assertEqual(batch['languages'], ['ko'])
            self.assertEqual(batch['waveforms'].shape, batch['assistant_waveforms'].shape)

    def test_overlapping_words_cannot_be_split(self):
        words = [['one', [0, 2], 'S'], ['two', [1.9, 3], 'S']]
        self.assertIsNone(MonoContinuationDataset._split_time(4, words, 0.5, 1, 1))

    def test_nonfinite_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            MonoContinuationDataset._validate_alignments([['word', [0, float('nan')], 'S']], 4)
