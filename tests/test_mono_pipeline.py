"""Regression checks for folder-based mono post-training."""
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from full_duplex.dataset import (
    MonoContinuationDataset,
    RecoverableSampleError,
    collate_duplex,
)
from full_duplex.train import SkipInvalidDataset


class Tokenizer:
    def encode(self, text, **kwargs):
        return [2, 3, 4] if text == ' multi' else [5]


class MonoPipelineTests(unittest.TestCase):
    def test_lazy_invalid_sample_is_skipped_and_cached_within_split(self):
        class Samples:
            def __init__(self):
                self.calls = []

            def __len__(self):
                return 3

            def __getitem__(self, index):
                self.calls.append(index)
                if index == 0:
                    raise RecoverableSampleError('collision')
                return index

        samples = Samples()
        dataset = SkipInvalidDataset(samples)
        self.assertEqual(dataset[0], 1)
        self.assertEqual(dataset[0], 1)
        self.assertEqual(samples.calls, [0, 1, 1])

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

    def test_suffix_text_filters_apply_to_mono_target(self):
        entries = [{
            'path': 'short-target.wav',
            'duration': 6,
            'alignments': [
                ['prefix', [0, 2], 'S'],
                ['only', [2, 3], 'S'],
                ['two', [3, 4], 'S'],
            ],
        }]
        with patch.object(MonoContinuationDataset, '_split_time', return_value=2):
            with self.assertRaisesRegex(ValueError, 'empty mono continuation dataset'):
                MonoContinuationDataset(
                    None,
                    Tokenizer(),
                    entries=entries,
                    text_padding_id=10,
                    text_epad_id=11,
                    min_words=3,
                    min_words_per_second=0.5,
                    max_words_per_second=2.5,
                    gap_seconds=(0, 0),
                )

    def test_suffix_density_boundaries_and_lazy_filtering(self):
        words = [['prefix', [0, 2], 'S'], ['one', [2, 2.2], 'S'],
                 ['two', [2.5, 2.7], 'S']]
        cases = [
            (6, 2, 0.5, 2.5, None),  # Exactly the minimum density.
            (6, 3, 0.5, 2.5, 'words'),
            (7, 2, 0.5, 2.5, 'words_per_second'),
            (4, 2, 0.0, 0.9, 'words_per_second'),
            (4, 2, 0.0, 1.0, None),  # Exactly the maximum density.
        ]
        for duration, minimum, low, high, rejection in cases:
            for lazy in (False, True):
                with self.subTest(duration=duration, minimum=minimum, lazy=lazy):
                    entry = {'path': 'sample.wav', 'duration': duration,
                             'language': 'en', 'alignments': None if lazy else words}
                    kwargs = dict(text_padding_id=10, text_epad_id=11,
                                  min_words=minimum, min_words_per_second=low,
                                  max_words_per_second=high, gap_seconds=(0, 0))
                    with patch('full_duplex.dataset.discover_mono_continuation_entries',
                               return_value=([entry], {})), patch.object(
                                   MonoContinuationDataset, '_split_time', return_value=2):
                        if lazy:
                            ds = MonoContinuationDataset(None, Tokenizer(), folders='unused:en',
                                                         online_alignment=True, **kwargs)
                            with patch.object(ds, '_extract_alignment', return_value=words):
                                if rejection:
                                    with self.assertRaisesRegex(RecoverableSampleError, rejection):
                                        ds._prepare_lazy_entry(0)
                                    self.assertEqual(ds.rejected[rejection], 1)
                                else:
                                    self.assertEqual(ds._prepare_lazy_entry(0)['split_seconds'], 2)
                        elif rejection:
                            with self.assertRaisesRegex(ValueError, 'empty mono continuation dataset'):
                                MonoContinuationDataset(None, Tokenizer(), entries=[entry], **kwargs)
                        else:
                            ds = MonoContinuationDataset(None, Tokenizer(), entries=[entry], **kwargs)
                            self.assertEqual(len(ds), 1)
