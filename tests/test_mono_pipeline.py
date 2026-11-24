"""Regression checks for folder-based mono post-training."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from full_duplex.dataset import (
    MonoPostTrainingDataset,
    RecoverableSampleError,
    collate_duplex,
)
from full_duplex.train import SkipInvalidDataset


class Tokenizer:
    def encode(self, text, **kwargs):
        return [2, 3, 4] if text == ' multi' else [5]


class MonoPipelineTests(unittest.TestCase):
    def test_stereo_folder_uses_native_user_and_assistant_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = np.stack([np.full(96000, 0.25), np.full(96000, -0.5)], axis=1)
            sf.write(root / 'sample.wav', audio, 24000)
            (root / 'sample.alignment.json').write_text(json.dumps({
                'assistant_channel': 0,
                'speakers': ['assistant', 'user'],
                'channel_alignments': {
                    '0': [['hello', [1, 2], 'assistant']],
                    '1': [['there', [1, 2], 'user']],
                },
                'alignments': [['hello', [1, 2], 'assistant']],
            }))
            ds = MonoPostTrainingDataset(
                None, Tokenizer(), folders=directory + ':en', audio_mode='stereo',
                text_padding_id=10, text_epad_id=11, min_duration=0,
            )
            item = ds[0]
            self.assertTrue(item['user_audio_present'])
            self.assertTrue((item['waveform'] < 0).all())
            self.assertTrue((item['assistant_waveform'] > 0).all())

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
            ds = MonoPostTrainingDataset(None, Tokenizer(), entries=entries,
                text_padding_id=10, text_epad_id=11)
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
                ds = MonoPostTrainingDataset(None, Tokenizer(), folders=directory + ':ko',
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

    def test_nonfinite_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            MonoPostTrainingDataset._validate_alignments([['word', [0, float('nan')], 'S']], 4)

    def test_minimum_mono_bpe_tokens_eager_and_online(self):
        for lazy in (False, True):
            for count in (4, 5):
                with self.subTest(lazy=lazy, tokens=count):
                    words = [['utterance', [0, 4], 'S']]
                    tokenizer = Mock()
                    tokenizer.encode.return_value = list(range(20, 20 + count))
                    entry = {'path': 'sample.wav', 'duration': 4, 'language': 'en',
                             'alignments': None if lazy else words}
                    kwargs = dict(text_padding_id=10, text_epad_id=11,
                                  min_text_tokens=5)
                    with patch('full_duplex.dataset.discover_speech_entries',
                               return_value=([entry], {})):
                        if lazy:
                            ds = MonoPostTrainingDataset(None, tokenizer, folders='unused:en',
                                                         online_alignment=True, **kwargs)
                            with patch.object(ds, '_extract_alignment', return_value=words):
                                if count < 5:
                                    with self.assertRaisesRegex(RecoverableSampleError,
                                                               'text_tokens=4, min_text_tokens=5'):
                                        ds._prepare_lazy_entry(0)
                                    self.assertEqual(ds.rejected['text_tokens'], 1)
                                else:
                                    ds._prepare_lazy_entry(0)
                        elif count < 5:
                            with self.assertRaisesRegex(ValueError, 'empty mono post-training dataset'):
                                MonoPostTrainingDataset(None, tokenizer, entries=[entry], **kwargs)
                        else:
                            ds = MonoPostTrainingDataset(None, tokenizer, entries=[entry], **kwargs)
                            self.assertEqual(len(ds), 1)
                        tokenizer.encode.assert_any_call(' utterance', add_special_tokens=False)
