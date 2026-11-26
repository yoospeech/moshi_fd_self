"""Offline regression checks: python -m unittest discover -s tests -v."""

import tempfile
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaModel

from full_duplex.dataset import StereoDialogueDataset, MonoPostTrainingDataset, TextAlignmentCollision
from full_duplex.model import (
    FullDuplexConfig, MoshiFullDuplexModel, masked_multistream_loss,
)
from full_duplex.inference import (
    generate_assistant_text, generate_assistant, load_config,
)
from full_duplex.train import (
    acoustic_prompt_mask, mask_audio_batch_text, mix_text_only_samples,
    resolve_model_source, text_only_streams, uses_mixed_text_training,
)
from moshi.models.lm_utils import _delay_sequence


class AlignmentTests(unittest.TestCase):
    def test_audio_macro_cross_entropy_averages_token_ids_equally(self):
        streams = torch.tensor([[[10, 10, 10], [1, 1, 2]]])
        text_logits = torch.zeros(1, 1, 3, 16)
        text_mask = torch.ones(1, 1, 3, dtype=torch.bool)
        audio_logits = torch.tensor([[[
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
        ]]])
        audio_mask = torch.ones(1, 1, 3, dtype=torch.bool)
        _, metrics = masked_multistream_loss(
            text_logits, text_mask, streams,
            audio_logits=audio_logits,
            audio_mask=audio_mask,
            audio_token_balance_alpha=1.0,
        )
        per_frame = torch.nn.functional.cross_entropy(
            audio_logits[0, 0], streams[0, 1], reduction="none"
        )
        expected = (per_frame[:2].mean() + per_frame[2]) / 2
        torch.testing.assert_close(
            metrics["assistant_audio/rq/0_loss"], expected
        )

    def test_audio_accuracy_excludes_sine_frames_but_loss_includes_them(self):
        streams = torch.tensor([[[10, 10, 10], [1, 2, 3]]])
        text_logits = torch.zeros(1, 1, 3, 16)
        text_mask = torch.ones(1, 1, 3, dtype=torch.bool)
        audio_logits = torch.zeros(1, 1, 3, 8)
        audio_logits[0, 0, 0, 1] = 1
        audio_logits[0, 0, 1, 0] = 1  # Wrong, but this frame is sine.
        audio_logits[0, 0, 2, 3] = 1
        audio_mask = torch.ones(1, 1, 3, dtype=torch.bool)
        _, metrics = masked_multistream_loss(
            text_logits, text_mask, streams,
            audio_logits=audio_logits,
            audio_mask=audio_mask,
            audio_accuracy_mask=torch.tensor([[True, False, True]]),
        )
        self.assertEqual(metrics["assistant_rq_accuracy_1"].item(), 1.0)
        self.assertEqual(metrics["assistant_audio/accuracy"].item(), 1.0)
        expected_loss = torch.nn.functional.cross_entropy(
            audio_logits[0, 0], streams[0, 1]
        )
        torch.testing.assert_close(
            metrics["assistant_audio/rq/0_loss"], expected_loss
        )

    def test_lexical_accuracy_excludes_padding_epad_and_control_tokens(self):
        targets = torch.tensor([[[10, 11, 12, 13, 14, 2, 3]]])
        logits = torch.zeros(1, 1, 7, 16)
        # Special tokens are intentionally predicted incorrectly. Of the two
        # lexical tokens, token 2 is correct and token 3 is incorrect.
        logits[..., 0] = 1
        logits[0, 0, 5, 2] = 2
        mask = torch.ones(1, 1, 7, dtype=torch.bool)
        _, metrics = masked_multistream_loss(
            logits, mask, targets,
            text_padding_id=10,
            text_epad_id=11,
            text_control_ids=(12, 13, 14),
        )
        self.assertEqual(metrics["assistant/lexical_accuracy"].item(), 0.5)

    def test_stereo_minimum_bpe_tokens(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [2]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0.json").write_text(json.dumps({"alignments": [
                ["assistant", [.8, 1.0], "SPEAKER_MAIN"]
            ]}))
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"path": "0.wav", "duration": 2}) + "\n")
            with self.assertRaisesRegex(ValueError, "empty manifest"):
                StereoDialogueDataset(
                    manifest, Tokenizer(), text_padding_id=10, text_epad_id=11,
                    min_duration=0, min_text_tokens=2,
                    min_speech_coverage=0, max_speech_coverage=1,
                )

    def test_mono_training_uses_full_audio_without_user_prompt(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {" hello": [6], " world": [7], "hello world": [6, 7]}[text]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.jsonl").write_text(json.dumps({
                "path": "0.wav", "duration": 4,
                "alignments": [["hello", [0, 2], "S"], ["world", [2, 4], "S"]],
            }) + "\n")
            dataset = MonoPostTrainingDataset(
                root / "manifest.jsonl", Tokenizer(), text_padding_id=10, text_epad_id=11,
                min_duration=0,
            )
            audio = torch.linspace(-1, 1, 96000, dtype=torch.float32).numpy()
            with patch("full_duplex.dataset.sphn.read", return_value=(audio, 24000)):
                item = dataset[0]
            self.assertTrue((item["waveform"] == 0).all())
            self.assertTrue((item["assistant_waveform"] == torch.from_numpy(audio)).all())
            self.assertIn(6, item["assistant_text_tokens"].tolist())
            self.assertIn(7, item["assistant_text_tokens"].tolist())

    def test_audio_text_mask_preserves_epad_and_original_targets(self):
        streams = torch.full((2, 9, 5), -1, dtype=torch.long)
        streams[:, 0] = torch.tensor([10, 11, 3, 4, 10])
        masked, targets = mask_audio_batch_text(
            streams, padding_id=10, epad_id=11, probability=1.0,
            acoustic_mask=torch.tensor([True, False]),
        )
        self.assertEqual(masked[0, 0].tolist(), [10, 11, 10, 10, 10])
        self.assertEqual(targets[0, 0].tolist(), [10, 11, 3, 4, 10])
        self.assertEqual(masked[1, 0].tolist(), [10, 11, 3, 4, 10])

    def test_mixed_batch_contains_audio_and_dense_text_rows(self):
        mask = acoustic_prompt_mask(4, 0.5, device="cpu")
        self.assertEqual(mask.sum().item(), 2)
        streams = torch.full((4, 9, 3), -1, dtype=torch.long)
        streams[:, 0] = 7
        streams[mask, 1:5] = 2
        batch = {"text_only_tokens": torch.tensor([
            [1, 2, 3, 4], [5, 6, -1, -1], [7, 8, 9, -1], [3, 2, 1, -1]
        ])}
        mixed = mix_text_only_samples(streams, batch, mask)
        self.assertEqual(mixed.shape[-1], 4)
        self.assertTrue((mixed[~mask, 1:] == -1).all())
        self.assertTrue((mixed[mask, 1:5, :3] == 2).all())
        self.assertTrue(torch.equal(mixed[~mask, 0], batch["text_only_tokens"][~mask]))

    def test_natural_monologue_finetuning_uses_mixed_text_rows(self):
        self.assertTrue(uses_mixed_text_training("finetuning", True))
        self.assertTrue(uses_mixed_text_training("mono_post_training", False))
        self.assertFalse(uses_mixed_text_training("finetuning", False))

    def test_text_only_stream_uses_entire_batch_and_no_audio(self):
        batch = {"text_only_tokens": torch.tensor([[1, 2, -1], [3, 4, 5]])}
        streams = text_only_streams(batch, torch.device("cpu"), FullDuplexConfig())
        self.assertEqual(streams[:, 0].tolist(), [[1, 2, -1], [3, 4, 5]])
        self.assertTrue((streams[:, 1:] == -1).all())

    def test_manifest_target_and_audio_channel(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {" assistant": [2]}[text]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0.json").write_text(json.dumps({"alignments": [
                ["assistant", [.8, 1.0], "SPEAKER_MAIN"]
            ]}))
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "path": "0.wav", "duration": 2, "user_channel": 0,
                "assistant_channel": 1,
            }) + "\n")
            ds = StereoDialogueDataset(
                manifest, Tokenizer(), text_padding_id=10, text_epad_id=11,
                min_duration=0, min_text_tokens=1,
                min_speech_coverage=0, max_speech_coverage=1,
            )
            audio = torch.stack([torch.ones(48000), torch.zeros(48000)]).numpy()
            with patch("full_duplex.dataset.sphn.read", return_value=(audio, 24000)):
                item = ds[0]
            self.assertTrue((item["waveform"] == 1).all())
            self.assertTrue((item["assistant_waveform"] == 0).all())
            self.assertEqual(item["assistant_text_tokens"][10].item(), 2)

    def test_padding_and_collision(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {" multi": [1, 2], " next": [3]}[text]

        ds = object.__new__(StereoDialogueDataset)
        ds.tokenizer = Tokenizer()
        ds.frame_rate = 12.5
        ds.text_padding_id, ds.text_epad_id = 10, 11
        user = [["multi", [0, .2], "SPEAKER_USER"],
                ["next", [.24, .4], "SPEAKER_USER"]]
        a = ds._aligned_text_stream(user, 8)
        self.assertEqual(a.tolist(), [11, 1, 2, 3, 10, 10, 10, 10])
        user[1][1][0] = .16
        self.assertEqual(ds._aligned_text_stream(user, 8).tolist(), a.tolist())
        # Same-start words retain all pieces, even beyond the audio tail.
        user[1][1][0] = 0
        self.assertEqual(ds._aligned_text_stream(user, 2).tolist(), [11, 1, 2, 3])
        # A later gap absorbs the shift; timestamps are not modified.
        user.append(["next", [.56, .64], "SPEAKER_USER"])
        self.assertEqual(ds._aligned_text_stream(user, 8).tolist(),
                         [11, 1, 2, 3, 10, 10, 11, 3])
        self.assertEqual(user[1][1][0], 0)


class ModelTests(unittest.TestCase):
    def test_moshi_interleaving_delay_for_both_audio_streams(self):
        cfg = FullDuplexConfig(num_codebooks=4, acoustic_delay_steps=1)
        self.assertEqual(cfg.delays, [0, 1, 1, 1, 1, 1, 1, 1, 1])
        raw = torch.arange(9 * 4).view(1, 9, 4)
        initial = torch.full((1, 9, 1), 99, dtype=torch.long)
        delayed = _delay_sequence(cfg.delays, raw, initial)
        for stream in (0,):
            torch.testing.assert_close(delayed[:, stream], raw[:, stream])
        for stream in range(1, 9):
            self.assertEqual(delayed[0, stream, 0].item(), 99)
            torch.testing.assert_close(
                delayed[0, stream, 1:], raw[0, stream, :-1]
            )

    def test_models_directory_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "models"
            path, repository = resolve_model_source(
                str(root / "SmolLM2-135M"), str(root)
            )
            self.assertEqual(path, root / "SmolLM2-135M")
            self.assertEqual(repository, "HuggingFaceTB/SmolLM2-135M")
            path.mkdir(parents=True)
            existing, repository = resolve_model_source(str(path), str(root))
            self.assertEqual(existing, path)
            self.assertIsNone(repository)

    def test_inference_rejects_incompatible_delay_pattern(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            base = {
                "text_context_mode": "moshi",
                "text_delay_seconds": 0.08,
                "acoustic_delay_steps": 1,
            }
            config_path.write_text(json.dumps(base))
            args = SimpleNamespace(
                checkpoint=str(Path(directory) / "checkpoint.pt"),
                config=str(config_path),
                model=None,
            )
            self.assertEqual(load_config(args).delays, [0, 1, 1, 1, 1, 1, 1, 1, 1])
            base["acoustic_delay_steps"] = 2
            config_path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "expected"):
                load_config(args)

    def test_inference_training_parity_and_gradients(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            LlamaModel(LlamaConfig(
                vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2,
            )).save_pretrained(directory)
            cfg = FullDuplexConfig(
                temporal_model=directory, text_card=34,
                text_padding_id=32, text_epad_id=33, audio_card=16,
                depformer_dim=32, depformer_heads=4, depformer_layers=2,
                depformer_ffn_dim=64,
                sine_audio_token_ids=[1, 2, 3, 4],
            )
            model = MoshiFullDuplexModel(cfg, device="cpu", dtype=torch.float32)
            model.eval()
            torch.testing.assert_close(
                model.delay_padding()[0, 1:, 0], torch.tensor([1, 2, 3, 4] * 2)
            )
            self.assertEqual(model.checkpoint_metadata()["sine_audio_token_ids"], [1, 2, 3, 4])
            self.assertEqual(cfg.delays, [0, 1, 1, 1, 1, 1, 1, 1, 1])
            causal_probe = torch.randint(0, 16, (1, 9, 12))
            with patch.object(model, "forward_text", wraps=model.forward_text) as recorded:
                model.forward_training(causal_probe, predict_audio=False)
                torch.testing.assert_close(recorded.call_args.args[0][0, 1:, 1], torch.tensor([1, 2, 3, 4] * 2))
                text_only = causal_probe.clone()
                text_only[:, 1:] = -1
                model.forward_training(text_only, predict_audio=False)
                self.assertTrue((recorded.call_args.args[0][0, 1:, 1] == -1).all())
            with torch.no_grad():
                reference = model.forward_training(
                    causal_probe, predict_audio=False
                )[0]
                changed = causal_probe.clone()
                changed[:, 1, 4] = (changed[:, 1, 4] + 1) % cfg.audio_card
                current = model.forward_training(changed, predict_audio=False)[0]
                self.assertTrue(torch.isfinite(current[:, 0, 4]).all())
                changed = causal_probe.clone()
                changed[:, 1:, 5:] = (changed[:, 1:, 5:] + 1) % cfg.audio_card
                future = model.forward_training(changed, predict_audio=False)[0]
            audio = torch.randint(0, 16, (1, 4, 8))
            observed = []
            forward = model.forward_text

            def record(prefix):
                result = forward(prefix)
                observed.append(result[1][:, :, -1].clone())
                return result

            model.forward_text = record
            a = generate_assistant_text(
                model, audio, cfg, temperature=0, top_k=0, greedy=True,
            )
            model.forward_text = forward
            self.assertEqual(len(a), 8)
            generated_text, generated_audio = generate_assistant(
                model, audio[:, :, :3], cfg,
                text_temperature=0, text_top_k=0,
                audio_temperature=0, audio_top_k=0, greedy=True,
            )
            self.assertEqual(len(generated_text), 3)
            self.assertEqual(generated_audio.shape, (1, 4, 3))
            self.assertTrue((generated_audio >= 0).all())
            # Replay generated raw streams through training: every greedy
            # prediction must match, including delay startup and flush frames.
            replay = torch.full((1, 9, 4), -1, dtype=torch.long)
            replay[:, 0] = cfg.text_padding_id
            replay[0, 0, :3] = torch.tensor(generated_text)
            replay[:, 1:5, :3] = generated_audio
            replay[:, 5:, :3] = audio[:, :, :3]
            replay_text, _, replay_audio, _ = model.forward_training(replay)
            torch.testing.assert_close(
                replay_text[0, 0, :3].argmax(-1), torch.tensor(generated_text)
            )
            torch.testing.assert_close(
                replay_audio[:, :, :3].argmax(-1), generated_audio
            )
            streams = torch.full((1, 9, 8 + max(cfg.delays)), -1, dtype=torch.long)
            streams[:, 0] = cfg.text_padding_id
            streams[0, 0, :8] = torch.tensor(a)
            streams[:, 5:, :8] = audio
            logits, mask, audio_logits, audio_mask = model.forward_training(streams)
            for t in range(8):
                self.assertTrue(mask[0, 0, t])
                torch.testing.assert_close(logits[0, 0, t], observed[t][0, 0], atol=1e-5, rtol=1e-5)
            # The reference inference has no assistant feedback. Test parity
            # with identical inputs, then separately exercise audio training.
            self.assertFalse(audio_mask.any())
            empty_loss, empty_metrics = masked_multistream_loss(
                logits, mask, streams, audio_logits=audio_logits, audio_mask=audio_mask,
            )
            self.assertTrue(torch.isfinite(empty_loss))
            self.assertEqual(empty_metrics['loss/assistant_audio'].item(), 0)
            streams[:, 1:5, :8] = torch.randint(0, 16, (1, 4, 8))
            logits, mask, audio_logits, audio_mask = model.forward_training(streams)
            self.assertEqual(audio_logits.shape, (1, 4, streams.shape[-1], 16))
            self.assertTrue(audio_mask[:, :, :8].all())
            text_logits, text_mask, no_audio_logits, no_audio_mask = (
                model.forward_training(streams, predict_audio=False)
            )
            torch.testing.assert_close(text_logits, logits)
            torch.testing.assert_close(text_mask, mask)
            self.assertIsNone(no_audio_logits)
            self.assertIsNone(no_audio_mask)
            loss, before = masked_multistream_loss(
                logits, mask, streams, audio_logits=audio_logits, audio_mask=audio_mask,
            )
            user_changed = streams.clone()
            user_changed[:, 5:] = 0
            same_loss, metrics = masked_multistream_loss(
                logits, mask, user_changed, audio_logits=audio_logits, audio_mask=audio_mask,
            )
            torch.testing.assert_close(loss, same_loss)
            self.assertFalse(any('user_audio' in key for key in metrics))
            for rq in range(1, 5):
                self.assertIn(f"assistant_rq_accuracy_{rq}", metrics)
            # Future feedback must not alter earlier temporal predictions.
            future_streams = streams.clone()
            future_streams[:, 1:5, 4:] = 0
            future_logits = model.forward_training(future_streams)[0]
            torch.testing.assert_close(logits[:, :, :5], future_logits[:, :, :5])
            self.assertGreater((logits - model.forward_training(
                streams.clone().index_fill(1, torch.arange(1, 5), -1)
            )[0]).abs().sum().item(), 0)
            changed = streams.clone()
            changed[:, 0] = (changed[:, 0] + 1) % cfg.text_card
            _, after = masked_multistream_loss(logits, mask, changed)
            self.assertNotEqual(before["loss/assistant"].item(), after["loss/assistant"].item())
            loss.backward()
            for head in (model.text_linear, model.linears[0]):
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                self.assertGreater(head.weight.grad.abs().sum().item(), 0)
            checkpoint = Path(directory) / "roundtrip.pt"
            torch.save(model.state_dict(), checkpoint)
            model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)


if __name__ == "__main__":
    unittest.main()
