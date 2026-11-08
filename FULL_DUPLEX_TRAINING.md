# Moshi Full-Duplex Training

The trainer learns one text stream from user audio:

```text
assistant PAD/EPAD text + user Mimi audio
    -> shared causal Temporal Transformer
    -> assistant text head + Moshi Depth Transformer
```

User speech is condition-only and has no text target. The manifest may contain `user_hyp`, but it
is ignored. Assistant targets are loaded from the JSON sidecar matching each
WAV stem.

## Interleaving

```text
streams: [assistant_text, user_semantic, user_acoustic_1, user_acoustic_2, user_acoustic_3]
delays:  [0,              0,             1,               1,               1]
```

Assistant text uses Moshi PAD/EPAD alignment at 12.5 Hz. Semantic audio is
current-frame; acoustic codebooks are delayed by one frame.

## Training

Use the complete tested command in
[README.md](README.md#two-gpu-ddp-training). It reads the original
`dailytalk.jsonl`, uses `models/SmolLM2-135M`, and writes to
`runs/smollm2_135m_moshi_full_duplex`.

The documented commands use local paths under `models/`. Both
`models/LFM3-1.7B` and `models/SmolLM2-135M` are recognized aliases and are
downloaded automatically when absent. SmolLM2 does not need
`--trust-remote-code`; LFM3 does. Keep their checkpoints in separate output
folders.

DDP uses one model and frozen Mimi per GPU. Do not pass `--mimi-device` under
torchrun. `--batch-size` is per rank and `--steps` counts microbatches.

## Metrics

- `assistant_np_accuracy`: accuracy excluding PAD; use it for lexical quality.
- `assistant_all_accuracy`: accuracy including PAD; expect it to be higher.
- `audio_acc`: mean accuracy across four Mimi codebooks during validation.
- `loss`: assistant CE plus weighted audio CE.

`--text-padding-weight 0.5` weights each PAD frame at 0.5 while lexical and
EPAD targets remain at 1.0. Non-PAD targets therefore contribute twice as
much per frame. This affects text loss only and does not redefine accuracy.

The default objective weights are `--text-loss-weight 1.0` and
`--audio-loss-weight 1.0`. PAD/non-PAD balancing is handled separately by
`--text-padding-weight`.

Deterministic train/validation indices are saved to `OUTPUT/split.json`.
TensorBoard events are saved to `OUTPUT/tensorboard`.

## Checkpoints

Rank 0 writes `checkpoint_XXXXXXXX.pt` and `config.json`. Checkpoints contain
model and optimizer states. Inference memory-maps the package and loads the
model state into the instantiated network.
