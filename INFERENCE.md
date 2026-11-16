# Moshi Inference Guide

Return to [README](README.md). For training, see [TRAINING.md](TRAINING.md).

Inference encodes the selected user waveform with frozen Mimi, force-feeds
the observed user codebooks, and autoregressively generates assistant text
and assistant audio. Generated assistant audio is fed back through its causal
history. No user transcript or annotation sidecar is required.

## Run inference

Activate the project environment and run from the repository root:

```bash
PYTHONPATH=. python infer_full_duplex.py \
  --checkpoint runs/mono_post_training/checkpoint_00001000.pt \
  --audio /path/to/user.wav --user-channel 0 \
  --device cuda --mimi-device cuda:0 \
  --tail-seconds 10 \
  --text-temperature 0.7 --text-top-k 25 \
  --audio-temperature 0.8 --audio-top-k 250 \
  --output inference_outputs/response.json \
  --audio-output inference_outputs/response.wav
```

Replace the checkpoint path with one produced by training. The adjacent
`config.json` is used by default; override with `--config`.
Use `--model` to override a stored backbone path when moving checkpoints.
Add `--greedy` to use argmax for text and audio instead of sampling.
Temperature and top-k no longer affect greedy decoding.

## Input channels and response duration

For stereo dialogue, the default user channel is 1. For a mono user recording,
pass `--user-channel 0`. If mono is supplied with channel 1, the CLI warns and
falls back to channel 0.

`--tail-seconds` appends silence before Mimi encoding, extending the response
timeline. Generation runs for the input-plus-tail duration. Text EOS latches
text to PAD while audio continues on that fixed timeline.
Increase the tail if the response is truncated.

## Checkpoint compatibility

The stream layout is defined in the [training guide](TRAINING.md#streams).
Generation follows that delayed timeline and undelays assistant codebooks
before Mimi decoding.

The loader requires `text_context_mode: "moshi"` and the corresponding delay
pattern. With four codebooks this is `[0,0,1,1,1,0,1,1,1]`, not the old
five-stream text-only layout. Old text-only checkpoints lack assistant audio
embeddings/heads and cannot be loaded directly into this architecture.

## Outputs

The JSON contains:

- `assistant_text` and `transcript.assistant_text`: readable response.
- `channels.assistant_text.events`: generated PAD/EPAD/lexical events with timing.
- `turns`: the assistant entry spanning the first and last lexical events.
- `stream_order`, `delay_pattern`, decoding settings and checkpoint step.

The WAV has two time-aligned channels:

| Channel | Content |
|---|---|
| 0 | Generated assistant speech |
| 1 | Selected user waveform, including silence tail |

If `--audio-output` is omitted, the JSON output path with a `.wav` extension
is used. Output directories are created automatically.

## Runtime limitations

Reference decoding recomputes the Temporal prefix per frame and the Depth
prefix per codebook. Streaming KV-cache decoding is not implemented, so long
inputs can be slow.

For LFM3 environments where Triton's assembler does not recognize the GPU,
`--disable-lfm3-triton` selects its PyTorch causal-convolution fallback.
