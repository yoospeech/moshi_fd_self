# Moshi Full-Duplex Fine-Tuning

> Experimental research prototype.

This project takes the user audio channel as a prompt and jointly predicts the
assistant response text and assistant audio. User audio is conditioning
only and is never an audio loss target.

```text
stereo WAV channel 1 --> frozen Mimi --> user audio conditioning stream
stereo WAV channel 0 --> frozen Mimi --> assistant audio target stream
assistant PAD/EPAD timeline ------------> Temporal Transformer
                                             |-- assistant text head
                                             `-- assistant audio Depth Transformer
```

## Streams and timing

```text
streams: [assistant_text, assistant_audio(q0..q3), user_audio(q0..q3)]
delays:  [0,              0,1,1,1,          0,1,1,1]
```

Assistant text and both semantic audio codebooks are placed at the current
12.5 Hz frame. The three acoustic codebooks in each audio stream lag by one
frame (80 ms), following Moshi's semantic/acoustic interleaving. Assistant
audio is teacher-forced only through its causal past; its current frame is the
Depth prediction target. User audio is prompt conditioning and has no loss.

Assistant word timestamps are converted to the original Moshi PAD/EPAD
timeline. EPAD is placed immediately before a word unless that would overwrite
a lexical token. No speaker BEGIN/END tokens are used.

Training treats user Mimi tokens as an acoustic prompt and optimizes assistant
text cross-entropy plus assistant Mimi codebook cross-entropy. User Mimi tokens
are never audio targets. Offline inference force-feeds user Mimi tokens,
autoregressively samples assistant text and audio, feeds generated assistant
audio back into later Temporal frames, and decodes it to a WAV file.

Each training batch mixes acoustic-conditioned and text-only examples. With
the documented `--batch-size 4 --text-only-fraction 0.5`, every rank receives
two examples with the user-audio prompt and two examples with the acoustic
streams removed. Their positions are sampled randomly in every batch; both
groups retain the assistant text target. The text-only half is replay from the
paired assistant-text dataset, not a separate external text corpus, and helps
keep the pretrained text model active while adapting it to acoustic prompts.

## Repository layout

```text
full_duplex/model.py       model, interleaving, Depth Transformer, losses
full_duplex/dataset.py     stereo user prompt, assistant audio, and PAD/EPAD target
full_duplex/train.py       single-GPU/DDP training and validation
full_duplex/inference.py   force-fed user-audio inference
train_full_duplex.py       training CLI
infer_full_duplex.py       inference CLI
tests/test_full_duplex.py      offline regression tests
```

## Setup

```bash
git clone --recurse-submodules https://github.com/yoospeech/moshi_fd_self.git
cd moshi_fd_self
pip install -r requirements.txt
pip install --no-deps -e thirdparty/moshi/moshi
```

The project uses Transformers 5 tokenizer metadata. Moshi is installed with
`--no-deps` because its upstream package metadata still caps
`huggingface-hub` and `safetensors` below the versions used here; the Moshi
APIs exercised by this project are covered by the smoke and regression tests.

Model paths are standardized under the project `models/` directory:

```text
models/SmolLM2-135M
models/LFM3-1.7B
```

If either known path is absent, rank 0 downloads the corresponding official
Hugging Face repository into that path and the other DDP ranks wait. LFM3
remote model code is trusted automatically; SmolLM2 does not require it. An
arbitrary Hugging Face repository ID or existing local directory is also
accepted. If `--mimi-checkpoint` is omitted, Mimi is resolved from Kyutai.

## Data

Use the original manifest:

```text
/data/db/daily-talk-contiguous/dailytalk.jsonl
```

For a row whose audio path is `data_stereo/406.wav`, user audio is read from
WAV channel 1 by default and assistant response timestamps are read from
`data_stereo/406.json`. `user_hyp`, `user_406.json`, and Whisper transcripts
are not used. Relative audio paths are resolved from the manifest directory.

Default filtering keeps samples with duration 5–200 seconds, at least 5
assistant words, 0.5–2.5 assistant words/second, valid timestamps, 5–80%
assistant timestamp coverage, and no lexical BPE collision at 12.5 Hz.

## Smoke test

This checks model construction, Temporal and Depth forward passes, losses,
backward, and optimizer updates without loading a dataset or Mimi:

```bash
cd /home/speech/Documents/workspace/moshi_fd_self

PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
train_full_duplex.py \
  --model models/SmolLM2-135M \
  --model-download-root models \
  --smoke-test \
  --smoke-steps 1 \
  --acoustic-delay 1 \
  --batch-size 1 \
  --text-only-fraction 0 \
  --output /tmp/moshi_full_duplex_smoke
```

## Current machine: single-GPU LFM3-1.7B training

The following command uses the local Conda environment, downloaded LFM3 model,
and DailyTalk manifest on this machine. Run it directly with one Python process;
do not set `--nproc_per_node=2` when only one GPU is visible.

```bash
cd /home/ysw/Documents/eee/moshi_fd_self

CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=4 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=thirdparty/moshi/moshi \
/home/ysw/Documents/anaconda3/envs/moshi_finetune/bin/python \
train_full_duplex.py \
  --manifest /home/ysw/Documents/eee/moshi-finetune/daily-talk-contiguous/dailytalk.jsonl \
  --model models/LFM3-1.7B \
  --model-download-root models \
  --trust-remote-code \
  --num-codebooks 4 \
  --assistant-text-delay-steps 0 \
  --acoustic-delay 1 \
  --depformer-dim 1024 \
  --depformer-heads 16 \
  --depformer-layers 6 \
  --depformer-ffn-dim 4096 \
  --audio-loss-weight 0.1 \
  --text-loss-weight 1.0 \
  --semantic-loss-weight 100 \
  --text-padding-weight 0.5 \
  --batch-size 4 \
  --text-only-fraction 0.5 \
  --grad-accum 1 \
  --steps 100000 \
  --learning-rate 2e-4 \
  --temporal-learning-rate 2e-5 \
  --min-duration 5 \
  --max-duration 100 \
  --min-words 5 \
  --min-words-per-second 0.5 \
  --max-words-per-second 2.5 \
  --valid-ratio 0.05 \
  --valid-every 500 \
  --valid-batches 54 \
  --log-every 10 \
  --save-every 500 \
  --print-model \
  --output runs/LFM3-1.7B_mixed_full_duplex
```

This gives an effective batch of `4 × accumulation 1 = 4`, split evenly
between acoustic and text-only samples. If memory is insufficient, use
`--batch-size 2 --grad-accum 2`; the effective batch remains 4.

## Two-GPU DDP training

```bash
cd /home/speech/Documents/workspace/moshi_fd_self

CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
-m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  train_full_duplex.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.jsonl \
  --model models/SmolLM2-135M \
  --model-download-root models \
  --num-codebooks 4 \
  --assistant-text-delay-steps 0 \
  --acoustic-delay 1 \
  --depformer-dim 1024 \
  --depformer-heads 16 \
  --depformer-layers 6 \
  --depformer-ffn-dim 4096 \
  --audio-loss-weight 0.1 \
  --text-loss-weight 1.0 \
  --semantic-loss-weight 100 \
  --text-padding-weight 0.5 \
  --batch-size 4 \
  --text-only-fraction 0.5 \
  --grad-accum 2 \
  --steps 100000 \
  --learning-rate 2e-4 \
  --temporal-learning-rate 2e-5 \
  --min-duration 5 \
  --max-duration 200 \
  --min-words 5 \
  --min-words-per-second 0.5 \
  --max-words-per-second 2.5 \
  --valid-ratio 0.05 \
  --valid-every 500 \
  --valid-batches 20 \
  --log-every 10 \
  --save-every 500 \
  --print-model \
  --output runs/smollm2_135m_moshi_full_duplex
```

### LFM3-1.7B alternative

LFM3-1.7B uses the same architecture, streams, targets, losses, and inference
code. To run the larger model, change these arguments:

```bash
  --model models/LFM3-1.7B \
  --model-download-root models \
  --trust-remote-code \
  --output runs/lfm3_1_7b_moshi_full_duplex
```

SmolLM2-135M is the documented default for pipeline validation and small-data
overfitting. LFM3-1.7B is the higher-capacity alternative.

Do not pass `--mimi-device` in DDP; every rank places frozen Mimi and its model
on the local GPU. Batch size is per rank, so the effective batch above is
`2 GPUs × batch 4 × accumulation 2 = 16`: 8 acoustic-conditioned and 8
text-only examples per optimizer update. `--steps` counts microbatches. Only
rank 0 validates and writes checkpoints.

`--text-only-fraction` is applied independently inside every rank's training
batch. A value of `0.5` reserves half the samples for text-only replay. At
least one acoustic-conditioned sample is retained, including in a short final
batch. Validation remains fully acoustic-conditioned.

For one GPU, run the script directly without `torch.distributed.run`. A separate
Mimi device can then be selected with `--mimi-device cuda:1`.

### CLI metrics

```text
loss                    total optimized loss
assistant_text          assistant response cross-entropy
assistant_audio         assistant Mimi codebook loss (only when audio loss weight > 0)
assistant_np_accuracy   assistant lexical/EPAD accuracy excluding PAD
assistant_all_accuracy  assistant accuracy including PAD frames
audio_acc               mean assistant Mimi codebook accuracy (only when audio loss weight > 0)
```

PAD occupies many frames, so `assistant_all_accuracy` is usually higher. Use
`assistant_np_accuracy` to judge whether actual assistant tokens improve.
`--text-padding-weight 0.5` gives each non-PAD target twice the loss weight
of each PAD target; it changes optimization, not the reported accuracies.
The documented default objective is joint assistant text and audio:

```text
loss = 1.0 * assistant_text + 0.1 * assistant_audio
```

The user audio stream still conditions the acoustic half of each batch; setting
`--audio-loss-weight 0` skips assistant audio prediction and its Depth forward
pass while retaining user audio as conditioning. `--semantic-loss-weight`
controls the first assistant codebook when audio loss is enabled.

```bash
tensorboard --logdir runs/smollm2_135m_moshi_full_duplex/tensorboard
```

## Inference

```bash
cd /home/speech/Documents/workspace/moshi_fd_self

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
infer_full_duplex.py \
  --checkpoint runs/smollm2_135m_moshi_full_duplex/checkpoint_00100000.pt \
  --audio /data/db/daily-talk-contiguous/data_stereo/0.wav \
  --user-channel 1 \
  --device cuda \
  --mimi-device cuda:0 \
  --tail-seconds 10 \
  --text-temperature 0.7 \
  --text-top-k 25 \
  --audio-temperature 0.8 \
  --audio-top-k 250 \
  --audio-output inference_outputs/0.wav \
  --output inference_outputs/0.json
```

Add `--greedy` for deterministic argmax decoding; temperature and top-k are
then ignored. For mono user audio use `--user-channel 0`. `--tail-seconds 10`
appends silence before Mimi encoding so the audio clock has time to emit a
response.

The command writes assistant response metadata to JSON and a time-aligned
stereo WAV: channel 0 is generated assistant speech, channel 1 is the selected
original user waveform with the requested silence tail. For mono recordings
use `--user-channel 0`. No transcript or timestamp sidecar is needed at inference.
Generation runs on the delayed timeline, then undelays codebooks before Mimi
decoding. Startup acoustic slots use INIT and final out-of-range slots are
absent, matching training. Text EOS latches text to PAD; audio runs through the
fixed input-plus-tail duration.

```json
{
  "stream_order": ["assistant_text", "assistant_audio_q0", "assistant_audio_q1", "assistant_audio_q2", "assistant_audio_q3", "user_audio_q0", "user_audio_q1", "user_audio_q2", "user_audio_q3"],
  "delay_pattern": [0, 0, 1, 1, 1, 0, 1, 1, 1],
  "assistant_text": "Assistant response",
  "transcript": {"assistant_text": "Assistant response"},
  "channels": {"assistant_text": {"text": "Assistant response", "events": []}}
}
```

## Compatibility and limitations

New checkpoints record `text_context_mode: "moshi"`. Older incompatible
checkpoints are rejected because their stream count, heads, and delay pattern
are incompatible.

Reference inference recomputes the Temporal prefix once per delayed frame and
the Depth prefix per codebook. KV-cache streaming is not implemented.
Old text-only checkpoints do not have the extra audio embeddings required by
this dual-audio architecture and cannot be loaded directly.

Run regression tests with:

```bash
/data/util/anaconda3/envs/duplexomni/bin/python \
  -m unittest discover -s tests -v
```

See [training details](FULL_DUPLEX_TRAINING.md) and
[inference details](FULL_DUPLEX_INFERENCE.md).
