# Moshi Training

> Experimental research prototype.

Train a full-duplex model to predict assistant text and speech from user audio.

- [Training guide](TRAINING.md): architecture, mono post-training, stereo fine-tuning,
  online alignment, losses and DDP.
- [Inference guide](INFERENCE.md): checkpoint loading, decoding, channels and outputs.

## Setup

Run commands from the repository root with the intended Python environment active.

```bash
git clone --recurse-submodules https://github.com/yoospeech/moshi_fd_self.git
cd moshi_fd_self
# CUDA 13.0: install matching PyTorch and TorchAudio builds together.
python -m pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install --no-deps -e thirdparty/moshi/moshi
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__, torch.version.cuda)"
```

LFM3 config loading has been checked with Python 3.12.14, transformers 5.22.1
and huggingface-hub 2.4.0. LFM3 requires newer Transformers APIs; the previous
Transformers 4.x setup is not sufficient. Install the vendored Moshi with
`--no-deps` as above because its legacy dependency metadata requires Hub below
1.0. Prefer an isolated environment for this project.

TorchAudio must match both the PyTorch release and its binary build. The
requirements pin both to 2.9.0; the CUDA command above selects their build source
([official installation commands](https://pytorch.org/get-started/previous-versions/)).
For an existing environment with `_torchaudio.so: undefined symbol`, reinstall
the pair from that same source rather than relying on already installed wheels:

```bash
python -m pip install --force-reinstall --no-cache-dir torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__, torch.version.cuda)"
```

Use a matching CPU or other CUDA index for other hardware. Vendor-specific
PyTorch builds need a matching vendor TorchAudio build (or a source build against
that PyTorch); matching version numbers alone does not guarantee binary compatibility.

## Repository layout

```text
full_duplex/model.py       model, streams, Depth Transformer and losses
full_duplex/dataset.py     shared mono/stereo post-training datasets
full_duplex/natural_monologue_dataset.py  VAD-driven text-leading monologues
full_duplex/train.py       single-GPU/DDP training, validation and timing
full_duplex/inference.py   user-conditioned assistant text/audio generation
scripts/annotate.py        word alignment extraction
train_full_duplex.py       training CLI
infer_full_duplex.py       inference CLI
tests/                     regression tests
```

## Natural Monologue sine conditioning

With `--enable-sine-conditioning`, user speech pauses assistant content.
Interrupted assistant text/audio frames are queued in order and resume after
the last delayed user audio frame. Empty source frames are not queued; unused frames in both
audio channels contain a sine wave. Pending content can extend the sample.
The feature is disabled by default.

The resulting raw frame layout remains
`assistant_text | assistant_Mimi | user_Mimi`. Moshi's existing delay pattern
is then applied to the two audio groups (`[1, 1, 1, 1]` per Mimi stream); the
control symbols live only in the assistant text stream and do not consume an
audio codebook slot.

Text stays at its source frame and audio is delayed one frame. Controls describe
the delayed audio: HEAR for user speech, SPEAK for assistant speech, WAIT for
neither. HEAR is retained through the final delayed user frame. Queued lexical
BPE resumes on the following frame and otherwise takes precedence over controls.
SPEAK below denotes the existing `<SPK>` tokenizer token. Example after delay:

```text
Frame:           [0    |1    |2     |3     |4     |5        |6       |7       |8       |9        |10      |11    |12   ]
User audio:      [sine |sine |sine  |"Wait"|"one" |"second" |sine    |sine    |sine    |sine     |sine    |sine  |sine ]
Assistant audio: [sine |sine |"I"   |sine  |sine  |sine     |sine    |"think" |"we"    |"should" |"leave" |"now" |sine ]
Text:            [WAIT |[I]  |SPEAK |HEAR  |HEAR  |HEAR     |[think] |[we]    |[should]|[leave]  |[now]   |SPEAK |WAIT ]
```

This is a symbolic frame example. Real utterances use contiguous BPE starting
at the annotated utterance onset, without word timestamps; individual spoken
words need not occupy one frame or line up with their BPE piece boundaries.
Initial delayed audio slots use Mimi-encoded sine IDs saved in the checkpoint;
the model's internal BOS remains separate from the displayed timeline.

Behavior-SD user backchannels nested under an assistant utterance are exempt
from pause/resume. Their real user audio overlaps the continuing assistant
audio, while the assistant BPE sequence continues without a HEAR token. With a
one-frame audio delay, a user backchannel at source frame 2 is audible at frame
3 on the user audio stream, alongside the assistant audio at frame 3.
Assistant backchannels nested under user speech are suppressed: their text is
not placed on the lexical timeline and their audio frames become sine. The text
target remains HEAR for the delayed user-audio frames, so overlapping assistant
backchannel annotations cannot collide in the single text stream.

Enable it with a VAD-driven natural manifest or a Behavior-SD folder containing
adjacent same-stem WAV/JSON pairs. Training registers
`<WAIT>`, `<SPK>`, and `<HEAR>` as tokenizer special tokens, saves that tokenizer
next to the checkpoint, and uses the resulting IDs consistently:

    PYTHONPATH=. /home/ysw/Documents/anaconda3/envs/moshi_finetune/bin/python \
      train_full_duplex.py \
      --task finetuning --natural-monologue \
      --folders '/path/to/behavior-sd/validation:en' \
      --enable-sine-conditioning \
      --sine-frequency 440 --sine-amplitude 0.5 \
      --model models/LFM3-1.7B --batch-size 4 --steps 1000 \
      --output runs/natural_sine

When enabled, a text timeline that exceeds its audio timeline is treated as
invalid and stops loading rather than truncating conditioning. The default
disabled mode preserves the previous Natural Monologue behavior.

For one Behavior-SD JSON/WAV pair, generate a stereo WAV (user, conditioned
assistant), an aligned eight-panel PNG, and JSON sine statistics with:

    PYTHONPATH=. /home/ysw/Documents/anaconda3/envs/moshi_finetune/bin/python \
      scripts/debug_natural_monologue_sine.py \
      --metadata /path/to/behavior-sd/validation/0001/0000003623.json \
      --output-prefix /tmp/natural_sine_debug

Pass the exact training Mimi file through --mimi-checkpoint to also report
changed Mimi token/frame counts. No PersonaPlex SINE_TOKENS are hardcoded.
The sine is a training-only acoustic conditioning target, not user-facing
assistant speech. Muting or suppressing it during inference is separate work.
