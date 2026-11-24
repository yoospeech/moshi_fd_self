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
full_duplex/train.py       single-GPU/DDP training, validation and timing
full_duplex/inference.py   user-conditioned assistant text/audio generation
scripts/annotate.py        word alignment extraction
train_full_duplex.py       training CLI
infer_full_duplex.py       inference CLI
tests/                     regression tests
```
