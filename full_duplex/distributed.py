"""Process-local device selection and the DDP training entry point."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from .model import masked_multistream_loss


def setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise ValueError("LOCAL_RANK exceeds the visible GPU count")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            timeout=timedelta(minutes=60),
        )
    return rank, world_size, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


class TrainingObjective(nn.Module):
    """DDP must invoke forward, not a method on the unwrapped model."""

    def __init__(
        self, model, padding_id, padding_weight, text_loss_weight,
        audio_loss_weight, semantic_loss_weight
    ):
        super().__init__()
        self.model = model
        self.padding_id = padding_id
        self.padding_weight = padding_weight
        self.text_loss_weight = text_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.semantic_loss_weight = semantic_loss_weight

    def forward(self, streams):
        text_logits, text_mask, audio_logits, audio_mask = self.model.forward_training(
            streams, predict_audio=self.audio_loss_weight > 0
        )
        return masked_multistream_loss(
            text_logits, text_mask, streams,
            text_padding_id=self.padding_id,
            text_padding_weight=self.padding_weight,
            text_loss_weight=self.text_loss_weight,
            audio_logits=audio_logits,
            audio_mask=audio_mask,
            audio_loss_weight=self.audio_loss_weight,
            semantic_loss_weight=self.semantic_loss_weight,
        )
