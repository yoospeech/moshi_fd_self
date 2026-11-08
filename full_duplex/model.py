"""Audio-conditioned Hugging Face temporal model for full-duplex response generation."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


_MOSHI_SOURCE = Path(__file__).resolve().parent.parent / "thirdparty" / "moshi" / "moshi"
if str(_MOSHI_SOURCE) not in sys.path:
    sys.path.insert(0, str(_MOSHI_SOURCE))

from moshi.models.lm import LMModel  # noqa: E402
from moshi.models.lm_utils import _delay_sequence, _undelay_sequence  # noqa: E402


@dataclass
class FullDuplexConfig:
    temporal_model: str = "models/SmolLM2-135M"
    trust_remote_code: bool = False
    text_card: int = 49154
    audio_card: int = 2048
    num_codebooks: int = 4
    context: int = 3000
    frame_rate: float = 12.5
    text_delay_seconds: float = 0.08
    assistant_text_delay_steps: int = 0
    text_context_mode: str = "moshi"
    acoustic_delay_steps: int = 1
    depformer_dim: int = 1024
    depformer_heads: int = 16
    depformer_layers: int = 6
    depformer_ffn_dim: int = 4096
    audio_loss_weight: float = 0.1
    text_loss_weight: float = 1.0
    semantic_loss_weight: float = 100.0
    text_padding_weight: float = 0.5
    text_padding_id: int = 49152
    text_epad_id: int = 49153

    @property
    def text_delay_steps(self) -> int:
        return int(round(self.text_delay_seconds * self.frame_rate))

    @property
    def delays(self) -> list[int]:
        # Moshi ordering: assistant text, assistant audio, then user audio.
        # Both Mimi streams use one semantic codebook followed by delayed
        # acoustic codebooks.
        return [self.assistant_text_delay_steps] + self.audio_delays * 2

    @property
    def audio_delays(self) -> list[int]:
        return [0] + [self.acoustic_delay_steps] * (self.num_codebooks - 1)


class MoshiFullDuplexModel(LMModel):
    """Moshi-style assistant text conditioned on force-fed user Mimi audio."""

    @property
    def num_text_streams(self) -> int:
        return 1

    @property
    def num_codebooks(self) -> int:
        return self.num_text_streams + self.n_q

    @property
    def audio_offset(self) -> int:
        return self.num_text_streams

    @property
    def user_audio_offset(self) -> int:
        return self.audio_offset + self.full_duplex_config.num_codebooks

    def __init__(
        self,
        config: FullDuplexConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if config.text_context_mode != "moshi":
            raise ValueError("This model requires Moshi-style text context")
        if config.assistant_text_delay_steps != 0:
            raise ValueError(
                "assistant_text_delay_steps must be 0 for assistant inner-monologue"
            )
        temporal_config = AutoConfig.from_pretrained(
            config.temporal_model,
            trust_remote_code=config.trust_remote_code,
        )
        dim = int(temporal_config.hidden_size)
        self.full_duplex_config = config

        # Keep Moshi's native Temporal -> Depth Transformer training path.
        super().__init__(
            delays=config.delays,
            n_q=config.num_codebooks * 2,
            dep_q=config.num_codebooks,
            card=config.audio_card,
            text_card=config.text_card,
            dim=dim,
            num_heads=int(temporal_config.num_attention_heads),
            num_layers=1,
            hidden_scale=4,
            norm="rms_norm_f32",
            causal=True,
            context=config.context,
            existing_text_padding_id=config.text_padding_id,
            depformer_dim=config.depformer_dim,
            depformer_dim_feedforward=config.depformer_ffn_dim,
            depformer_num_heads=config.depformer_heads,
            depformer_num_layers=config.depformer_layers,
            depformer_context=config.num_codebooks,
            depformer_gating="silu",
            depformer_pos_emb="none",
            depformer_multi_linear=True,
            depformer_weights_per_step=True,
            device=device,
            dtype=dtype,
        )
        self.transformer = AutoModel.from_pretrained(
            config.temporal_model,
            torch_dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        ).to(device=device)
        self.out_norm = nn.Identity()
        self._seed_text_embeddings_from_temporal()

    def _get_initial_token(self) -> torch.Tensor:
        device = self.device
        text = torch.full(
            (1, self.num_text_streams, 1),
            self.text_initial_token_id,
            device=device,
            dtype=torch.long,
        )
        audio = torch.full(
            (1, self.num_audio_codebooks, 1),
            self.initial_token_id,
            device=device,
            dtype=torch.long,
        )
        return torch.cat([text, audio], dim=1)

    def _seed_text_embeddings_from_temporal(self) -> None:
        """Copy pretrained text embeddings while retaining DSM special rows."""
        source = self.transformer.get_input_embeddings().weight
        count = min(source.shape[0], self.text_emb.weight.shape[0])
        with torch.no_grad():
            self.text_emb.weight[:count].copy_(source[:count].to(self.text_emb.weight))

    def forward_text(
        self,
        sequence: torch.Tensor,
        sum_condition: torch.Tensor | None = None,
        cross_attention_src: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cross_attention_src is not None:
            raise ValueError("Hugging Face temporal mode does not use Moshi cross-attention conditions")

        _, codebooks, _ = sequence.shape
        if codebooks != self.num_codebooks:
            raise ValueError(f"expected {self.num_codebooks} streams, got {codebooks}")

        text_hidden = self.text_emb(sequence[:, 0])
        audio_hidden = torch.zeros_like(text_hidden)
        for index, embedding in enumerate(self.emb):
            audio_hidden = audio_hidden + embedding(sequence[:, index + self.audio_offset])
        if sum_condition is not None:
            audio_hidden = audio_hidden + sum_condition.to(audio_hidden)

        hidden = text_hidden + audio_hidden
        audio_mask = (sequence[:, self.audio_offset:] != self.zero_token_id).any(dim=1)
        attention_mask = audio_mask | (sequence[:, 0] != self.zero_token_id)
        output = self.transformer(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        temporal_hidden = output.last_hidden_state
        text_logits = self.text_linear(temporal_hidden)
        return temporal_hidden, text_logits.unsqueeze(1)

    def forward_assistant_audio_training(
        self, sequence: torch.Tensor, temporal_state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict assistant codebooks given past Temporal state and depth prefix."""
        assert self.depformer_text_emb is not None
        assert self.depformer_emb is not None
        assert self.depformer is not None
        batch, _, frames = sequence.shape
        inputs = []
        for codebook in range(self.dep_q):
            # moshi_tts uses one Temporal projection per Depth step.
            temporal = self.depformer_in[codebook](temporal_state)
            if codebook == 0:
                token = self.depformer_text_emb(sequence[:, 0])
            else:
                token = self.depformer_emb[codebook - 1](
                    sequence[:, self.audio_offset + codebook - 1]
                )
            inputs.append(token + temporal)
        depth_input = torch.stack(inputs, dim=2).view(batch * frames, self.dep_q, -1)
        depth_output = self.depformer(depth_input)
        logits = [
            self.linears[q](self.depformer_norms[q](depth_output[:, q])).view(batch, frames, -1)
            for q in range(self.dep_q)
        ]
        return torch.stack(logits, dim=1)

    def checkpoint_metadata(self) -> dict[str, object]:
        return asdict(self.full_duplex_config)

    def forward_training(
        self, streams: torch.Tensor, *, predict_audio: bool = True
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        """Predict assistant text and optionally its separate Mimi targets."""
        batch, codebooks, _ = streams.shape
        if codebooks != self.num_codebooks:
            raise ValueError(f"expected {self.num_codebooks} streams, got {codebooks}")
        initial = self._get_initial_token().expand(batch, -1, -1)
        # Construct Moshi's V_s interleaving for training: text and semantic
        # streams stay at s, while assistant/user acoustic streams move to
        # s + acoustic_delay. Inference generates this same delayed timeline.
        delayed = _delay_sequence(self.delays, streams, initial)
        delayed = torch.cat([initial, delayed], dim=2)
        temporal, text_logits = self.forward_text(delayed[:, :, :-1])
        text_logits, text_mask = _undelay_sequence(
            self.delays[: self.num_text_streams], text_logits, fill_value=float("nan")
        )
        text_mask &= streams[:, : self.num_text_streams] != self.zero_token_id
        if not predict_audio:
            return text_logits, text_mask, None, None
        audio_logits = self.forward_assistant_audio_training(
            delayed[:, :, 1:], temporal
        )
        audio_logits, audio_mask = _undelay_sequence(
            self.full_duplex_config.audio_delays,
            audio_logits, fill_value=float("nan")
        )
        audio_mask &= streams[
            :, self.audio_offset : self.user_audio_offset
        ] != self.zero_token_id
        return text_logits, text_mask, audio_logits, audio_mask


def masked_multistream_loss(
    text_logits: torch.Tensor,
    text_mask: torch.Tensor,
    streams: torch.Tensor,
    *,
    text_padding_id: int | None = None,
    text_padding_weight: float = 0.5,
    audio_logits: torch.Tensor | None = None,
    audio_mask: torch.Tensor | None = None,
    audio_loss_weight: float = 1.0,
    text_loss_weight: float = 1.0,
    semantic_loss_weight: float = 100.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute assistant text and feedback-audio CE losses."""
    text_targets = streams[:, :1]
    channel_losses = []
    metrics: dict[str, torch.Tensor] = {}
    for channel, name in enumerate(("assistant",)):
        mask = text_mask[:, channel]
        logits = text_logits[:, channel][mask].float()
        targets = text_targets[:, channel][mask]
        losses = nn.functional.cross_entropy(logits, targets, reduction="none")
        if text_padding_id is not None:
            weights = torch.where(
                targets == text_padding_id,
                losses.new_tensor(text_padding_weight),
                losses.new_tensor(1.0),
            )
            channel_loss = (losses * weights).sum() / weights.sum()
        else:
            channel_loss = losses.mean()
        predictions = logits.argmax(dim=-1)
        non_pad = (
            targets != text_padding_id
            if text_padding_id is not None
            else torch.ones_like(targets, dtype=torch.bool)
        )
        non_pad_accuracy = (
            (predictions[non_pad] == targets[non_pad]).float().mean()
            if non_pad.any()
            else logits.new_zeros(())
        )
        channel_losses.append(channel_loss)
        metrics[f"loss/{name}"] = channel_loss.detach()
        metrics[f"{name}/accuracy"] = (predictions == targets).float().mean().detach()
        metrics[f"{name}/non_pad_accuracy"] = non_pad_accuracy.detach()

    text_loss = torch.stack(channel_losses).mean()
    total = text_loss_weight * text_loss
    if audio_logits is not None and audio_mask is not None:
        audio_targets = streams[:, 1 : 1 + audio_logits.shape[1]]
        per_codebook = []
        accuracies = []
        for codebook in range(audio_logits.shape[1]):
            mask = audio_mask[:, codebook]
            logits = audio_logits[:, codebook][mask].float()
            targets = audio_targets[:, codebook][mask]
            codebook_loss = (
                nn.functional.cross_entropy(logits, targets)
                if targets.numel() else logits.sum() * 0.0
            )
            per_codebook.append(codebook_loss)
            accuracy = (
                (logits.argmax(dim=-1) == targets).float().mean()
                if targets.numel() else logits.new_zeros(())
            )
            accuracies.append(accuracy)
            metrics[f"assistant_audio/rq/{codebook}_loss"] = codebook_loss.detach()
            metrics[f"assistant_audio/rq/{codebook}_accuracy"] = accuracy.detach()
        weights = audio_logits.new_ones(len(per_codebook), dtype=torch.float32)
        weights[0] = semantic_loss_weight
        audio_loss = (torch.stack(per_codebook) * weights).sum() / weights.sum()
        audio_accuracy = torch.stack(accuracies).mean()
        total = text_loss_weight * text_loss + audio_loss_weight * audio_loss
        metrics["loss/assistant_audio"] = audio_loss.detach()
        metrics["assistant_audio/accuracy"] = audio_accuracy.detach()
    metrics["loss/total"] = total.detach()
    return total, metrics
