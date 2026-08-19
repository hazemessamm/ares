from __future__ import annotations
from ares.models import layers
from ares.models.encoder import EncoderLayer
from ares.models.config import AresConfig
from transformers import PreTrainedModel
import torch
from typing import Optional
from ares.models.utils import soft_capping

from transformers.modeling_outputs import MaskedLMOutput
from torch import nn
import torch.nn.functional as F
from ares.models.layers import get_norm
from ares.models.packing import build_packing_attention_bias
from ares.models.rotary import RotaryEmbedding


def mlm_loss(
    logits,
    targets,
    ignore_index=-100,
):
    V = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def prepare_attention_mask(mask, dtype, device):
    """
    Accepts either:
    - 2D mask [B, L] legacy padding mask
    - 3D mask [B, L, L] precomputed block mask

    Returns additive bias [B, 1, *, L] for attention.
    """
    if mask is None:
        return None

    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]

    attn_bias = torch.zeros_like(mask, dtype=dtype, device=device)
    attn_bias.masked_fill_(mask == 0, float("-inf"))
    return attn_bias


class Ares(PreTrainedModel):
    config_class = AresConfig
    # Newer transformers expects this mapping to
    # exist during from_pretrained().
    # Ares does not tie any weights, so the mapping is empty.
    all_tied_weights_keys = {}

    def __init__(self, config: AresConfig):
        super().__init__(config)

        self.sequence_embeddings = layers.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=config.pad_token_id,
        )

        self.rotary_embedding = RotaryEmbedding(
            dim=config.embed_dim // config.num_heads,
            base_frequency=config.rope_frequency,
            scale=config.rope_scale,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = get_norm(
            config.norm_type,
            config.embed_dim,
        )

        self.logits_capping_value = config.logits_capping_value

        self.output_layer = nn.Linear(
            config.embed_dim,
            config.vocab_size,
            bias=config.bias,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(
            self.output_layer.weight, mean=0, std=self.config.embed_dim**-0.5
        )
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        sequence_ids: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
    ) -> MaskedLMOutput:
        embeddings = self.sequence_embeddings(input_ids)
        attention_bias = prepare_attention_mask(
            attention_mask,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )

        if sequence_ids is not None:
            packing_attention_bias = build_packing_attention_bias(
                sequence_ids,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            if attention_bias is None:
                attention_bias = packing_attention_bias
            else:
                attention_bias = attention_bias + packing_attention_bias

        cos_sin_emb = self.rotary_embedding(
            embeddings,
            position_ids=position_ids,
        )

        for layer in self.layers:
            embeddings = layer(
                embeddings,
                padding_mask=attention_mask,
                attention_bias=attention_bias,
                position_ids=position_ids,
                cos_sin_emb=cos_sin_emb,
            )

        embeddings = self.norm(embeddings)
        logits = self.output_layer(embeddings)

        if (
            self.logits_capping_value is not None
            and self.logits_capping_value > 0.0
        ):  # noqa
            logits = soft_capping(
                logits,
                capping_value=self.logits_capping_value,
            )

        loss = None
        if labels is not None:
            loss = mlm_loss(
                logits,
                labels,
                ignore_index=-100,
            )

        if return_dict:
            return MaskedLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=(embeddings,),
            )
        else:
            return (loss, logits)
