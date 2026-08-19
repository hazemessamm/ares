from __future__ import annotations
import torch
from transformers.models import convbert
from torch import nn


class ConvBert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ):
        """
        Base ConvBert encoder model.

        Args:
            input_dim: Dimension of the input embeddings.
            nhead: Integer specifying the number of heads for the `ConvBert`
                   model.
            hidden_dim: Integer specifying the hidden dimension for the
                        `ConvBert` model.
            num_layers: Integer specifying the number of layers for the
                        `ConvBert` model.
            kernel_size: Integer specifying the filter size for the
                         `ConvBert` model. Default: 7
        """
        super().__init__()

        config = convbert.ConvBertConfig(
            hidden_size=input_dim,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_dim,
            conv_kernel_size=kernel_size,
            num_hidden_layers=num_layers,
            hidden_dropout_prob=dropout,
        )
        self.encoder = convbert.ConvBertModel(config).encoder

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Copied from: https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/modeling_utils.py#L863 # noqa

        Args:
            attention_mask: Tensor of shape [batch_size, seq_len] containing ones in unmasked
                indices and zeros in masked indices.

        Returns:
            Tensor of extended attention mask that can be fed to the ConvBert model.
        """
        attn_mask = attention_mask[:, None, None, :]
        attn_mask = (1.0 - attn_mask) * torch.finfo(attention_mask.dtype).min
        return attn_mask

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.FloatTensor:
        attn_mask = self.get_extended_attention_mask(
            attention_mask.to(hidden_states.dtype)
        )
        hidden_states = self.encoder(hidden_states, attention_mask=attn_mask)[
            0
        ]
        return hidden_states
