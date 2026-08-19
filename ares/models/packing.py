"""Attention-bias construction for packed sequences.

Kept free of data-pipeline dependencies so the model can be imported
with only the core requirements installed.
"""

import torch


def build_packing_attention_bias(sequence_ids, dtype, device):
    """
    Build [B, 1, L, L] additive attention bias from sequence_ids [B, L].

    Call this ONCE in the parent model's forward, then pass to all layers.
    Padding positions share the last sequence's ID, so cross-sequence
    blocking alone is sufficient. Padding-key blocking is handled by
    prepare_attention_mask additively.

    Args:
        sequence_ids: [B, L] integer tensor (no -1; padding shares last seq ID)
        dtype: runtime attention dtype (e.g., torch.bfloat16)
        device: runtime attention device

    Returns:
        attn_bias: [B, 1, L, L] with 0 where attention is allowed,
                   -inf where blocked
    """
    if sequence_ids.dim() == 1:
        sequence_ids = sequence_ids.unsqueeze(0)

    same_seq = sequence_ids.unsqueeze(-1) == sequence_ids.unsqueeze(-2)
    attn_bias = torch.zeros(
        *sequence_ids.shape,
        sequence_ids.size(1),
        dtype=dtype,
        device=device,
    )
    attn_bias.masked_fill_(~same_seq, float("-inf"))

    return attn_bias.unsqueeze(1)
