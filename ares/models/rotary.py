from typing import Tuple, Optional

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    # Cast cos/sin to x's dtype. The rotary inv_freq buffer is registered in
    # fp32 (for precision when computing cos/sin), so cos/sin can be fp32
    # while x is bf16. Without this cast the multiply errors out under pure
    # bf16 inference. Under fp32 or autocast this is a no-op.
    cos = cos[:, :, : x.shape[-2], :].to(x.dtype)
    sin = sin[:, :, : x.shape[-2], :].to(x.dtype)

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        base_frequency: int = 10000,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale
        inv_freq = 1.0 / (
            base_frequency ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _compute_cos_sin(self, bsz, seq_length, device, position_ids=None):
        if position_ids is None:
            t = (
                torch.arange(
                    seq_length,
                    device=device,
                    dtype=self.inv_freq.dtype,
                )
                .unsqueeze(0)
                .expand(bsz, -1)
            )
        else:
            t = position_ids.to(dtype=self.inv_freq.dtype, device=device)

        t = t * self.scale
        freqs = torch.einsum("bi,j->bij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        cos_emb = emb.cos()[:, None, :, :]
        sin_emb = emb.sin()[:, None, :, :]
        return cos_emb, sin_emb

    def forward(
        self, q: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = q.shape[0]
        seq_length = q.shape[1]
        device = q.device
        cos_emb, sin_emb = self._compute_cos_sin(
            bsz,
            seq_length,
            device,
            position_ids=position_ids,
        )
        return cos_emb, sin_emb
