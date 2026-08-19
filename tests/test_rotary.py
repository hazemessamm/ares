import torch
import pytest
from ares.models.rotary import (
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)


class TestRotateHalf:
    def test_output_shape(self):
        x = torch.randn(2, 4, 8, 16)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_values(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(out, expected)

    def test_double_rotate_negates(self):
        x = torch.randn(2, 4, 8, 16)
        out = rotate_half(rotate_half(x))
        assert torch.allclose(out, -x)


class TestRotaryEmbedding:
    """RotaryEmbedding computes (cos, sin) from a [B, L, D] tensor.

    Rotation is applied separately by apply_rotary_pos_emb to a
    [B, H, L, head_dim] tensor.
    """

    @pytest.fixture
    def rope(self):
        return RotaryEmbedding(dim=16)

    def test_output_shapes(self, rope):
        x = torch.randn(2, 8, 16)
        cos, sin = rope(x)
        assert cos.shape == (2, 1, 8, 16)
        assert sin.shape == (2, 1, 8, 16)

        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        assert apply_rotary_pos_emb(q, cos, sin).shape == q.shape
        assert apply_rotary_pos_emb(k, cos, sin).shape == k.shape

    def test_preserves_norm(self, rope):
        x = torch.randn(2, 8, 16)
        cos, sin = rope(x)
        q = torch.randn(2, 4, 8, 16)
        q_rot = apply_rotary_pos_emb(q, cos, sin)
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)

    def test_with_position_ids(self, rope):
        x = torch.randn(2, 8, 16)
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
        cos, sin = rope(x, position_ids=position_ids)
        assert cos.shape == (2, 1, 8, 16)

        q = torch.randn(2, 4, 8, 16)
        assert apply_rotary_pos_emb(q, cos, sin).shape == q.shape

    def test_position_ids_match_default_ordering(self, rope):
        """Explicit sequential position_ids reproduce the implicit arange."""
        x = torch.randn(2, 8, 16)
        cos_default, sin_default = rope(x)
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
        cos_explicit, sin_explicit = rope(x, position_ids=position_ids)
        assert torch.allclose(cos_default, cos_explicit, atol=1e-6)
        assert torch.allclose(sin_default, sin_explicit, atol=1e-6)

    def test_custom_base_frequency(self):
        x = torch.randn(1, 4, 16)
        cos_a, _ = RotaryEmbedding(dim=16, base_frequency=10000)(x)
        cos_b, _ = RotaryEmbedding(dim=16, base_frequency=50000)(x)
        assert cos_a.shape == (1, 1, 4, 16)
        assert not torch.allclose(
            cos_a, cos_b
        ), "Different base frequencies should produce different embeddings"

    def test_scale_parameter(self):
        x = torch.randn(1, 4, 16)
        cos_default, _ = RotaryEmbedding(dim=16, scale=1.0)(x)
        cos_scaled, _ = RotaryEmbedding(dim=16, scale=2.0)(x)
        assert not torch.allclose(
            cos_default, cos_scaled
        ), "Different scales should produce different outputs"

    def test_different_seq_lengths(self, rope):
        for seq_len in (4, 8, 16):
            x = torch.randn(2, seq_len, 16)
            cos, sin = rope(x)
            assert cos.shape == (2, 1, seq_len, 16)
            q = torch.randn(2, 4, seq_len, 16)
            assert apply_rotary_pos_emb(q, cos, sin).shape == q.shape

    def test_inv_freq_buffer(self, rope):
        assert hasattr(rope, "inv_freq")
        assert rope.inv_freq.shape == (8,)
        assert not rope.inv_freq.requires_grad
