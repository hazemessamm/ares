import torch
import pytest
from ares.models.config import AresConfig
from ares.models.model import Ares
from ares.models.encoder import EncoderLayer
from ares.models.expert_choice_router import ExpertChoiceRouting
from ares.models.soft_router import SoftRouter
from ares.models.attention import MultiHeadAttention
from ares.models.model import prepare_attention_mask
from ares.pipelines.debugging import NaNObserver
from ares.models.layers import FeedForward, get_norm, Embedding, ScaleNorm
from ares.models.rotary import RotaryEmbedding


def rotary_for(x, num_heads):
    """Build the (cos, sin) pair MultiHeadAttention/EncoderLayer require.

    ``x`` is [B, L, embed_dim]; rotary operates on per-head slices.
    """
    rope = RotaryEmbedding(dim=x.shape[-1] // num_heads)
    return rope(x)


class TestPreparePaddingMask:
    def test_none_returns_none(self):
        assert prepare_attention_mask(None, torch.float32, "cpu") is None

    def test_output_shape(self):
        mask = torch.ones(2, 8, dtype=torch.long)
        out = prepare_attention_mask(mask, torch.float32, "cpu")
        assert out.shape == (2, 1, 1, 8)

    def test_all_ones_gives_all_zeros(self):
        mask = torch.ones(2, 8, dtype=torch.long)
        out = prepare_attention_mask(mask, torch.float32, "cpu")
        assert (out == 0.0).all()

    def test_all_zeros_gives_all_neginf(self):
        mask = torch.zeros(2, 8, dtype=torch.long)
        out = prepare_attention_mask(mask, torch.float32, "cpu")
        assert torch.isinf(out).all()
        assert (out < 0).all()

    def test_no_nan_in_output(self):
        """The critical test: padding mask must never produce NaN."""
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[:, -3:] = 0
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            out = prepare_attention_mask(mask, dtype, "cpu")
            assert not torch.isnan(
                out
            ).any(), f"NaN in mask with dtype={dtype}"

    def test_padding_positions_are_neginf(self):
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[0, 6:] = 0
        mask[1, 4:] = 0
        out = prepare_attention_mask(mask, torch.float32, "cpu")

        assert (out[0, 0, 0, :6] == 0.0).all()
        assert torch.isinf(out[0, 0, 0, 6:]).all()
        assert (out[1, 0, 0, :4] == 0.0).all()
        assert torch.isinf(out[1, 0, 0, 4:]).all()

    def test_dtype_matches_requested(self):
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[:, -2:] = 0
        for dtype in [torch.float32, torch.bfloat16]:
            out = prepare_attention_mask(mask, dtype, "cpu")
            assert out.dtype == dtype

    def test_bf16_no_nan_all_padding(self):
        """Edge case: entirely padded sequence in bf16 must produce -inf,
        not NaN."""
        mask = torch.zeros(1, 16, dtype=torch.long)
        out = prepare_attention_mask(mask, torch.bfloat16, "cpu")
        assert not torch.isnan(out).any()
        assert torch.isinf(out).all()

    def test_single_valid_token_bf16(self):
        mask = torch.zeros(1, 16, dtype=torch.long)
        mask[0, 0] = 1
        out = prepare_attention_mask(mask, torch.bfloat16, "cpu")
        assert not torch.isnan(out).any()
        assert out[0, 0, 0, 0] == 0.0
        assert torch.isinf(out[0, 0, 0, 1:]).all()


class TestAttentionWithMask:
    """Test that attention produces no NaN with various mask configurations."""

    @pytest.fixture()
    def attn(self):
        return MultiHeadAttention(embed_dim=64, num_heads=4)

    def test_partial_padding_no_nan(self, attn):
        x = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[:, -3:] = 0
        bias = prepare_attention_mask(mask, x.dtype, x.device)
        out = attn(x, attention_bias=bias, cos_sin_emb=rotary_for(x, 4))
        assert not torch.isnan(out).any()

    def test_heavy_padding_no_nan(self, attn):
        """Only 1 real token per sequence."""
        x = torch.randn(2, 16, 64)
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 0] = 1
        bias = prepare_attention_mask(mask, x.dtype, x.device)
        out = attn(x, attention_bias=bias, cos_sin_emb=rotary_for(x, 4))
        assert not torch.isnan(out).any()

    def test_uneven_padding_no_nan(self, attn):
        x = torch.randn(4, 12, 64)
        mask = torch.ones(4, 12, dtype=torch.long)
        mask[0, 3:] = 0  # 3 real
        mask[1, 8:] = 0  # 8 real
        mask[2, :] = 1  # all real
        mask[3, 1:] = 0  # 1 real
        bias = prepare_attention_mask(mask, x.dtype, x.device)
        out = attn(x, attention_bias=bias, cos_sin_emb=rotary_for(x, 4))
        assert not torch.isnan(out).any()

    def test_gradient_flow_with_padding(self, attn):
        x = torch.randn(2, 8, 64, requires_grad=True)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[:, -4:] = 0
        bias = prepare_attention_mask(mask, x.dtype, x.device)
        out = attn(x, attention_bias=bias, cos_sin_emb=rotary_for(x, 4))
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestGetNorm:
    def test_rms_norm(self):
        norm = get_norm("rms", 64)
        assert norm is not None
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_scale_norm(self):
        norm = get_norm("scale", 64)
        assert isinstance(norm, ScaleNorm)

    def test_none(self):
        norm = get_norm(None, 64)
        assert norm is None

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_norm("invalid", 64)


class TestEmbedding:
    def test_padding_idx_zero(self):
        emb = Embedding(100, 32, padding_idx=0)
        assert torch.all(emb.weight[0] == 0)

    def test_output_shape(self):
        emb = Embedding(100, 32)
        ids = torch.tensor([[1, 2, 3]])
        out = emb(ids)
        assert out.shape == (1, 3, 32)


class TestFeedForward:
    def test_output_shape(self):
        ff = FeedForward(embed_dim=32, ff_dim=64, bias=True)
        x = torch.randn(2, 8, 32)
        out = ff(x)
        assert out.shape == (2, 8, 32)

    def test_gated(self):
        ff = FeedForward(embed_dim=32, ff_dim=64, gated=True)
        x = torch.randn(2, 8, 32)
        out = ff(x)
        assert out.shape == (2, 8, 32)

    def test_no_norm(self):
        ff = FeedForward(embed_dim=32, ff_dim=64, ff_norm_type=None)
        assert ff.norm is None


class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = attn(x, cos_sin_emb=rotary_for(x, attn.num_heads))
        assert out.shape == (2, 8, 64)

    def test_gqa(self):
        attn = MultiHeadAttention(embed_dim=64, num_heads=8, num_kv_heads=4)
        x = torch.randn(2, 8, 64)
        out = attn(x, cos_sin_emb=rotary_for(x, attn.num_heads))
        assert out.shape == (2, 8, 64)
        assert attn.num_kv_groups == 2

    def test_with_padding_mask(self):
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[:, -2:] = 0
        bias = prepare_attention_mask(mask, x.dtype, x.device)
        out = attn(x, attention_bias=bias, cos_sin_emb=rotary_for(x, 4))
        assert out.shape == (2, 8, 64)

    def test_with_capping(self):
        attn = MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            capping_value=30.0,
        )
        x = torch.randn(2, 8, 64)
        out = attn(x, cos_sin_emb=rotary_for(x, attn.num_heads))
        assert out.shape == (2, 8, 64)

    def test_with_qk_norm(self):
        attn = MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            qk_norm=True,
        )
        assert attn.qk_norm
        x = torch.randn(2, 8, 64)
        out = attn(x, cos_sin_emb=rotary_for(x, attn.num_heads))
        assert out.shape == (2, 8, 64)

    def test_invalid_kv_heads_raises(self):
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadAttention(embed_dim=64, num_heads=8, num_kv_heads=3)

    def test_gradient_flow(self):
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = attn(x, cos_sin_emb=rotary_for(x, attn.num_heads))
        out.sum().backward()
        assert x.grad is not None


class TestEncoderLayer:
    @pytest.fixture
    def config(self):
        return AresConfig(
            embed_dim=64,
            num_heads=4,
            num_kv_heads=4,
            ff_dim=128,
            num_layers=1,
            vocab_size=30,
            norm_type="rms",
        )

    def test_output_shape(self, config):
        layer = EncoderLayer(config, layer_idx=0)
        x = torch.randn(2, 8, 64)
        out = layer(x, cos_sin_emb=rotary_for(x, 4))
        assert out.shape == (2, 8, 64)

    def test_residual_connection(self, config):
        layer = EncoderLayer(config, layer_idx=0)
        x = torch.zeros(2, 8, 64)
        out = layer(x, cos_sin_emb=rotary_for(x, 4))
        assert out.shape == x.shape

    def test_moe_layer_expert_choice(self):
        config = AresConfig(
            embed_dim=64,
            num_heads=4,
            num_kv_heads=4,
            ff_dim=128,
            num_experts=2,
            moe_after_num_layers=0,
            moe_type="expert_choice",
            norm_type="rms",
        )
        layer = EncoderLayer(config, layer_idx=0)
        assert layer._using_moe
        x = torch.randn(2, 8, 64)
        out = layer(x, cos_sin_emb=rotary_for(x, 4))
        assert out.shape == (2, 8, 64)

    def test_moe_layer_soft_router(self):
        config = AresConfig(
            embed_dim=64,
            num_heads=4,
            num_kv_heads=4,
            ff_dim=128,
            num_experts=2,
            moe_after_num_layers=0,
            moe_type="soft_router",
            moe_num_slots=2,
            norm_type="rms",
        )
        layer = EncoderLayer(config, layer_idx=0)
        assert layer._using_moe
        x = torch.randn(2, 8, 64)
        out = layer(x, cos_sin_emb=rotary_for(x, 4))
        assert out.shape == (2, 8, 64)

    def test_dense_layer_below_threshold(self):
        config = AresConfig(
            embed_dim=64,
            num_heads=4,
            num_kv_heads=4,
            ff_dim=128,
            moe_after_num_layers=2,
            norm_type="rms",
        )
        layer = EncoderLayer(config, layer_idx=0)
        assert not layer._using_moe


class TestAresModel:
    @pytest.fixture
    def config(self):
        return AresConfig(
            embed_dim=64,
            vocab_size=30,
            num_heads=4,
            num_kv_heads=4,
            num_layers=2,
            ff_dim=128,
            norm_type="rms",
        )

    @pytest.fixture
    def model(self, config):
        return Ares(config)

    def test_forward_shape(self, model):
        ids = torch.randint(0, 30, (2, 16))
        out = model(ids)
        assert out.logits.shape == (2, 16, 30)

    def test_forward_with_mask(self, model):
        ids = torch.randint(0, 30, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)
        mask[:, -4:] = 0
        out = model(ids, attention_mask=mask)
        assert out.logits.shape == (2, 16, 30)

    def test_forward_with_labels(self, model):
        ids = torch.randint(0, 30, (2, 16))
        labels = torch.randint(0, 30, (2, 16))
        out = model(ids, labels=labels)
        assert out.loss is not None
        assert out.loss.ndim == 0

    def test_logits_capping(self):
        cfg = AresConfig(
            embed_dim=64,
            vocab_size=30,
            num_heads=4,
            num_kv_heads=4,
            num_layers=2,
            ff_dim=128,
            norm_type="rms",
            logits_capping_value=30.0,
        )
        capped_model = Ares(cfg)
        ids = torch.randint(0, 30, (2, 16))
        out = capped_model(ids)
        assert out.logits.abs().max() <= cfg.logits_capping_value

    def test_no_loss_without_labels(self, model):
        ids = torch.randint(0, 30, (2, 16))
        out = model(ids)
        assert out.loss is None

    def test_moe_model(self):
        config = AresConfig(
            embed_dim=64,
            vocab_size=30,
            num_heads=4,
            num_kv_heads=4,
            num_layers=2,
            ff_dim=128,
            num_experts=2,
            moe_after_num_layers=1,
            moe_type="expert_choice",
            norm_type="rms",
        )
        model = Ares(config)
        ids = torch.randint(0, 30, (2, 8))
        out = model(ids)
        assert out.logits.shape == (2, 8, 30)

    def test_switching_moe_type_changes_model_layers(self):
        common_kwargs = dict(
            embed_dim=64,
            vocab_size=30,
            num_heads=4,
            num_kv_heads=4,
            num_layers=3,
            ff_dim=128,
            num_experts=2,
            moe_after_num_layers=1,
            norm_type="rms",
        )
        expert_choice_model = Ares(
            AresConfig(
                **common_kwargs,
                moe_type="expert_choice",
            )
        )
        soft_router_model = Ares(
            AresConfig(
                **common_kwargs,
                moe_type="soft_router",
                moe_num_slots=2,
            )
        )

        assert isinstance(expert_choice_model.layers[0].ff, FeedForward)
        assert isinstance(soft_router_model.layers[0].ff, FeedForward)

        assert isinstance(
            expert_choice_model.layers[1].ff,
            ExpertChoiceRouting,
        )
        assert isinstance(
            expert_choice_model.layers[2].ff,
            ExpertChoiceRouting,
        )
        assert isinstance(soft_router_model.layers[1].ff, SoftRouter)
        assert isinstance(soft_router_model.layers[2].ff, SoftRouter)

        assert type(expert_choice_model.layers[1].ff) is not type(
            soft_router_model.layers[1].ff
        )


# ── NaN / Inf stability tests ────────────────────────────────────────────────

DENSE_CFG = dict(
    embed_dim=64,
    vocab_size=30,
    num_heads=4,
    num_kv_heads=4,
    num_layers=2,
    ff_dim=128,
    norm_type="rms",
)
EC_CFG = dict(
    **DENSE_CFG,
    num_experts=2,
    moe_after_num_layers=1,
    moe_type="expert_choice",
)
SR_CFG = dict(
    **DENSE_CFG,
    num_experts=2,
    moe_after_num_layers=1,
    moe_type="soft_router",
    moe_num_slots=4,
)


class TestNaNObserver:
    """Run the model under NaNObserver with various attention masks.

    ``nan_only=True`` because masked attention legitimately produces ``-inf``.
    """

    @pytest.fixture(
        params=[DENSE_CFG, EC_CFG, SR_CFG],
        ids=["dense", "expert_choice", "soft_router"],
    )
    def model(self, request):
        m = Ares(AresConfig(**request.param))
        m.eval()
        return m

    def test_no_mask(self, model):
        ids = torch.randint(0, 30, (2, 16))
        with NaNObserver(nan_only=True):
            out = model(ids)
        assert not torch.isnan(out.logits).any()

    def test_full_ones_mask(self, model):
        ids = torch.randint(0, 30, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)
        with NaNObserver(nan_only=True):
            out = model(ids, attention_mask=mask)
        assert not torch.isnan(out.logits).any()

    def test_partial_padding(self, model):
        ids = torch.randint(0, 30, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)
        mask[:, -4:] = 0
        with NaNObserver(nan_only=True):
            out = model(ids, attention_mask=mask)
        assert not torch.isnan(out.logits).any()

    def test_single_valid_token(self, model):
        ids = torch.randint(0, 30, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 0] = 1
        with NaNObserver(nan_only=True):
            out = model(ids, attention_mask=mask)
        assert not torch.isnan(out.logits).any()

    def test_uneven_lengths(self, model):
        ids = torch.randint(0, 30, (3, 12))
        mask = torch.ones(3, 12, dtype=torch.long)
        mask[0, 6:] = 0  # half padding
        mask[1, 10:] = 0  # slight padding
        mask[2, :] = 1  # no padding
        with NaNObserver(nan_only=True):
            out = model(ids, attention_mask=mask)
        assert not torch.isnan(out.logits).any()

    def test_with_labels(self, model):
        ids = torch.randint(0, 30, (2, 16))
        labels = ids.clone()
        labels[:, ::2] = -100
        mask = torch.ones(2, 16, dtype=torch.long)
        mask[:, -3:] = 0
        with NaNObserver(nan_only=True):
            out = model(ids, attention_mask=mask, labels=labels)
        assert out.loss is not None
        assert not torch.isnan(out.loss)

    def test_all_padding(self, model):
        """All-zero mask is an extreme edge case; just verify no hard crash."""
        ids = torch.randint(0, 30, (2, 8))
        mask = torch.zeros(2, 8, dtype=torch.long)
        out = model(ids, attention_mask=mask)
        assert out.logits.shape == (2, 8, 30)
