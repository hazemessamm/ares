import torch
import pytest
from ares.models.soft_router import SoftRouter, SoftMoEFeedForward, softmax

EMBED_DIM = 32
NUM_EXPERTS = 4
NUM_SLOTS = 2
FF_DIM = 64
BATCH = 2
SEQ_LEN = 8


class TestMultiDimSoftmax:
    def test_single_dim(self):
        x = torch.randn(4, 5)
        out = softmax(x, dim=1)
        sums = out.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_multi_dim(self):
        x = torch.randn(2, 3, 4)
        out = softmax(x, dim=(1, 2))
        sums = out.sum(dim=(1, 2))
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_preserves_input_dtype(self):
        x = torch.randn(4, 5, dtype=torch.bfloat16)
        out = softmax(x, dim=1)
        assert out.dtype == x.dtype

    def test_non_negative(self):
        x = torch.randn(4, 5)
        out = softmax(x, dim=0)
        assert (out >= 0).all()


class TestSoftMoEFeedForward:
    @pytest.fixture
    def ff(self):
        return SoftMoEFeedForward(
            num_experts=NUM_EXPERTS,
            embed_dim=EMBED_DIM,
            ff_dim=FF_DIM,
            bias=True,
            gated=False,
        )

    @pytest.fixture
    def ff_gated(self):
        return SoftMoEFeedForward(
            num_experts=NUM_EXPERTS,
            embed_dim=EMBED_DIM,
            ff_dim=FF_DIM,
            bias=True,
            gated=True,
        )

    def test_output_shape(self, ff):
        x = torch.randn(BATCH, NUM_EXPERTS, NUM_SLOTS, EMBED_DIM)
        out = ff(x)
        assert out.shape == (BATCH, NUM_EXPERTS, NUM_SLOTS, EMBED_DIM)

    def test_gated_output_shape(self, ff_gated):
        x = torch.randn(BATCH, NUM_EXPERTS, NUM_SLOTS, EMBED_DIM)
        out = ff_gated(x)
        assert out.shape == (BATCH, NUM_EXPERTS, NUM_SLOTS, EMBED_DIM)

    def test_no_bias(self):
        ff = SoftMoEFeedForward(
            num_experts=NUM_EXPERTS,
            embed_dim=EMBED_DIM,
            ff_dim=FF_DIM,
            bias=False,
        )
        assert ff.b1 is None
        assert ff.b2 is None
        x = torch.randn(BATCH, NUM_EXPERTS, NUM_SLOTS, EMBED_DIM)
        out = ff(x)
        assert out.shape == (BATCH, NUM_EXPERTS, NUM_SLOTS, EMBED_DIM)


class TestSoftRouter:
    @pytest.fixture
    def router(self):
        return SoftRouter(
            embed_dim=EMBED_DIM,
            num_experts=NUM_EXPERTS,
            num_slots=NUM_SLOTS,
            ff_dim=FF_DIM,
            gated=True,
            bias=False,
        )

    def test_output_shape(self, router):
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        out = router(x)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)

    def test_output_dtype_matches_input(self, router):
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        out = router(x)
        assert out.dtype == x.dtype

    def test_with_padding_mask(self, router):
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.bool)
        mask[:, -2:] = False
        out = router(x, padding_mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)

    def test_noise_only_during_training(self, router):
        router.noise_scale = 0.1
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)

        router.eval()
        torch.manual_seed(42)
        out1 = router(x)
        torch.manual_seed(42)
        out2 = router(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_with_normalize(self):
        router = SoftRouter(
            embed_dim=EMBED_DIM,
            num_experts=NUM_EXPERTS,
            num_slots=NUM_SLOTS,
            ff_dim=FF_DIM,
            normalize=True,
        )
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        out = router(x)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)

    def test_gradient_flow(self, router):
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, requires_grad=True)
        out = router(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
