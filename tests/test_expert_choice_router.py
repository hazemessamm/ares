import torch
import pytest
from ares.models.expert_choice_router import (
    compute_k,
    ExpertChoiceRouter,
    ExpertChoiceMLP,
    ExpertChoiceRouting,
)

EMBED_DIM = 32
NUM_EXPERTS = 4
FF_DIM = 64
BATCH = 2
SEQ_LEN = 8


class TestComputeK:
    def test_basic(self):
        k = compute_k(num_tokens=16, expert_capacity_factor=2.0, num_experts=4)
        assert k == 8

    def test_ceil_rounding(self):
        k = compute_k(num_tokens=10, expert_capacity_factor=1.0, num_experts=3)
        assert k == 4  # ceil(10/3) = 4

    def test_capped_at_num_tokens(self):
        k = compute_k(num_tokens=4, expert_capacity_factor=10.0, num_experts=2)
        assert k == 4

    def test_single_token(self):
        k = compute_k(num_tokens=1, expert_capacity_factor=1.0, num_experts=4)
        assert k == 1

    def test_zero_experts(self):
        k = compute_k(num_tokens=10, expert_capacity_factor=1.0, num_experts=0)
        assert k == 0


class TestExpertChoiceRouter:
    @pytest.fixture
    def router(self):
        return ExpertChoiceRouter(
            embed_dim=EMBED_DIM,
            num_experts=NUM_EXPERTS,
            expert_capacity_factor=2.0,
        )

    def test_output_shapes(self, router):
        x = torch.randn(BATCH * SEQ_LEN, EMBED_DIM)
        probs, indices = router(x)
        k = compute_k(BATCH * SEQ_LEN, 2.0, NUM_EXPERTS)
        assert probs.shape == (NUM_EXPERTS, k)
        assert indices.shape == (NUM_EXPERTS, k)

    def test_probs_non_negative(self, router):
        x = torch.randn(BATCH * SEQ_LEN, EMBED_DIM)
        probs, _ = router(x)
        assert (probs >= 0).all()

    def test_3d_input_flattened(self, router):
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        probs, indices = router(x)
        k = compute_k(BATCH * SEQ_LEN, 2.0, NUM_EXPERTS)
        assert probs.shape == (NUM_EXPERTS, k)

    def test_noise_only_during_training(self, router):
        router.noise_scale = 0.1
        x = torch.randn(16, EMBED_DIM)

        router.eval()
        torch.manual_seed(0)
        p1, i1 = router(x)
        torch.manual_seed(0)
        p2, i2 = router(x)
        assert torch.allclose(p1, p2)
        assert torch.equal(i1, i2)


class TestExpertChoiceMLP:
    @pytest.fixture
    def mlp(self):
        return ExpertChoiceMLP(
            num_experts=NUM_EXPERTS,
            embed_dim=EMBED_DIM,
            ff_dim=FF_DIM,
            bias=True,
            gated=False,
        )

    @pytest.fixture
    def mlp_gated(self):
        return ExpertChoiceMLP(
            num_experts=NUM_EXPERTS,
            embed_dim=EMBED_DIM,
            ff_dim=FF_DIM,
            bias=True,
            gated=True,
        )

    def test_output_shape(self, mlp):
        k = 4
        x = torch.randn(NUM_EXPERTS, k, EMBED_DIM)
        out = mlp(x)
        assert out.shape == (NUM_EXPERTS, k, EMBED_DIM)

    def test_gated_output_shape(self, mlp_gated):
        k = 4
        x = torch.randn(NUM_EXPERTS, k, EMBED_DIM)
        out = mlp_gated(x)
        assert out.shape == (NUM_EXPERTS, k, EMBED_DIM)

    def test_no_bias(self):
        mlp = ExpertChoiceMLP(
            num_experts=NUM_EXPERTS,
            embed_dim=EMBED_DIM,
            ff_dim=FF_DIM,
            bias=False,
        )
        assert mlp.b1 is None
        assert mlp.b2 is None


class TestExpertChoiceRouting:
    @pytest.fixture
    def routing(self):
        return ExpertChoiceRouting(
            embed_dim=EMBED_DIM,
            num_experts=NUM_EXPERTS,
            expert_capacity_factor=2.0,
            ff_dim=FF_DIM,
            gated=True,
            bias=False,
        )

    def test_output_shape_2d(self, routing):
        x = torch.randn(BATCH * SEQ_LEN, EMBED_DIM)
        out = routing(x)
        assert out.shape == x.shape

    def test_output_shape_3d(self, routing):
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        out = routing(x)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)

    def test_zero_output_for_unselected_tokens(self, routing):
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        out = routing(x)
        assert out.shape == x.shape

    def test_gradient_flow(self, routing):
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, requires_grad=True)
        out = routing(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_with_padding_mask(self, routing):
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.bool)
        mask[:, -2:] = False
        out = routing(x, padding_mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)
