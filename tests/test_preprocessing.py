import random
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock

from ares.preprocessing.utils import (
    MLMProbabilitySampler,
    create_default_mlm_weights,
)
from ares.preprocessing.truncation import random_truncation
from ares.preprocessing.scheduling import LinearScheduler
from ares.preprocessing.noising import SequenceCorruptor

# ── MLMProbabilitySampler ──────────────────────────────────────────────


class TestMLMProbabilitySampler:
    def test_sample_returns_triplet(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.15],
            masking_probs=[0.8],
            mutation_probs=[0.1],
        )
        mlm, mask, mut = sampler.sample()
        assert mlm == 0.15
        assert mask == 0.8
        assert mut == 0.1

    def test_sample_from_multiple(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.8, 0.8, 0.8],
            mutation_probs=[0.1, 0.1, 0.1],
        )
        mlm, _, _ = sampler.sample()
        assert mlm in [0.1, 0.2, 0.3]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            MLMProbabilitySampler(
                mlm_probs=[0.1, 0.2],
                masking_probs=[0.8],
                mutation_probs=[0.1, 0.1],
            )

    def test_with_scheduler(self):
        scheduler = LinearScheduler(
            initial_weights=[1.0, 0.0],
            final_weights=[0.0, 1.0],
            warmup_steps=10,
        )
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.3],
            masking_probs=[0.8, 0.8],
            mutation_probs=[0.1, 0.1],
            scheduler=scheduler,
        )
        mlm, _, _ = sampler.sample()
        assert mlm in [0.1, 0.3]

    def test_scheduler_weights_bias_sampling(self):
        scheduler = MagicMock()
        scheduler.sample.return_value = [0.0, 0.0, 1.0]
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.8, 0.8, 0.8],
            mutation_probs=[0.1, 0.1, 0.1],
            scheduler=scheduler,
        )
        for _ in range(20):
            mlm, _, _ = sampler.sample()
            assert mlm == 0.3

    def test_scheduler_zero_weights_fallback(self):
        scheduler = MagicMock()
        scheduler.sample.return_value = [0.0, 0.0]
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2],
            masking_probs=[0.8, 0.8],
            mutation_probs=[0.1, 0.1],
            scheduler=scheduler,
        )
        mlm, _, _ = sampler.sample()
        assert mlm in [0.1, 0.2]


# ── create_default_mlm_weights ────────────────────────────────────────


class TestCreateDefaultMLMWeights:
    def test_basic_range(self):
        weights = create_default_mlm_weights(0.1, 0.3, 0.1)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 3
        np.testing.assert_allclose(weights, [0.1, 0.2, 0.3], atol=1e-10)

    def test_single_value(self):
        weights = create_default_mlm_weights(0.5, 0.5, 0.1)
        assert len(weights) == 1
        np.testing.assert_allclose(weights[0], 0.5, atol=1e-10)


# ── random_truncation ─────────────────────────────────────────────────


class TestRandomTruncation:
    def test_short_sequence_unchanged(self):
        seq, start, end, truncated = random_truncation("ACDEF", max_length=10)
        assert seq == "ACDEF"
        assert start == 0
        assert end == 5
        assert truncated is False

    def test_exact_length(self):
        seq, start, end, truncated = random_truncation("ACDEF", max_length=5)
        assert seq == "ACDEF"
        assert truncated is False

    def test_truncation_length(self):
        seq, start, end, truncated = random_truncation(
            "ACDEFGHIJK", max_length=5
        )
        assert len(seq) == 5
        assert truncated is True
        assert end - start == 5

    def test_truncation_is_substring(self):
        original = "ACDEFGHIJKLMNOP"
        seq, start, end, truncated = random_truncation(original, max_length=5)
        assert seq == original[start:end]

    def test_truncation_randomness(self):
        original = "A" * 100
        starts = set()
        for _ in range(50):
            _, start, _, _ = random_truncation(original, max_length=10)
            starts.add(start)
        assert len(starts) > 1, "Truncation should be random"


# ── SequenceCorruptor ─────────────────────────────────────────────────


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.all_special_ids = [0, 1, 2, 3, 4]
    tok.get_vocab.return_value = {chr(i + 65): i + 5 for i in range(20)}
    tok.pad_token_id = 0
    tok.mask_token_id = 4
    tok.eos_token_id = 2
    tok.cls_token_id = 1
    return tok


class TestSequenceCorruptor:
    @pytest.fixture
    def corruptor(self):
        tok = _make_mock_tokenizer()
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.5],
            masking_probs=[0.8],
            mutation_probs=[0.1],
        )
        return SequenceCorruptor(
            tokenizer=tok, mlm_probability_sampler=sampler
        )

    def test_output_shapes(self, corruptor):
        seqs = torch.tensor([[1, 5, 6, 7, 8, 2, 0, 0]])
        corrupted, labels = corruptor(seqs)
        assert corrupted.shape == seqs.shape
        assert labels.shape == seqs.shape

    def test_special_tokens_preserved(self, corruptor):
        seqs = torch.tensor([[1, 5, 6, 7, 8, 2, 0, 0]])
        corrupted, labels = corruptor(seqs)
        assert corrupted[0, 0].item() == 1
        assert corrupted[0, 5].item() == 2
        assert corrupted[0, 6].item() == 0
        assert corrupted[0, 7].item() == 0

    def test_labels_ignore_non_mlm(self, corruptor):
        seqs = torch.tensor([[1, 5, 6, 7, 8, 2, 0, 0]])
        _, labels = corruptor(seqs)
        assert labels[0, 0].item() == -100
        assert labels[0, 5].item() == -100
        assert labels[0, 6].item() == -100
        assert labels[0, 7].item() == -100

    def test_does_not_modify_input(self, corruptor):
        seqs = torch.tensor([[1, 5, 6, 7, 8, 2, 0, 0]])
        original = seqs.clone()
        corruptor(seqs)
        assert torch.equal(seqs, original)

    def test_mask_token_appears(self):
        tok = _make_mock_tokenizer()
        sampler = MLMProbabilitySampler(
            mlm_probs=[1.0],
            masking_probs=[1.0],
            mutation_probs=[0.0],
        )
        corruptor = SequenceCorruptor(
            tokenizer=tok, mlm_probability_sampler=sampler
        )
        seqs = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 2]])
        corrupted, _ = corruptor(seqs)
        non_special = corrupted[0, 1:7]
        assert (
            non_special == 4
        ).all(), "All non-special tokens should be masked"
