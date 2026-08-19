"""Tests for the dataset pipeline using a dummy in-memory dataset.

Replicates the exact ops from HFDataset.__iter__ (truncation, boundary tokens,
tokenization, noising, label masking, position IDs) and validates shapes,
dtypes, value ranges, and special-token handling.
"""

import random
import pytest
import torch
from torch.utils.data import IterableDataset, DataLoader
from ares.tokenization import AresProteinTokenizer
from ares.preprocessing.noising import SequenceCorruptor
from ares.preprocessing.utils import MLMProbabilitySampler
from ares.preprocessing.truncation import random_truncation
from ares.pipelines.dataset import add_boundary_tokens, AresCollator

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
MAX_LENGTH = 32


@pytest.fixture()
def tokenizer():
    return AresProteinTokenizer()


@pytest.fixture()
def noise_fn(tokenizer):
    sampler = MLMProbabilitySampler(
        mlm_probs=[0.15],
        masking_probs=[0.80],
        mutation_probs=[0.10],
    )
    return SequenceCorruptor(tokenizer, sampler)


@pytest.fixture()
def special_token_ids(tokenizer):
    return list(tokenizer.all_special_ids)


def _make_sequence(length: int) -> list[str]:
    """Generate a deterministic amino-acid sequence of a given length."""
    return [AMINO_ACIDS[i % len(AMINO_ACIDS)] for i in range(length)]


def _pipeline(sequence, tokenizer, noise_fn, special_token_ids, max_length):
    """Run the exact same ops as HFDataset.__iter__ on a single sequence."""
    if isinstance(sequence, str):
        sequence = list(sequence)

    truncated, start_idx, end_idx, _ = random_truncation(sequence, max_length)

    add_cls = start_idx == 0
    add_eos = end_idx == len(sequence)
    num_special = int(add_cls) + int(add_eos)

    target_aa_len = max_length - num_special
    if len(truncated) > target_aa_len:
        excess = len(truncated) - target_aa_len
        if add_eos and not add_cls:
            truncated = truncated[excess:]
            start_idx += excess
        else:
            truncated = truncated[:target_aa_len]
            end_idx = start_idx + target_aa_len

    truncated = add_boundary_tokens(truncated, add_cls, add_eos)

    encoded_inputs = tokenizer(
        truncated,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    input_ids, labels = noise_fn(input_ids, excluded_ids=special_token_ids)

    labels = labels.masked_fill(
        input_ids == tokenizer.pad_token_id,
        -100,
    )

    position_ids = torch.arange(
        start_idx,
        start_idx + max_length,
    ).unsqueeze(0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": position_ids,
    }


# ── Shape and dtype tests ───────────────────────────────────────────────


class TestOutputShapes:
    @pytest.mark.parametrize(
        "seq_len", [1, 5, MAX_LENGTH - 2, MAX_LENGTH, MAX_LENGTH + 50]
    )
    def test_shapes(self, tokenizer, noise_fn, special_token_ids, seq_len):
        seq = _make_sequence(seq_len)
        out = _pipeline(
            seq, tokenizer, noise_fn, special_token_ids, MAX_LENGTH
        )

        assert out["input_ids"].shape == (1, MAX_LENGTH)
        assert out["attention_mask"].shape == (1, MAX_LENGTH)
        assert out["labels"].shape == (1, MAX_LENGTH)
        assert out["position_ids"].shape == (1, MAX_LENGTH)

    @pytest.mark.parametrize("seq_len", [1, MAX_LENGTH, MAX_LENGTH + 50])
    def test_dtypes(self, tokenizer, noise_fn, special_token_ids, seq_len):
        seq = _make_sequence(seq_len)
        out = _pipeline(
            seq, tokenizer, noise_fn, special_token_ids, MAX_LENGTH
        )

        assert out["input_ids"].dtype == torch.long
        assert out["attention_mask"].dtype == torch.long
        assert out["labels"].dtype == torch.long
        assert out["position_ids"].dtype == torch.long


# ── Boundary token tests ────────────────────────────────────────────────


class TestBoundaryTokens:
    def test_short_sequence_has_cls_and_eos(
        self, tokenizer, noise_fn, special_token_ids
    ):
        """A sequence shorter than max_length starts at 0 and covers the whole
        thing, so it should get both <cls> and <eos>."""
        seq = _make_sequence(5)
        out = _pipeline(
            seq, tokenizer, noise_fn, special_token_ids, MAX_LENGTH
        )

        ids = out["input_ids"][0]
        assert ids[0].item() == tokenizer.cls_token_id
        assert ids[6].item() == tokenizer.convert_tokens_to_ids("<eos>")

    def test_full_length_sequence_has_cls_and_eos(
        self, tokenizer, noise_fn, special_token_ids
    ):
        """A sequence exactly max_length-2 should fit with both boundary tokens."""
        seq = _make_sequence(MAX_LENGTH - 2)
        out = _pipeline(
            seq, tokenizer, noise_fn, special_token_ids, MAX_LENGTH
        )

        ids = out["input_ids"][0]
        mask = out["attention_mask"][0]
        assert ids[0].item() == tokenizer.cls_token_id
        last_real = mask.sum().item() - 1
        assert ids[last_real].item() == tokenizer.convert_tokens_to_ids(
            "<eos>"
        )


# ── Attention mask tests ────────────────────────────────────────────────


class TestAttentionMask:
    def test_short_sequence_padding(
        self, tokenizer, noise_fn, special_token_ids
    ):
        seq = _make_sequence(5)
        out = _pipeline(
            seq, tokenizer, noise_fn, special_token_ids, MAX_LENGTH
        )

        mask = out["attention_mask"][0]
        num_real = 5 + 2  # 5 AA + <cls> + <eos>
        assert mask[:num_real].sum().item() == num_real
        assert mask[num_real:].sum().item() == 0

    def test_long_sequence_no_padding(
        self, tokenizer, noise_fn, special_token_ids
    ):
        """A sequence longer than max_length should have no padding."""
        seq = _make_sequence(MAX_LENGTH + 50)
        out = _pipeline(
            seq, tokenizer, noise_fn, special_token_ids, MAX_LENGTH
        )

        mask = out["attention_mask"][0]
        assert mask.sum().item() == MAX_LENGTH


# ── Label tests ─────────────────────────────────────────────────────────


class TestLabels:
    def test_padding_positions_are_ignored(
        self, tokenizer, noise_fn, special_token_ids
    ):
        seq = _make_sequence(5)
        out = _pipeline(
            seq, tokenizer, noise_fn, special_token_ids, MAX_LENGTH
        )

        labels = out["labels"][0]
        mask = out["attention_mask"][0]
        padding_labels = labels[mask == 0]
        assert (padding_labels == -100).all()

    def test_special_tokens_are_ignored(
        self, tokenizer, noise_fn, special_token_ids
    ):
        """<cls> and <eos> positions should never be MLM targets."""
        seq = _make_sequence(5)
        out = _pipeline(
            seq, tokenizer, noise_fn, special_token_ids, MAX_LENGTH
        )

        labels = out["labels"][0]
        assert labels[0].item() == -100  # <cls>

    def test_label_values_in_range(
        self, tokenizer, noise_fn, special_token_ids
    ):
        seq = _make_sequence(MAX_LENGTH + 10)
        out = _pipeline(
            seq, tokenizer, noise_fn, special_token_ids, MAX_LENGTH
        )

        labels = out["labels"][0]
        vocab_size = len(tokenizer)
        valid = (labels == -100) | ((labels >= 0) & (labels < vocab_size))
        assert valid.all()


# ── Truncation tests ────────────────────────────────────────────────────


class TestTruncation:
    def test_short_no_truncation(self):
        """Sequence shorter than max_length: no truncation, start=0, end=len."""
        seq = _make_sequence(10)
        truncated, start_idx, end_idx, was_truncated = random_truncation(
            seq, MAX_LENGTH
        )

        assert was_truncated is False
        assert start_idx == 0
        assert end_idx == len(seq)
        assert truncated == seq

    def test_exact_length_no_truncation(self):
        """Sequence exactly max_length: no truncation needed."""
        seq = _make_sequence(MAX_LENGTH)
        truncated, start_idx, end_idx, was_truncated = random_truncation(
            seq, MAX_LENGTH
        )

        assert was_truncated is False
        assert start_idx == 0
        assert end_idx == MAX_LENGTH
        assert truncated == seq

    def test_long_sequence_is_truncated(self):
        """Sequence longer than max_length: must be truncated."""
        seq = _make_sequence(MAX_LENGTH * 3)
        truncated, start_idx, end_idx, was_truncated = random_truncation(
            seq, MAX_LENGTH
        )

        assert was_truncated is True
        assert len(truncated) == MAX_LENGTH

    def test_indices_are_valid_bounds(self):
        """start_idx and end_idx must be valid slicing bounds for the original."""
        seq = _make_sequence(MAX_LENGTH * 3)
        truncated, start_idx, end_idx, _ = random_truncation(seq, MAX_LENGTH)

        assert 0 <= start_idx < len(seq)
        assert start_idx < end_idx <= len(seq)
        assert end_idx - start_idx == len(truncated)

    def test_truncated_content_matches_original(self):
        """The truncated fragment must be the exact slice from the original."""
        seq = _make_sequence(MAX_LENGTH * 3)
        truncated, start_idx, end_idx, _ = random_truncation(seq, MAX_LENGTH)

        assert truncated == seq[start_idx:end_idx]

    @pytest.mark.parametrize("run", range(20))
    def test_indices_stay_in_range_across_runs(self, run):
        """Across many random runs, indices must always be valid."""
        seq = _make_sequence(MAX_LENGTH + 15)
        truncated, start_idx, end_idx, _ = random_truncation(seq, MAX_LENGTH)

        assert 0 <= start_idx <= len(seq) - MAX_LENGTH
        assert end_idx == start_idx + MAX_LENGTH
        assert truncated == seq[start_idx:end_idx]


# ── Truncation + boundary token adjustment tests ───────────────────────


class TestTruncationWithBoundaryTokens:
    """Verifies the full truncation → boundary-token adjustment logic from
    the dataset pipeline, including that start_idx/end_idx stay consistent
    after trimming to make room for <cls>/<eos>."""

    def _truncate_and_adjust(self, sequence, max_length):
        """Replicates the truncation + adjustment logic from HFDataset.__iter__,
        returning intermediate values for inspection."""
        truncated, start_idx, end_idx, was_truncated = random_truncation(
            sequence,
            max_length,
        )
        add_cls = start_idx == 0
        add_eos = end_idx == len(sequence)
        num_special = int(add_cls) + int(add_eos)

        target_aa_len = max_length - num_special
        if len(truncated) > target_aa_len:
            excess = len(truncated) - target_aa_len
            if add_eos and not add_cls:
                truncated = truncated[excess:]
                start_idx += excess
            else:
                truncated = truncated[:target_aa_len]
                end_idx = start_idx + target_aa_len

        return truncated, start_idx, end_idx, add_cls, add_eos

    def test_short_gets_both_tokens(self):
        seq = _make_sequence(5)
        trunc, start, end, add_cls, add_eos = self._truncate_and_adjust(
            seq, MAX_LENGTH
        )

        assert add_cls is True
        assert add_eos is True
        assert start == 0
        assert end == len(seq)
        assert len(trunc) == len(seq)

    def test_exact_fit_with_both_tokens(self):
        """max_length - 2 AAs should fit exactly with both boundary tokens."""
        seq = _make_sequence(MAX_LENGTH - 2)
        trunc, start, end, add_cls, add_eos = self._truncate_and_adjust(
            seq, MAX_LENGTH
        )

        assert add_cls is True
        assert add_eos is True
        assert len(trunc) + 2 == MAX_LENGTH

    def test_full_length_trimmed_for_boundary(self):
        """A sequence of exactly max_length gets truncated to start=0, so
        it needs <cls> and <eos>. The AA content is trimmed to max_length-2."""
        seq = _make_sequence(MAX_LENGTH)
        trunc, start, end, add_cls, add_eos = self._truncate_and_adjust(
            seq, MAX_LENGTH
        )

        assert add_cls is True
        assert add_eos is True
        assert len(trunc) == MAX_LENGTH - 2
        assert trunc == seq[: MAX_LENGTH - 2]

    def test_truncated_content_still_matches_original(self):
        """After adjustment, the AA content must still be a valid slice of the original."""
        seq = _make_sequence(MAX_LENGTH * 3)
        trunc, start, end, _, _ = self._truncate_and_adjust(seq, MAX_LENGTH)

        assert trunc == seq[start:end]

    @pytest.mark.parametrize("run", range(30))
    def test_total_length_never_exceeds_max(self, run):
        """AAs + boundary tokens must never exceed max_length."""
        seq = _make_sequence(MAX_LENGTH + 20)
        trunc, start, end, add_cls, add_eos = self._truncate_and_adjust(
            seq, MAX_LENGTH
        )

        num_special = int(add_cls) + int(add_eos)
        total = len(trunc) + num_special
        assert total <= MAX_LENGTH


# ── Position ID tests ───────────────────────────────────────────────────


class TestPositionIds:
    def _get_position_ids_and_start(self, sequence, max_length):
        """Run truncation + adjustment and return (position_ids, start_idx)."""
        truncated, start_idx, end_idx, _ = random_truncation(
            sequence, max_length
        )

        add_cls = start_idx == 0
        add_eos = end_idx == len(sequence)
        num_special = int(add_cls) + int(add_eos)
        target_aa_len = max_length - num_special
        if len(truncated) > target_aa_len:
            excess = len(truncated) - target_aa_len
            if add_eos and not add_cls:
                start_idx += excess
            else:
                end_idx = start_idx + target_aa_len

        position_ids = torch.arange(start_idx, start_idx + max_length)
        return position_ids, start_idx

    def test_short_sequence_starts_at_zero(self):
        seq = _make_sequence(5)
        pos, start = self._get_position_ids_and_start(seq, MAX_LENGTH)

        assert start == 0
        assert pos[0].item() == 0
        assert pos[-1].item() == MAX_LENGTH - 1

    def test_position_ids_are_contiguous(self):
        seq = _make_sequence(MAX_LENGTH + 50)
        pos, _ = self._get_position_ids_and_start(seq, MAX_LENGTH)

        diffs = pos[1:] - pos[:-1]
        assert (diffs == 1).all()

    def test_position_ids_start_at_start_idx(self):
        """position_ids[0] must equal the (possibly adjusted) start_idx."""
        seq = _make_sequence(MAX_LENGTH * 3)
        pos, start = self._get_position_ids_and_start(seq, MAX_LENGTH)

        assert pos[0].item() == start

    def test_position_ids_length(self):
        seq = _make_sequence(MAX_LENGTH * 3)
        pos, _ = self._get_position_ids_and_start(seq, MAX_LENGTH)

        assert len(pos) == MAX_LENGTH

    @pytest.mark.parametrize("run", range(20))
    def test_position_ids_consistent_with_truncation(self, run):
        """For every random truncation, position_ids must start at start_idx,
        be contiguous, and have exactly max_length elements."""
        seq = _make_sequence(MAX_LENGTH + 30)
        pos, start = self._get_position_ids_and_start(seq, MAX_LENGTH)

        assert pos[0].item() == start
        assert pos[-1].item() == start + MAX_LENGTH - 1
        assert len(pos) == MAX_LENGTH
        assert (pos[1:] - pos[:-1] == 1).all()

    def test_position_ids_match_full_pipeline(
        self, tokenizer, noise_fn, special_token_ids
    ):
        """Verify position_ids from _pipeline match the expected start_idx."""
        seq = _make_sequence(MAX_LENGTH * 2)

        random.seed(42)
        _, start_idx_direct, _, _ = random_truncation(seq, MAX_LENGTH)
        add_cls = start_idx_direct == 0
        add_eos = (start_idx_direct + MAX_LENGTH) == len(seq)
        num_special = int(add_cls) + int(add_eos)
        if MAX_LENGTH > MAX_LENGTH - num_special:
            excess = MAX_LENGTH - (MAX_LENGTH - num_special)
            if add_eos and not add_cls:
                start_idx_direct += excess

        random.seed(42)
        out = _pipeline(
            seq, tokenizer, noise_fn, special_token_ids, MAX_LENGTH
        )
        pos = out["position_ids"][0]

        assert pos[0].item() == start_idx_direct
        assert pos[-1].item() == start_idx_direct + MAX_LENGTH - 1


# ── DataLoader + Collator tests ─────────────────────────────────────────


class _DummyIterableDataset(IterableDataset):
    """Mimics HFDataset: yields one sample at a time through _pipeline."""

    def __init__(
        self, sequences, tokenizer, noise_fn, special_token_ids, max_length
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.noise_fn = noise_fn
        self.special_token_ids = special_token_ids
        self.max_length = max_length

    def __iter__(self):
        for seq in self.sequences:
            yield _pipeline(
                seq,
                self.tokenizer,
                self.noise_fn,
                self.special_token_ids,
                self.max_length,
            )


class TestDataLoader:
    def _make_loader(
        self, sequences, tokenizer, noise_fn, special_token_ids, batch_size
    ):
        ds = _DummyIterableDataset(
            sequences,
            tokenizer,
            noise_fn,
            special_token_ids,
            MAX_LENGTH,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=AresCollator(),
            num_workers=0,
        )

    @pytest.mark.parametrize("batch_size", [2, 4, 8])
    def test_batch_shapes(
        self, tokenizer, noise_fn, special_token_ids, batch_size
    ):
        sequences = [_make_sequence(10 + i * 5) for i in range(batch_size)]
        loader = self._make_loader(
            sequences, tokenizer, noise_fn, special_token_ids, batch_size
        )

        batch = next(iter(loader))
        for key in ("input_ids", "attention_mask", "labels", "position_ids"):
            assert batch[key].shape == (
                batch_size,
                MAX_LENGTH,
            ), f"{key}: expected ({batch_size}, {MAX_LENGTH}), got {batch[key].shape}"

    @pytest.mark.parametrize("batch_size", [2, 4, 8])
    def test_batch_dtypes(
        self, tokenizer, noise_fn, special_token_ids, batch_size
    ):
        sequences = [_make_sequence(10 + i * 5) for i in range(batch_size)]
        loader = self._make_loader(
            sequences, tokenizer, noise_fn, special_token_ids, batch_size
        )

        batch = next(iter(loader))
        for key in ("input_ids", "attention_mask", "labels", "position_ids"):
            assert batch[key].dtype == torch.long

    def test_mixed_lengths_batch(self, tokenizer, noise_fn, special_token_ids):
        """Batch with very short, exact, and very long sequences."""
        sequences = [
            _make_sequence(1),
            _make_sequence(MAX_LENGTH - 2),
            _make_sequence(MAX_LENGTH),
            _make_sequence(MAX_LENGTH * 3),
        ]
        loader = self._make_loader(
            sequences, tokenizer, noise_fn, special_token_ids, batch_size=4
        )

        batch = next(iter(loader))
        assert batch["input_ids"].shape == (4, MAX_LENGTH)

        for i in range(4):
            mask = batch["attention_mask"][i]
            ids = batch["input_ids"][i]
            labels = batch["labels"][i]
            pos = batch["position_ids"][i]

            num_real = mask.sum().item()
            assert num_real >= 1
            assert num_real <= MAX_LENGTH

            padding_labels = labels[mask == 0]
            assert (padding_labels == -100).all()

            padding_ids = ids[mask == 0]
            assert (padding_ids == tokenizer.pad_token_id).all()

            diffs = pos[1:] - pos[:-1]
            assert (diffs == 1).all()

    def test_multiple_batches(self, tokenizer, noise_fn, special_token_ids):
        """Iterate through multiple batches and validate every one."""
        num_samples = 20
        batch_size = 4
        sequences = [_make_sequence(5 + i * 7) for i in range(num_samples)]
        loader = self._make_loader(
            sequences, tokenizer, noise_fn, special_token_ids, batch_size
        )

        total_samples = 0
        for batch in loader:
            bs = batch["input_ids"].shape[0]
            assert bs <= batch_size
            total_samples += bs

            for key in (
                "input_ids",
                "attention_mask",
                "labels",
                "position_ids",
            ):
                assert batch[key].shape == (bs, MAX_LENGTH)

            for i in range(bs):
                mask = batch["attention_mask"][i]
                labels = batch["labels"][i]
                assert (labels[mask == 0] == -100).all()

                vocab_size = len(tokenizer)
                valid = (labels == -100) | (
                    (labels >= 0) & (labels < vocab_size)
                )
                assert valid.all()

        assert total_samples == num_samples

    def test_last_incomplete_batch(
        self, tokenizer, noise_fn, special_token_ids
    ):
        """When num_samples is not divisible by batch_size, the last batch
        should be smaller but still correct."""
        num_samples = 7
        batch_size = 4
        sequences = [_make_sequence(15) for _ in range(num_samples)]
        loader = self._make_loader(
            sequences, tokenizer, noise_fn, special_token_ids, batch_size
        )

        batches = list(loader)
        assert len(batches) == 2
        assert batches[0]["input_ids"].shape[0] == 4
        assert batches[1]["input_ids"].shape[0] == 3

        for batch in batches:
            bs = batch["input_ids"].shape[0]
            for key in (
                "input_ids",
                "attention_mask",
                "labels",
                "position_ids",
            ):
                assert batch[key].shape == (bs, MAX_LENGTH)
