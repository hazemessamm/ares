import random


def _shuffle_and_skip_bins(bin_indices, skip_bins):
    shuffled_bin_indices = list(bin_indices)
    random.shuffle(shuffled_bin_indices)

    if skip_bins <= 0:
        return shuffled_bin_indices
    if skip_bins >= len(shuffled_bin_indices):
        return []
    return shuffled_bin_indices[skip_bins:]


def _truncate_for_packing(sequence, max_length, random_truncation):
    truncated, start_idx, end_idx, _ = random_truncation(sequence, max_length)

    add_cls = start_idx == 0
    add_eos = end_idx == len(sequence)
    num_special = int(add_cls) + int(add_eos)

    target_aa_len = max_length - num_special
    if len(truncated) > target_aa_len:
        excess = len(truncated) - target_aa_len
        if add_eos and not add_cls:
            truncated = truncated[excess:]
        else:
            truncated = truncated[:target_aa_len]

    if add_cls:
        truncated = ["<cls>"] + truncated
    if add_eos:
        truncated = truncated + ["<eos>"]

    return truncated


def test_shuffle_and_skip_uses_shuffled_order():
    bin_indices = list(range(12))
    skip_bins = 4

    random.seed(1234)
    expected = list(bin_indices)
    random.shuffle(expected)

    random.seed(1234)
    actual = _shuffle_and_skip_bins(bin_indices, skip_bins)

    assert actual == expected[skip_bins:]


def test_shuffle_and_skip_handles_large_skip():
    random.seed(7)
    assert _shuffle_and_skip_bins([0, 1, 2], 3) == []


def test_truncate_for_packing_does_not_add_fake_boundaries_to_internal_fragment():
    sequence = list("ABCDEFGHIJ")
    truncated = _truncate_for_packing(
        sequence,
        5,
        random_truncation=lambda seq, max_length: (seq[3:8], 3, 8, True),
    )

    assert truncated == list("DEFGH")
    assert "<cls>" not in truncated
    assert "<eos>" not in truncated


if __name__ == "__main__":
    test_shuffle_and_skip_uses_shuffled_order()
    test_shuffle_and_skip_handles_large_skip()
    test_truncate_for_packing_does_not_add_fake_boundaries_to_internal_fragment()
    print("All tests passed")
