import torch

from ares.pipelines.sequence_packing_batch_masking import PackedCollator
from ares.preprocessing.utils import MLMProbabilitySampler


class MockTokenizer:
    def __init__(self):
        self.all_special_ids = [0, 1, 2, 4]
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 4
        self._vocab = {chr(i + 65): i + 5 for i in range(20)}
        self._vocab.update(
            {
                "<pad>": self.pad_token_id,
                "<cls>": self.cls_token_id,
                "<eos>": self.eos_token_id,
                "<mask>": self.mask_token_id,
            }
        )

    def get_vocab(self):
        return self._vocab


def _make_collator():
    tokenizer = MockTokenizer()
    sampler = MLMProbabilitySampler(
        mlm_probs=[1.0],
        masking_probs=[1.0],
        mutation_probs=[0.0],
    )
    return PackedCollator(
        tokenizer=tokenizer,
        mlm_probability_sampler=sampler,
    )


def test_packed_collator_masks_batched_non_padding_sequences():
    collator = _make_collator()
    batch = [
        {
            "input_ids": torch.tensor([1, 5, 6, 2], dtype=torch.long),
            "sequence_ids": torch.tensor([0, 0, 0, 0], dtype=torch.long),
            "position_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([1, 7, 8, 2], dtype=torch.long),
            "sequence_ids": torch.tensor([0, 0, 0, 0], dtype=torch.long),
            "position_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
        },
    ]

    output = collator(batch)

    expected_inputs = torch.tensor(
        [
            [1, 4, 4, 2],
            [1, 4, 4, 2],
        ],
        dtype=torch.long,
    )
    expected_labels = torch.tensor(
        [
            [-100, 5, 6, -100],
            [-100, 7, 8, -100],
        ],
        dtype=torch.long,
    )

    assert torch.equal(output["input_ids"], expected_inputs)
    assert torch.equal(output["labels"], expected_labels)
    assert output["sequence_ids"].shape == (2, 4)
    assert output["position_ids"].shape == (2, 4)
    assert output["attention_mask"].shape == (2, 4)


def test_packed_collator_keeps_padding_unmasked_in_batched_input():
    collator = _make_collator()
    batch = [
        {
            "input_ids": torch.tensor([1, 5, 6, 2, 0, 0], dtype=torch.long),
            "sequence_ids": torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long),
            "position_ids": torch.tensor([0, 1, 2, 3, 0, 0], dtype=torch.long),
            "attention_mask": torch.tensor(
                [1, 1, 1, 1, 0, 0], dtype=torch.long
            ),
        },
        {
            "input_ids": torch.tensor([1, 7, 2, 0, 0, 0], dtype=torch.long),
            "sequence_ids": torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long),
            "position_ids": torch.tensor([0, 1, 2, 0, 0, 0], dtype=torch.long),
            "attention_mask": torch.tensor(
                [1, 1, 1, 0, 0, 0], dtype=torch.long
            ),
        },
    ]

    output = collator(batch)

    expected_inputs = torch.tensor(
        [
            [1, 4, 4, 2, 0, 0],
            [1, 4, 2, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    expected_labels = torch.tensor(
        [
            [-100, 5, 6, -100, -100, -100],
            [-100, 7, -100, -100, -100, -100],
        ],
        dtype=torch.long,
    )

    assert torch.equal(output["input_ids"], expected_inputs)
    assert torch.equal(output["labels"], expected_labels)
    assert torch.equal(
        output["attention_mask"],
        torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
    )
