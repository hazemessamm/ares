from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from dataclasses import dataclass
from collections import defaultdict
from ares.tokenization import AresProteinTokenizer
import os
import numpy as np
from typing import Optional

@dataclass
class SingleTargetCollator:
    tokenizer: AresProteinTokenizer
    sequences_column: str
    labels_column: str
    family: Optional[str] = None

    def group_batch(self, batch):
        grouped = defaultdict(list)
        for example in batch:
            for k, v in example.items():
                grouped[k].append(v)
        return grouped

    def __call__(self, batch):
        grouped = self.group_batch(batch)
        encoded = self.tokenizer(
            grouped[self.sequences_column],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        targets = torch.tensor(grouped[self.labels_column], dtype=torch.float32).view(-1, 1)
        special_tokens_mask = encoded["attention_mask"].clone()
        valid_lengths = encoded["attention_mask"].sum(dim=1)
        family = self.family.lower() if self.family is not None else None

        if family in {"ares", "esm", "esm2"}:
            special_tokens_mask[:, 0] = 0
            eos_positions = (valid_lengths - 1).clamp(min=0)
            special_tokens_mask.scatter_(1, eos_positions.unsqueeze(1), 0)
        elif family == "ankh":
            eos_positions = (valid_lengths - 1).clamp(min=0)
            special_tokens_mask.scatter_(1, eos_positions.unsqueeze(1), 0)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "special_tokens_mask": special_tokens_mask,
            "labels": targets,
        }


@dataclass
class MultiClassClassificationCollator:
    tokenizer: AresProteinTokenizer
    sequences_column: str
    labels_column: str
    family: Optional[str] = None
    max_length: Optional[int] = None

    def group_batch(self, batch):
        grouped = defaultdict(list)
        for example in batch:
            for k, v in example.items():
                grouped[k].append(v)
        return grouped

    def __call__(self, batch):
        grouped = self.group_batch(batch)

        encoded = self.tokenizer(
            grouped[self.sequences_column],
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )

        targets = torch.tensor(grouped[self.labels_column], dtype=torch.long)

        special_tokens_mask = encoded["attention_mask"].clone()
        valid_lengths = encoded["attention_mask"].sum(dim=1)
        family = self.family.lower() if self.family is not None else None

        if family in {"ares", "esm", "esm2"}:
            special_tokens_mask[:, 0] = 0
            eos_positions = (valid_lengths - 1).clamp(min=0)
            special_tokens_mask.scatter_(1, eos_positions.unsqueeze(1), 0)
        elif family == "ankh":
            eos_positions = (valid_lengths - 1).clamp(min=0)
            special_tokens_mask.scatter_(1, eos_positions.unsqueeze(1), 0)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "special_tokens_mask": special_tokens_mask,
            "labels": targets,
        }

@dataclass
class TokenClassificationCollator:
    """Collator for per-residue (token-level) classification tasks like SSP.

    Tokenizes a batch of protein sequences and aligns the per-residue label
    list of each example to the produced token sequence. Label entries for
    special tokens (CLS/EOS/sentinel) and padding positions are set to
    `ignore_index` so they are skipped by `cross_entropy(..., ignore_index=...)`.
    """

    tokenizer: AresProteinTokenizer
    sequences_column: str
    labels_column: str
    family: Optional[str] = None
    max_length: Optional[int] = None
    ignore_index: int = -100

    def group_batch(self, batch):
        grouped = defaultdict(list)
        for example in batch:
            for k, v in example.items():
                grouped[k].append(v)
        return grouped

    def _residue_offset(self, family: Optional[str]) -> int:
        # Number of leading special tokens before the first residue token.
        if family in {"ares", "esm", "esm2"}:
            return 1  # leading CLS/BOS
        if family == "ankh":
            # Ankh's sentencepiece tokenizer emits a leading <unk> sentinel
            # before the first residue.
            return 1
        return 0

    def __call__(self, batch):
        grouped = self.group_batch(batch)
        sequences = grouped[self.sequences_column]
        per_seq_labels = grouped[self.labels_column]

        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )

        family = self.family.lower() if self.family is not None else None
        left_offset = self._residue_offset(family)

        batch_size, seq_len = encoded["input_ids"].shape
        labels_tensor = torch.full(
            (batch_size, seq_len), self.ignore_index, dtype=torch.long
        )
        valid_lengths = encoded["attention_mask"].sum(dim=1)

        for i, lbls in enumerate(per_seq_labels):
            # Each tokenized sequence ends with a single EOS token, so the
            # number of residue tokens is valid_length - left_offset - 1.
            n_residues = int(valid_lengths[i].item()) - left_offset - 1
            n_residues = max(0, min(n_residues, len(lbls)))
            if n_residues == 0:
                continue
            labels_tensor[i, left_offset : left_offset + n_residues] = torch.tensor(
                lbls[:n_residues], dtype=torch.long
            )

        special_tokens_mask = encoded["attention_mask"].clone()
        if family in {"ares", "esm", "esm2"}:
            special_tokens_mask[:, 0] = 0
            eos_positions = (valid_lengths - 1).clamp(min=0)
            special_tokens_mask.scatter_(1, eos_positions.unsqueeze(1), 0)
        elif family == "ankh":
            eos_positions = (valid_lengths - 1).clamp(min=0)
            special_tokens_mask.scatter_(1, eos_positions.unsqueeze(1), 0)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "special_tokens_mask": special_tokens_mask,
            "labels": labels_tensor,
        }


@dataclass
class EmbeddingsCollator:
    """Batch precomputed embeddings for `RegressionEmbeddingsHead` training."""

    def __call__(self, batch):
        embeddings = torch.stack(
            [torch.as_tensor(ex["embeddings"], dtype=torch.float32) for ex in batch],
            dim=0,
        )
        labels = torch.tensor(
            [ex["labels"] for ex in batch],
            dtype=torch.float32,
        ).view(-1, 1)
        return {"embeddings": embeddings, "labels": labels}


@dataclass
class MultiClassEmbeddingsCollator:
    """Batch precomputed embeddings for `MultiClassEmbeddingsHead` training."""

    def __call__(self, batch):
        embeddings = torch.stack(
            [torch.as_tensor(ex["embeddings"], dtype=torch.float32) for ex in batch],
            dim=0,
        )
        labels = torch.tensor(
            [ex["labels"] for ex in batch],
            dtype=torch.long,
        )
        return {"embeddings": embeddings, "labels": labels}


class Fluorescence(Dataset):
    dataset_repo = "hazemessam/fluorescence"
    split_files = {
        "train": "fluorescence_train.json",
        "validation": "fluorescence_valid.json",
        "test": "fluorescence_test.json",
    }
    sequences_column = "primary"
    labels_column = "log_fluorescence"

    def __init__(self, split: str):
        self.split = split
        if split not in Fluorescence.split_files:
            raise ValueError(
                f"Unsupported split {split!r}. Expected one of "
                f"{sorted(Fluorescence.split_files)}."
            )

        # local_dir = get_local_dataset_dir()
        # split_file = local_dir / SPLIT_FILES[split]
        dataset = load_dataset(
            Fluorescence.dataset_repo,
            data_files={split: Fluorescence.split_files[split]},
            streaming=False,
        )
        self.dataset = dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        return {
            self.sequences_column: example[self.sequences_column],
            self.labels_column: float(example[self.labels_column][0]),
        }


class FluorescenceEmbeddings(Dataset):
    def __init__(self, path, split: str):
        self.path = path
        self.split = split
        self.dataset = Fluorescence(split)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        embedding = np.load(os.path.join(self.path, "{}.npy".format(idx)))
        example = self.dataset[idx]
        return {"embeddings": embedding, "labels": example[self.dataset.labels_column]}


class Stability(Dataset):
    dataset_repo = "hazemessam/stability"
    split_files = {
        "train": "stability_train.json",
        "validation": "stability_valid.json",
        "test": "stability_test.json",
    }
    sequences_column = "primary"
    labels_column = "stability_score"

    def __init__(self, split: str):
        self.split = split
        if split not in Stability.split_files:
            raise ValueError(
                f"Unsupported split {split!r}. Expected one of "
                f"{sorted(Stability.split_files)}."
            )

        dataset = load_dataset(
            Stability.dataset_repo,
            data_files={split: Stability.split_files[split]},
            streaming=False,
        )
        self.dataset = dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return {
            self.sequences_column: example[self.sequences_column],
            self.labels_column: float(example[self.labels_column][0]),
        }


class Solubility(Dataset):
    dataset_repo = "hazemessam/solubility"
    split_files = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }
    sequences_column = "sequences"
    labels_column = "labels"

    def __init__(self, split: str):
        self.split = split
        if split not in Solubility.split_files:
            raise ValueError(
                f"Unsupported split {split!r}. Expected one of "
                f"{sorted(Solubility.split_files)}."
            )

        dataset = load_dataset(
            Solubility.dataset_repo,
            data_files={split: Solubility.split_files[split]},
            streaming=False,
        )
        self.dataset = dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return {
            self.sequences_column: example[self.sequences_column],
            self.labels_column: float(example[self.labels_column]),
        }


class SolubilityEmbeddings(Dataset):
    def __init__(self, path, split: str):
        self.path = path
        self.split = split
        self.dataset = Solubility(split)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        embedding = np.load(os.path.join(self.path, "{}.npy".format(idx)))
        example = self.dataset[idx]
        return {"embeddings": embedding, "labels": example["labels"]}


class RemoteHomology(Dataset):
    dataset_repo = "hazemessam/remote-homology"
    split_files = {
        "train": "remote_homology_train.json",
        "test": "remote_homology_test_fold_holdout.json",
        "validation": "remote_homology_valid.json",
    }
    sequences_column = "primary"
    labels_column = "fold_label"
    num_classes = 1195

    def __init__(self, split: str):
        self.split = split
        if split not in RemoteHomology.split_files:
            raise ValueError(
                f"Unsupported split {split!r}. Expected one of "
                f"{sorted(RemoteHomology.split_files)}."
            )

        dataset = load_dataset(
            RemoteHomology.dataset_repo,
            data_files={split: RemoteHomology.split_files[split]},
            streaming=False,
        )
        self.dataset = dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return {
            self.sequences_column: example[self.sequences_column],
            self.labels_column: example[self.labels_column],
        }


class RemoteHomologyEmbeddings(Dataset):
    def __init__(self, path, split: str):
        self.path = path
        self.split = split
        self.dataset = RemoteHomology(split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        embedding = np.load(os.path.join(self.path, "{}.npy".format(idx)))
        example = self.dataset[idx]
        return {"embeddings": embedding, "labels": example[self.dataset.labels_column]}


class GB1(Dataset):
    dataset_repo = "hazemessam/gb1"
    split_files = {
        "train": "two_vs_rest.csv",
        "validation": "two_vs_rest.csv",
        "test": "two_vs_rest.csv",
    }
    sequences_column = "sequence"
    labels_column = "label"
    num_classes = 1

    def __init__(self, split: str):
        self.split = split
        if split not in GB1.split_files:
            raise ValueError(
                f"Unsupported split {split!r}. Expected one of "
                f"{sorted(GB1.split_files)}."
            )

        dataset = load_dataset(
            GB1.dataset_repo,
            data_files=GB1.split_files[split],
            streaming=False,
        )
        df = dataset["train"].to_pandas()

        if split == "train":
            train = df[(df["split"] == "train") & (df["validation"] == False)]
            self.dataset = train
        elif split == "test":
            test = df[df["split"] == "test"]
            self.dataset = test
        elif split == "validation":
            valid = df[(df["split"] == "train") & (df["validation"] == True)]
            self.dataset = valid
        else:
            raise ValueError(f"Unsupported split {split!r}. Expected one of "
                              f"train, test, validation.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset.iloc[idx, :]
        return {
            self.sequences_column: example[self.sequences_column],
            self.labels_column: example[self.labels_column],
        }


class GB1Embeddings(Dataset):
    def __init__(self, path, split: str):
        self.path = path
        self.split = split
        self.dataset = GB1(split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        embedding = np.load(os.path.join(self.path, "{}.npy".format(idx)))
        example = self.dataset[idx]
        return {"embeddings": embedding, "labels": example[self.dataset.labels_column]}


class SSP(Dataset):
    dataset_repo = "hazemessam/secondary_structure"
    split_files = {
        "train": "secondary_structure_train.json",
        "validation": "secondary_structure_valid.json",
        "casp12": "secondary_structure_casp12.json",
        "cb513": "secondary_structure_cb513.json",
        "ts115": "secondary_structure_ts115.json",
        "casp14": "secondary_structure_casp14.json",
    }
    sequences_column = "primary"
    ss3_column = "ss3"
    ss8_column = "ss8"
    disorder_column = "disorder"
    labels_column = "labels"
    
    def __init__(self, split: str, num_states: int = 3):
        self.split = split
        self.num_states = num_states
        if num_states not in (3, 8):
            raise ValueError(
                f"Unsupported number of states {num_states!r}. Expected 3 or 8."
            )
        self.num_classes = num_states
        if split not in SSP.split_files:
            raise ValueError(
                f"Unsupported split {split!r}. Expected one of "
                f"{sorted(SSP.split_files)}."
            )

        dataset = load_dataset(
            SSP.dataset_repo,
            data_files={split: SSP.split_files[split]},
            streaming=False,
        )
        self.dataset = dataset[split]

    def __len__(self):
        return len(self.dataset)

    def _mask_disorder(self, labels, mask):
        assert len(labels) == len(mask), "Length of labels and mask must be the same"
        return [-100 if m == 0 else l for l, m in zip(labels, mask)]

    def __getitem__(self, idx):
        example = self.dataset[idx]
        if self.num_states == 3:
            ss3 = example[self.ss3_column]
            mask = example[self.disorder_column]
            labels = self._mask_disorder(ss3, mask)
        elif self.num_states == 8:
            ss8 = example[self.ss8_column]
            mask = example[self.disorder_column]
            labels = self._mask_disorder(ss8, mask)
        else:
            raise ValueError(f"Unsupported number of states {self.num_states!r}. Expected one of 3, 8.")
        return {
            self.sequences_column: example[self.sequences_column],
            self.labels_column: labels,
        }


class SubcellularLocalization(Dataset):
    # 10-way DeepLoc localization built from the PEER LMDB tarball
    # (see evaluation/data/prepare_subcellular_localization.py) and
    # mirrored on the Hub as plain CSVs.
    dataset_repo = "hazemessam/subcellular-localization"
    split_files = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }
    sequences_column = "primary"
    labels_column = "localization"
    num_classes = 10

    def __init__(self, split: str):
        self.split = split
        if split not in SubcellularLocalization.split_files:
            raise ValueError(
                f"Unsupported split {split!r}. Expected one of "
                f"{sorted(SubcellularLocalization.split_files)}."
            )
        dataset = load_dataset(
            SubcellularLocalization.dataset_repo,
            data_files={split: SubcellularLocalization.split_files[split]},
            streaming=False,
        )
        self.dataset = dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return {
            self.sequences_column: example[self.sequences_column],
            self.labels_column: int(example[self.labels_column]),
        }


SUPPORTED_DATASETS = {
    "fluorescence": Fluorescence,
    "stability": Stability,
    "solubility": Solubility,
    "remote_homology": RemoteHomology,
    "gb1": GB1,
    "subcellular_localization": SubcellularLocalization,
}


def get(identifier: str, split: str):
    if identifier not in SUPPORTED_DATASETS:
        raise ValueError(f"Invalid dataset identifier: {identifier}")
    dataset = SUPPORTED_DATASETS[identifier](split)
    return dataset
