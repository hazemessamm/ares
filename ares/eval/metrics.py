from transformers import EvalPrediction, Trainer
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaseMetrics:
    def get_predictions_and_labels(self, p: EvalPrediction):
        predictions = (
            p.predictions[0]
            if isinstance(p.predictions, tuple)
            else p.predictions  # noqa
        )
        labels = p.label_ids
        if predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
        if labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        return predictions, labels

    def eval_and_save(
        self,
        trainer: Trainer,
        dataset: torch.utils.data.Dataset,
        output_file: str,
    ):
        outputs = trainer.evaluate(dataset)
        with open(output_file, "w") as f:
            yaml.dump(outputs, f, dumper=yaml.SafeDumper)
        return outputs


class RegressionMetrics(BaseMetrics):
    def __call__(self, p: EvalPrediction):
        predictions, labels = self.get_predictions_and_labels(p)
        pearson_corr, _ = pearsonr(predictions, labels)
        spearman_corr, _ = spearmanr(predictions, labels)
        return {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }


def compute_accuracy(predictions, labels):
    # labels could have -100 as ignore index
    is_valid = labels != -100
    num_correct = (predictions == labels) & is_valid
    num_valid = is_valid.sum()
    return num_correct.sum() / num_valid


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class BinaryClassificationMetrics(BaseMetrics):
    """Metrics for binary classification heads that emit raw logits.

    Logits are passed through a sigmoid to obtain probabilities before the
    `threshold` is applied, so the default `threshold=0.5` corresponds to the
    standard probabilistic decision rule.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, p: EvalPrediction):
        predictions, labels = self.get_predictions_and_labels(p)
        labels = labels.astype(int)

        probs = _sigmoid(predictions.astype(np.float32))
        preds = (probs > self.threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "mcc": matthews_corrcoef(labels, preds),
        }

        if len(np.unique(labels)) == 2:
            metrics["auroc"] = roc_auc_score(labels, probs)
        else:
            metrics["auroc"] = float("nan")

        return metrics


class MultiClassClassificationMetrics(BaseMetrics):
    def __call__(self, p: EvalPrediction):
        predictions, labels = self.get_predictions_and_labels(p)
        preds = predictions.argmax(axis=1)
        accuracy = (preds == labels).mean()
        # `matthews_corrcoef` handles the multi-class case natively, which
        # lets downstream configs select MCC as the best-model metric for
        # tasks like 10-way DeepLoc subcellular localization.
        mcc = matthews_corrcoef(labels, preds)
        return {
            "accuracy": accuracy,
            "mcc": mcc,
        }


class TokenClassificationMetrics(BaseMetrics):
    """Token-level accuracy for sequence labeling tasks (e.g. SSP).

    Expects predictions of shape (B, T, C) and labels of shape (B, T) where
    positions corresponding to special/padding/disorder tokens are marked
    with `ignore_index` (-100 by default) and excluded from the metric.
    """

    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index

    def __call__(self, p: EvalPrediction):
        predictions = (
            p.predictions[0]
            if isinstance(p.predictions, tuple)
            else p.predictions
        )
        labels = p.label_ids

        preds = predictions.argmax(axis=-1)
        valid = labels != self.ignore_index
        num_valid = valid.sum()
        if num_valid == 0:
            return {"accuracy": 0.0}
        num_correct = ((preds == labels) & valid).sum()
        return {"accuracy": float(num_correct) / float(num_valid)}