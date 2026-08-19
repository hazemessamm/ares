import torch

from ares.pipelines.metrics import Accuracy
from ares.pipelines.metrics import Perplexity
from ares.pipelines.metrics import Loss


class DataSource:
    def __init__(self):
        self.total_loss = 0.0
        self.total_perplexity = 0.0
        self.total_accuracy = 0.0
        self.loss = Loss()
        self.perplexity = Perplexity()
        self.accuracy = Accuracy()

        self.best_metrics = {
            "perplexity": float("inf"),
            "accuracy": 0.0,
            "loss": float("inf"),
        }

    def reset(self):
        """Reset running accumulators but preserve
        best_metrics across the run."""
        self.perplexity.reset()
        self.accuracy.reset()
        self.loss.reset()

    def get(self, identifier: str):
        return self.best_metrics[identifier]

    def compute_statistics(self):
        self.total_loss = self.loss.compute()
        self.total_perplexity = self.perplexity.compute()
        self.total_accuracy = self.accuracy.compute()

        self.best_metrics["perplexity"] = min(
            self.best_metrics["perplexity"],
            self.total_perplexity,
        )
        self.best_metrics["loss"] = min(
            self.best_metrics["loss"],
            self.total_loss,
        )
        self.best_metrics["accuracy"] = max(
            self.best_metrics["accuracy"],
            self.total_accuracy,
        )
        return {
            "loss": self.total_loss,
            "perplexity": self.total_perplexity,
            "accuracy": self.total_accuracy,
        }


class StateTracker:
    def __init__(
        self,
        sources: list[str] = ["train", "validation"],
        initial_step: int = 0,
    ) -> None:
        self.sources = sources
        self._sources = {source: DataSource() for source in sources}
        self.global_step = initial_step

    def get_source(self, source: str):
        if source not in self._sources:
            raise ValueError(f"Source {source} not found")
        return self._sources[source]

    def update_loss(self, loss: float, labels: torch.Tensor, source: str):
        self._sources[source].loss.update(loss, labels)

    def update_perplexity(
        self, loss: float, labels: torch.Tensor, source: str
    ):
        self._sources[source].perplexity.update(loss, labels)

    def step(self):
        self.global_step += 1

    def update_accuracy(
        self, preds: torch.Tensor, targets: torch.Tensor, source: str
    ):
        """Update accuracy components efficiently without storing tensors."""
        self._sources[source].accuracy.update(preds, targets)

    def update_accuracy_components(
        self,
        num_correct: torch.Tensor,
        num_valid: torch.Tensor,
        source: str,
    ):
        """Update accuracy using precomputed aggregate components."""
        self._sources[source].accuracy.update_from_components(
            num_correct, num_valid
        )

    def compute_statistics(self, source: str):
        return self._sources[source].compute_statistics()

    def reset(self, source: str):
        # Reminder: Global Step is not resettable.
        self._sources[source].reset()
