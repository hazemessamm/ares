"""Training-pipeline components.

Imports are resolved lazily so that pulling in a single submodule does not
require the optional training/data dependencies (``datasets``, ``wandb``,
``torch_xla``, ...) that other submodules need.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ares.pipelines.dataset import HFDataset
    from ares.pipelines.metrics import compute_accuracy

__all__ = ["HFDataset", "compute_accuracy"]

_LAZY = {
    "HFDataset": ("ares.pipelines.dataset", "HFDataset"),
    "compute_accuracy": ("ares.pipelines.metrics", "compute_accuracy"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
