import torch
from torch import nn
from torch.optim import Optimizer
from omegaconf import DictConfig
from ares.optimizers.soap import SOAP
from ares.optimizers.c_adamw import AdamW as CAdamW
from ares.models.model import Ares

try:
    from torch_xla.amp import syncfree
except ImportError:
    syncfree = None


def create_parameter_groups(model: Ares, config: DictConfig) -> list[dict]:
    non_weight_decay_params = []
    weight_decay_params = []

    for name, param in model.named_parameters():
        if "sequence_embeddings" in name:
            non_weight_decay_params.append(param)
        elif "norm" in name:
            non_weight_decay_params.append(param)
        elif "bias" in name:
            non_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)

    param_groups = [
        dict(
            params=non_weight_decay_params,
            weight_decay=0.0,
            lr=float(config.optimizer.lr),
        ),
        dict(
            params=weight_decay_params,
            weight_decay=float(config.optimizer.weight_decay),
            lr=float(config.optimizer.lr),
        ),
    ]
    return param_groups


def create_optimizer(model: Ares, config: DictConfig) -> Optimizer:
    name = config.optimizer.name.lower()
    use_autocast = config.training.get("autocast", True)
    if name != "adamw":
        raise ValueError(
            f"AdamW is the only supported optimizer for XLA for now. Got {name}."
        )

    if use_autocast and syncfree is not None:
        return syncfree.AdamW(
            create_parameter_groups(model, config),
            eps=float(config.optimizer.eps),
        )
    else:
        return torch.optim.AdamW(
            create_parameter_groups(model, config),
            eps=float(config.optimizer.eps),
        )
