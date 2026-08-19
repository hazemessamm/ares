import torch
import numpy as np
import random

from omegaconf import DictConfig


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_wandb_run_name(config: DictConfig):
    model_config = getattr(config, config.model.name)
    dataset_name = config.task.name
    checkpoint = model_config.checkpoint
    seed = config.training.seed
    freezed = config.shared_modeling_config.freeze
    pooler = config.shared_modeling_config.pooler
    mlp = config.shared_modeling_config.mlp
    intermediate_dim = config.shared_modeling_config.intermediate_dim
    return f"{dataset_name}-{checkpoint}-seed-{seed}-freeze-{freezed}-pooler-{pooler}-mlp-{mlp}-dim-{intermediate_dim}"