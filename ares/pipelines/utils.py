import torch_xla.runtime as xr
import wandb
from omegaconf import OmegaConf
from typing import Callable, Any
from omegaconf import DictConfig
import logging
from torch import nn

logger = logging.getLogger("ares")


def is_global_master():
    # xm.get_ordinal() returns the unique ID (0-63) across the whole Pod.
    # On VM 0, the chips are 0, 1, 2, 3.
    # On VM 1, the chips are 4, 5, 6, 7... and so on.
    return xr.process_index() == 0


def print_num_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total:,} total params, {trainable:,} trainable")


def global_master_process_call(func: Callable, *args, **kwargs) -> Any:
    if is_global_master():
        return func(*args, **kwargs)
    return None


def init_wandb_distributed(
    config: DictConfig,
    resume: bool = False,
    start_step: int = 0,
    resume_wandb: bool = False,
    run_on_master_only: bool = False,
):
    """
    Initialize WANDB for distributed training.

    Args:
        config: Training configuration
        resume: If True, resume the existing WANDB run using a deterministic
            run ID based on run_name and rank. This ensures the same run ID
            is used across training sessions.
    """
    # Every VM gets its own unique name based on its ordinal
    is_master = is_global_master()
    global_rank = xr.process_index()

    # All 16 VMs use the same Group Name
    group_id = config.training.get("run_name", "ares-training")
    project = config.training.get("wandb_project", "ares")

    init_kwargs = {
        "project": project,
        "group": group_id,  # This bundles the 16 runs together
        "name": f"RANK-{global_rank}",  # Individual name for each rank
        "job_type": "worker" if not is_master else "train",
    }

    # Use deterministic run ID for resumption
    # This ensures the same run ID is used across training sessions
    if resume and start_step > 0 and resume_wandb:
        # Generate deterministic run ID from group_id and rank
        # This will be the same across runs, allowing resumption
        import hashlib

        run_id_str = f"{group_id}-rank-{global_rank}"
        # Use hash to create a valid WANDB run ID (must be alphanumeric)
        run_id = hashlib.md5(run_id_str.encode()).hexdigest()[:8]
        init_kwargs["resume"] = "allow"
        init_kwargs["id"] = run_id
        if is_master:
            import logging

            logger = logging.getLogger("ares")
            logger.info(f"Resuming WANDB run with deterministic ID: {run_id}")

    # Only master process logs config to avoid duplication
    if is_master:
        init_kwargs["config"] = OmegaConf.to_container(config, resolve=True)

    if run_on_master_only and not is_master:
        init_kwargs["mode"] = "disabled"
    else:
        init_kwargs["mode"] = "online"

    settings = wandb.Settings(console="off")
    init_kwargs["settings"] = settings

    wandb.init(
        **init_kwargs,
    )


def wandb_log_on_master(payload: dict, step: int):
    """Log to wandb only on the global master process."""
    if is_global_master():
        wandb.log(payload, step=step)
