"""
SPMD Distributed Checkpointing for Multi-VM TPU Training.

This module provides checkpointing utilities compatible with PyTorch XLA SPMD
for distributed training across multiple TPU VMs.

Based on: https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html # noqa

Features:
- Synchronous and asynchronous checkpointing
- Managed checkpoints by step
- Auto-checkpointing on preemption (Cloud TPU)
- FSSpec support for GCS and other cloud storage
- Proper optimizer state restoration with prime_optimizer
"""

from __future__ import annotations

import gc
from typing import Any

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

try:
    import torch.distributed as dist
    import torch.distributed.checkpoint as dist_cp
    import torch_xla.distributed.xla_backend  # Register xla:// init_method # noqa # type: ignore
    import torch_xla.experimental.distributed_checkpoint as xc  # type: ignore
    from torch_xla.experimental.distributed_checkpoint import (
        CheckpointManager as XLACheckpointManager,
    )
    from torch_xla.experimental.distributed_checkpoint import (
        prime_optimizer,
    )  # type: ignore
    import torch_xla.core.xla_model as xm

    XLA_CHECKPOINT_AVAILABLE = True
except ImportError:
    XLA_CHECKPOINT_AVAILABLE = False
    xc = None
    dist_cp = None
    XLACheckpointManager = None
    prime_optimizer = None


import requests
import threading
import logging
from typing import Optional

# Module-level logger
logger = logging.getLogger("ares")


def wait_for_preemption():
    """
    Blocking function that waits for TPU preemption signal.

    Adding ?wait_for_change=true tells the server:
    "Don't answer me until I'm actually about to be preempted."
    This will block/pause here until the signal is sent.

    Returns:
        True if preemption detected, False on error.
    """
    url = (
        "http://metadata.google.internal/computeMetadata/v1/"
        "instance/preempted?wait_for_change=true"
    )
    headers = {"Metadata-Flavor": "Google"}

    try:
        # This will block/pause here until the signal is sent
        response = requests.get(url, headers=headers, timeout=None)
        if response.text == "TRUE":
            return True
    except Exception:
        return False
    return False


class PreemptionMonitor:
    """
    Non-blocking preemption monitor for spot TPUs.

    Runs wait_for_preemption() in a background thread and sets a flag
    when preemption is detected. Can be checked periodically during training.

    Note: While Python has GIL limitations, HTTP I/O operations release
    the GIL during network calls, so the background thread can effectively
    block on the metadata endpoint without impacting training performance.

    Example:
        monitor = PreemptionMonitor()
        monitor.start()

        for step in range(total_steps):
            if monitor.is_preempted():
                checkpoint_manager.save(step)
                break
            # ... training step ...
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._preempted = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.logger = logger or logging.getLogger("retroplm")

    def _monitor_loop(self):
        """Background thread that waits for preemption signal."""
        try:
            preempted = wait_for_preemption()
            if preempted:
                with self._lock:
                    self._preempted = True
                self.logger.warning(
                    "Preemption detected! Saving checkpoint..."
                )
        except Exception as e:
            self.logger.error(f"Error in preemption monitor: {e}")

    def start(self):
        """Start the background preemption monitoring thread."""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="PreemptionMonitor",
            )
            self._thread.start()
            self.logger.info("Started preemption monitor thread")

    def is_preempted(self) -> bool:
        """Check if preemption has been detected (non-blocking)."""
        with self._lock:
            return self._preempted

    def stop(self):
        """Stop the monitoring thread (not typically needed for daemon thread)."""
        # The thread will stop automatically when main process exits
        # since it's a daemon thread
        pass


def init_spmd_process_group():
    """
    Initialize a process group for SPMD distributed checkpointing.

    In SPMD mode, the 'xla' backend is not supported since the compiler
    handles all collectives. Instead, use 'gloo' backend
    with xla:// init_method for automatic master IP discovery on TPUs.

    Reference:
        https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html#process-groups
    """
    if not XLA_CHECKPOINT_AVAILABLE:
        raise ImportError(
            "torch_xla is required for SPMD checkpointing. "
            "Install with: pip install torch_xla"
        )

    if not dist.is_initialized():
        # The xla:// init_method automatically discovers master worker IP,
        # rank, and global world size without requiring environment config
        dist.init_process_group("gloo", init_method="xla://")
        logger.info(
            f"Initialized gloo process group: rank {dist.get_rank()}/{dist.get_world_size()}"  # noqa
        )


def prime_distributed_optimizer(optimizer: Optimizer):
    """Materialize lazy optimizer state before checkpointing in SPMD mode."""
    if not XLA_CHECKPOINT_AVAILABLE or prime_optimizer is None:
        raise ImportError(
            "torch_xla distributed checkpointing is required to prime "
            "optimizer state"
        )

    if optimizer.state:
        logger.info("Optimizer state already primed; skipping prime_optimizer")
        return

    logger.info("Priming optimizer state for SPMD checkpoint compatibility")
    prime_optimizer(optimizer)


class SPMDCheckpointManager:
    """
    High-level checkpoint manager for SPMD distributed training on TPUs.

    Wraps torch_xla's experimental CheckpointManager with additional utilities
    for managing model, optimizer, scheduler, and config state.

    Features:
        - Automatic checkpoint tracking by step
        - Async checkpointing to unblock training
        - Auto-checkpointing on TPU preemption
        - Direct checkpointing to GCS or other fsspec filesystems
        - Proper optimizer state priming for restoration

    Example:
        ```python
        chkpt_mgr = SPMDCheckpointManager(
            checkpoint_dir="gs://my-bucket/checkpoints",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_interval=100,
        )

        # Restore if checkpoints exist
        start_step = chkpt_mgr.restore_latest()

        for step in range(start_step, total_steps):
            # ... training step ...

            # Save checkpoint (async)
            chkpt_mgr.save_async(step)
        ```

    Reference:
        https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html#checkpointmanager
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        save_interval: int = 1,
        max_to_keep: int | None = None,
        async_checkpointing: bool = True,
        chkpt_on_preemption: bool = True,
    ):
        """
        Initialize the SPMD checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints. Supports
            GCS paths (gs://) and other fsspec-compatible filesystems.
            model: The model to checkpoint.
            optimizer: The optimizer to checkpoint.
            scheduler: Optional learning rate scheduler to checkpoint.
            config: Optional Hydra/OmegaConf config to save with checkpoints.
            save_interval: Save a checkpoint every N steps when save/save_async
                          is called. Default is 1 (save every call).
            max_to_keep: Maximum number of checkpoints to retain. None
            keeps all checkpoints.
            async_checkpointing: If True, use async checkpointing (default).
            chkpt_on_preemption: If True, auto-checkpoint on TPU preemption (default). # noqa
            Requires TPU provisioned with Autocheckpointing enabled.
        """
        if not XLA_CHECKPOINT_AVAILABLE:
            raise ImportError(
                "torch_xla is required for SPMD checkpointing. "
                "Install with: pip install torch_xla"
            )

        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.async_checkpointing = async_checkpointing

        # Initialize the XLA CheckpointManager
        # It handles managed checkpoints, async saving, and preemption
        self._xla_chkpt_mgr = XLACheckpointManager(
            path=checkpoint_dir,
            save_interval=save_interval,
            max_to_keep=max_to_keep,
            chkpt_on_preemption=chkpt_on_preemption,
        )

    def _get_state_dict(self) -> dict[str, Any]:
        """
        Get combined state dict for checkpointing.

        Note: With FSDP, model.state_dict() and optimizer.state_dict() may
        involve gathering sharded parameters, which can be expensive and
        potentially block if there are pending operations.
        """
        logger.info(
            "Collecting model state_dict (may be expensive with FSDP)..."
        )
        model_state = self.model.state_dict()
        logger.info(
            "Model state_dict collected, collecting optimizer state_dict..."
        )
        optimizer_state = self.optimizer.state_dict()
        logger.info("Optimizer state_dict collected")

        state_dict = {
            "model": model_state,
            "optimizer": optimizer_state,
        }
        if self.scheduler is not None:
            logger.info("Collecting scheduler state_dict...")
            state_dict["scheduler"] = self.scheduler.state_dict()
        logger.info("All state_dicts collected")
        return state_dict

    def _load_state_dict(self, state_dict: dict[str, Any]):
        """Load state dict into model, optimizer, and scheduler."""
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])

    @property
    def all_steps(self) -> list[int]:
        """Get all tracked checkpoint steps."""
        return self._xla_chkpt_mgr.all_steps()

    def save(self, step: int) -> bool:
        """
        Synchronously save a checkpoint.

        Args:
            step: The current training step.

        Returns:
            True if a checkpoint was saved, False
            otherwise (due to save_interval).
        """
        state_dict = self._get_state_dict()
        result = self._xla_chkpt_mgr.save(step, state_dict)
        del state_dict
        gc.collect()
        return result

    def save_async(self, step: int) -> bool:
        """
        Asynchronously save a checkpoint.

        The checkpoint is dispatched to a background thread after moving
        the sharded state_dict to CPU, unblocking training during the write.

        Args:
            step: The current training step.

        Returns:
            True if a checkpoint was saved, False
            otherwise (due to save_interval).

        Note: Even though this is "async", collecting state_dict can be
        expensive with FSDP as it may need to gather sharded parameters.
        The actual write to disk happens asynchronously, but state_dict
        collection happens synchronously.

        Note: We don't call xm.mark_step() here because train_step already
        calls torch_xla.sync() which ensures all XLA operations are complete.
        Calling xm.mark_step() again can cause deadlocks or freezes.
        """
        logger.info(f"Starting async checkpoint save at step {step}...")
        try:
            # Note: We assume train_step has already called torch_xla.sync()
            # which ensures all XLA operations are complete. We don't call
            # xm.mark_step() here to avoid potential deadlocks.

            # Collecting state_dict can be expensive with FSDP
            # This happens synchronously but is necessary for checkpoint
            # consistency. With FSDP, this may involve gathering sharded
            # parameters which can take time.
            logger.info(
                "Collecting state_dict (may be expensive with FSDP)..."
            )
            state_dict = self._get_state_dict()
            logger.info(
                f"State dict collected for step {step}, "
                f"dispatching to background thread..."
            )

            result = self._xla_chkpt_mgr.save_async(step, state_dict)
            logger.info(
                f"Async checkpoint save dispatched for step {step}. "
                f"Write will complete in background."
            )
            return result
        except Exception as e:
            logger.error(
                f"Error during async checkpoint save at step {step}: {e}",
                exc_info=True,
            )
            raise

    def restore(self, step: int) -> int:
        """
        Restore a checkpoint from a specific step.

        Before restoration, the optimizer is primed to ensure lazy state
        is properly initialized. According to PyTorch/XLA documentation:
        - The model should already be on the XLA device and have the desired
          sharding applied before calling prime_optimizer
        - prime_optimizer runs a fake train step (zero_grad + optimizer.step)
          to initialize optimizer states

        Args:
            step: The step to restore from.

        Returns:
            The restored step number.

        Reference:
            https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html
        """
        # Prime the optimizer before restoration
        # This is required because optimizer states are lazily created
        # prime_optimizer internally does: zero_grad() + optimizer.step()
        # The optimizer must reference XLA tensors (created after FSDP wrapping)
        # Reference: https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html
        prime_optimizer(self.optimizer)

        state_dict = self._get_state_dict()
        self._xla_chkpt_mgr.restore(step, state_dict)
        self._load_state_dict(state_dict)

        return step

    def restore_latest(self) -> int:
        """
        Restore the latest checkpoint if one exists.

        Returns:
            The restored step number, or 0 if no checkpoint exists.
        """
        tracked_steps = self.all_steps
        if not tracked_steps:
            logger.info("No checkpoints found, starting from step 0")
            return 0

        best_step = max(tracked_steps)
        logger.info(f"Restoring from checkpoint at step {best_step}")
        return self.restore(best_step)


def save_checkpoint_sync(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    step: int = 0,
):
    """
    Synchronously save a distributed checkpoint using SPMDSavePlanner.

    This is a lower-level API for direct control over checkpointing.
    For managed checkpoints, use SPMDCheckpointManager instead.

    Args:
        checkpoint_dir: Directory to save the checkpoint.
        model: The model to checkpoint.
        optimizer: The optimizer to checkpoint.
        scheduler: Optional scheduler to checkpoint.
        step: Current training step (saved as metadata).

    Reference:
        https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html
    """
    if not XLA_CHECKPOINT_AVAILABLE:
        raise ImportError("torch_xla is required for SPMD checkpointing")

    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scheduler is not None:
        state_dict["scheduler"] = scheduler.state_dict()

    import logging

    logger = logging.getLogger("retroplm")
    dist_cp.save(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(checkpoint_dir),
        planner=xc.SPMDSavePlanner(),
    )
    logger.info(f"Saved checkpoint to {checkpoint_dir} at step {step}")


def load_checkpoint_sync(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
) -> int:
    """
    Synchronously load a distributed checkpoint using SPMDLoadPlanner.

    The model should already be on the XLA device and have the desired
    sharding applied before calling this function.

    Args:
        checkpoint_dir: Directory containing the checkpoint.
        model: The model to load state into.
        optimizer: The optimizer to load state into.
        scheduler: Optional scheduler to load state into.

    Returns:
        The step number from the checkpoint.

    Reference:
        https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html
    """
    if not XLA_CHECKPOINT_AVAILABLE:
        raise ImportError("torch_xla is required for SPMD checkpointing")

    # Prime optimizer before loading
    prime_optimizer(optimizer)

    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": 0,
    }
    if scheduler is not None:
        state_dict["scheduler"] = scheduler.state_dict()

    dist_cp.load(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
        planner=xc.SPMDLoadPlanner(),
    )

    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    if scheduler is not None and "scheduler" in state_dict:
        scheduler.load_state_dict(state_dict["scheduler"])

    step = state_dict.get("step", 0)
    logger.info(f"Loaded checkpoint from {checkpoint_dir} at step {step}")
    return step
