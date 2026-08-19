"""
XLA/TPU Training Script for the Ares Protein Language Model.

Distributed training using PyTorch XLA with SPMD/FSDPv2 on TPU pods.

Usage:
    python training.py                             # uses config/config.yaml
    python training.py training.seed=123           # override via Hydra CLI
"""

# flake8: noqa: E402

import contextlib
import functools
import logging
import os
import random

import fsspec

fsspec.config.conf["gcs"] = {"block_size": 16 * 1024 * 1024}

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch_xla  # type: ignore
import torch_xla.core.xla_model as xm  # type: ignore
import torch_xla.distributed.spmd as xs  # type: ignore
import torch_xla.distributed.parallel_loader as pl  # type: ignore
from torch_xla import runtime as xr  # type: ignore
from torch_xla.amp import autocast, syncfree  # type: ignore
from torch_xla.distributed.fsdp import checkpoint_module, wrap  # type: ignore
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (  # type: ignore
    SpmdFullyShardedDataParallel as FSDPv2,
)
from ares.pipelines.metrics import compute_accuracy_components
from torch.utils.data import IterableDataset

import wandb
from typing import Iterable, Optional
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from ares.models.config import AresConfig
from ares.models.model import Ares
from ares.models.encoder import EncoderLayer
from ares.optimizers.init import create_optimizer
from ares.pipelines.dataset import AresCollator, HFDataset
from ares.pipelines.tracker import StateTracker
from ares.pipelines.checkpoint import SPMDCheckpointManager
from ares.pipelines.checkpoint import init_spmd_process_group
from ares.pipelines import xla_sharding
from ares.pipelines.utils import (
    init_wandb_distributed,
    global_master_process_call,
)
from ares.preprocessing import MLMProbabilitySampler, SequenceCorruptor
from ares.preprocessing import (
    LinearScheduler,
    EMAScheduler,
    Scheduler,
    StagedLinearScheduler,
)
from ares.tokenization import AresProteinTokenizer
from ares.pipelines.logging import setup_logging
from ares.pipelines.seed import set_seed
from ares.pipelines.utils import print_num_params, wandb_log_on_master
from ares.pipelines.sequence_packing import (
    PackedCollator as PackedEvalCollator,
)
from ares.pipelines.sequence_packing_batch_masking import (
    PackedCollator as PackedBatchMaskingCollator,
    PackedUniRef50Dataset,
)
from ares.pipelines.checkpoint import prime_distributed_optimizer

logger = logging.getLogger("ares")


# ── Factory helpers ───────────────────────────────────────────────────────────
def create_model(config: DictConfig, vocab_size: int) -> Ares:
    model_cfg = AresConfig(
        embed_dim=config.model.embed_dim,
        vocab_size=vocab_size,
        num_heads=config.model.num_heads,
        num_kv_heads=config.model.num_kv_heads,
        num_layers=config.model.num_layers,
        ff_dim=config.model.ff_dim,
        activation=config.model.activation,
        gated=config.model.gated,
        attn_dropout=config.model.attn_dropout,
        bias=config.model.bias,
        attn_capping_value=config.model.attn_capping_value,
        logits_capping_value=config.model.logits_capping_value,
        norm_type=config.model.norm_type,
        ff_norm_type=config.model.ff_norm_type,
        qk_norm=config.model.qk_norm,
        rope_frequency=config.model.rope_frequency,
        rope_scale=config.model.rope_scale,
        moe_type=config.model.moe_type,
        moe_after_num_layers=config.model.moe_after_num_layers,
        num_experts=config.model.num_experts,
        expert_capacity_factor=config.model.expert_capacity_factor,
        moe_noise_level=config.model.moe_noise_level,
        moe_normalize=config.model.moe_normalize,
        moe_num_slots=config.model.moe_num_slots,
        pad_token_id=0,
        moe_interleaved=config.model.moe_interleaved,
    )
    model = Ares(model_cfg)

    print_num_params(model)
    return model


def create_mlm_scheduler(config: DictConfig, total_steps: int):
    """Build the MLM probability scheduler from config, or None."""
    sched_cfg = config.masking.get("scheduler", None)
    if sched_cfg is None:
        return None

    sched_type = sched_cfg.get("type", None)
    if sched_type is None:
        return None

    initial = list(sched_cfg.initial_weights)
    final = list(sched_cfg.final_weights)

    if sched_type == "staged_linear":
        warmup_ratio = float(sched_cfg.warmup_ratio)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = StagedLinearScheduler(
            difficulties=config.masking.mlm_probability,
            warmup_steps=warmup_steps,
        )
        logger.info(
            f"MLM scheduler: staged_linear, ramp to uniform over "
            f"{warmup_steps} steps ({warmup_ratio:.1%} of training)"
        )
        return scheduler

    if sched_type == "linear":
        warmup_ratio = float(sched_cfg.warmup_ratio)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = LinearScheduler(
            initial_weights=initial,
            final_weights=final,
            warmup_steps=warmup_steps,
        )
        logger.info(
            f"MLM scheduler: linear, ramp to uniform over "
            f"{warmup_steps} steps ({warmup_ratio:.1%} of training)"
        )
        return scheduler

    if sched_type == "ema":
        scheduler = EMAScheduler(
            initial_weights=initial,
            final_weights=final,
            beta=float(sched_cfg.beta),
            multiplier=float(sched_cfg.multiplier),
        )
        logger.info(
            f"MLM scheduler: ema, beta={sched_cfg.beta}, "
            f"multiplier={sched_cfg.multiplier}"
        )
        return scheduler

    logger.warning(f"Unknown MLM scheduler type '{sched_type}', disabled")
    return None


def load_dataloader(
    config: DictConfig,
    dataset,
    collator,
    num_devices: int,
    training: bool = True,
    num_workers: Optional[int] = 0,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader producing global batches for SPMD sharding.

    MpDeviceLoader shards each batch across devices via input_sharding,
    so the DataLoader must produce per_device_batch * num_devices samples.
    """
    per_device_batch = (
        config.training.per_device_train_batch_size
        if training
        else config.training.per_device_eval_batch_size
    )

    if not training:
        num_workers = 0
        prefetch_factor = None

    global_batch_size = per_device_batch * num_devices
    logger.info(
        f"DataLoader: per_device_batch={per_device_batch}, "
        f"num_devices={num_devices}, "
        f"global_batch_size={global_batch_size}, "
        f"training={training}"
    )

    return DataLoader(
        dataset,
        batch_size=global_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
    )


def load_dataset(
    config, tokenizer, noise_fn, split, skip_examples=0, training=True
):
    if not training:
        return HFDataset(
            repo_id=config.dataset.repo_id,
            column_name=config.dataset.column_name,
            max_length=config.truncation.max_length,
            noise_fn=noise_fn,
            tokenizer=tokenizer,
            split=split,
            seed=config.dataset.seed,
        )

    if config.dataset.packed:
        return PackedUniRef50Dataset(
            repo_id=config.dataset.repo_id,
            column_name=config.dataset.column_name,
            max_length=config.truncation.max_length,
            noise_fn=noise_fn,
            tokenizer=tokenizer,
            split=split,
            seed=config.dataset.seed,
            skip_examples=skip_examples,
        )
    else:
        return HFDataset(
            repo_id=config.dataset.repo_id,
            column_name=config.dataset.column_name,
            max_length=config.truncation.max_length,
            noise_fn=noise_fn,
            tokenizer=tokenizer,
            split=split,
            seed=config.dataset.seed,
            skip_examples=skip_examples,
        )


def distributed_rendezvous(enabled: bool, tag: str):
    """Synchronize all ranks at critical startup boundaries."""
    if not enabled:
        return

    rank = xr.process_index()
    logger.info(f"[rank {rank}] Entering rendezvous: {tag}")
    xm.rendezvous(tag)
    logger.info(f"[rank {rank}] Passed rendezvous: {tag}")


# ── Train / eval steps ────────────────────────────────────────────────────────


def _amp_ctx(use_autocast: bool):
    if use_autocast:
        return autocast(xm.xla_device())
    return contextlib.nullcontext()


def train_step(
    model: nn.Module,
    batch: dict,
    gradient_accumulation_steps: int = 1,
    use_autocast: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward + backward micro-step.

    Loss is scaled by 1/gradient_accumulation_steps for correct
    gradient accumulation. The unscaled loss is returned for logging.
    Batch is already on XLA device via MpDeviceLoader.
    """
    with _amp_ctx(use_autocast):
        loss, logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            position_ids=batch.get("position_ids", None),
            sequence_ids=batch.get("sequence_ids", None),
            return_dict=False,
        )

    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    num_correct, num_valid = compute_accuracy_components(
        logits.detach(), batch["labels"]
    )
    return loss.detach(), num_correct.detach(), num_valid.detach()


@torch.no_grad()
def validation_step(
    model: nn.Module,
    batch: dict,
    use_autocast: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single validation forward pass. Batch is already on XLA device."""
    with _amp_ctx(use_autocast):
        loss, logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            position_ids=batch.get("position_ids", None),
            sequence_ids=batch.get("sequence_ids", None),
            return_dict=False,
        )
    num_correct, num_valid = compute_accuracy_components(
        logits.detach(), batch["labels"]
    )
    torch_xla.sync()
    return loss, num_correct, num_valid


# ── Epoch loops ───────────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    train_dataloader: Iterable,
    valid_dataset: IterableDataset,
    valid_dataloader: Iterable,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    config: DictConfig,
    state_tracker: StateTracker,
    epoch: int,
    checkpoint_manager: SPMDCheckpointManager,
    mlm_scheduler: Optional[Scheduler] = None,
    start_step: int = 0,
):
    """Train for one epoch with optional gradient accumulation."""
    state_tracker.reset(source="train")
    model.train()

    grad_accum_steps = config.training.get("gradient_accumulation_steps", 1)

    pbar = tqdm(
        train_dataloader,
        miniters=config.training.logging_steps,
        desc=f"Training - Epoch {epoch}",
        initial=start_step,
    )

    if start_step > 0:
        logger.info(
            f"Advancing counters to step {start_step} "
            f"(dataset handles example skipping)"
        )
        for _ in tqdm(
            range(start_step), desc="Advancing counters", leave=False
        ):
            state_tracker.step()
            if mlm_scheduler is not None:
                for _ in range(grad_accum_steps):
                    mlm_scheduler.sample()
                mlm_scheduler.step()
        logger.info(
            f"Counters advanced: global_step={state_tracker.global_step}"
        )

    optimizer.zero_grad(set_to_none=True)
    torch_xla.sync()
    micro_step = 0
    batch_loss = None
    total_gradient_norm = None

    for batch in pbar:
        # ── Forward + backward ──
        batch_loss, num_correct, num_valid = train_step(
            model=model,
            batch=batch,
            gradient_accumulation_steps=grad_accum_steps,
            use_autocast=config.training.get("autocast", False),
        )
        micro_step += 1

        state_tracker.update_accuracy_components(
            num_correct, num_valid, source="train"
        )
        state_tracker.update_loss(batch_loss, batch["labels"], source="train")
        state_tracker.update_perplexity(
            batch_loss, batch["labels"], source="train"
        )

        # ── Optimizer step ──
        if micro_step % grad_accum_steps == 0:
            total_gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config.training.max_grad_norm),
            )
            optimizer.step()
            lr_scheduler.step()
            if mlm_scheduler is not None:
                mlm_scheduler.step()
            torch_xla.sync()
            optimizer.zero_grad(set_to_none=True)

            state_tracker.step()

            # ── Logging ──
            if state_tracker.global_step % config.training.logging_steps == 0:
                stats = state_tracker.compute_statistics(source="train")
                lr = optimizer.param_groups[0]["lr"]
                grad_norm_scalar = (
                    total_gradient_norm.item()
                    if isinstance(total_gradient_norm, torch.Tensor)
                    else total_gradient_norm
                )

                pbar.set_postfix(
                    loss=f"{stats['loss']:.4f}",
                    ppl=f"{stats['perplexity']:.2f}",
                    acc=f"{stats['accuracy']:.4f}",
                    grad_norm=f"{grad_norm_scalar:.2f}",
                    lr=f"{lr:.2e}",
                )
                log_dict = {
                    "train/loss": stats["loss"],
                    "train/perplexity": stats["perplexity"],
                    "train/accuracy": stats["accuracy"],
                    "train/lr": lr,
                    "train/gradient_norm": grad_norm_scalar,
                    "train/global_step": state_tracker.global_step,
                    "train/batch_loss": batch_loss.item(),
                }
                if mlm_scheduler is not None:
                    w = mlm_scheduler.sample()
                    for i, wi in enumerate(w):
                        log_dict[f"train/mlm_weight_{i}"] = wi
                wandb_log_on_master(log_dict, step=state_tracker.global_step)

            # ── Step-based checkpoint ──
            if (
                config.training.save_strategy == "steps"
                and state_tracker.global_step % config.training.save_steps == 0
            ):
                logger.info(
                    f"Checkpoint save triggered at step "
                    f"{state_tracker.global_step}"
                )
                saved = checkpoint_manager.save(step=state_tracker.global_step)
                if saved:
                    logger.info(
                        f"Checkpoint saved at step "
                        f"{state_tracker.global_step}"
                    )

            # ── Mid-epoch evaluation ──
            if (
                config.training.eval_every > 0
                and state_tracker.global_step % config.training.eval_every == 0
            ):
                for with_noise in [False, True]:
                    validation_epoch(
                        model=model,
                        valid_dataset=valid_dataset,
                        valid_dataloader=valid_dataloader,
                        config=config,
                        state_tracker=state_tracker,
                        epoch=epoch,
                        checkpoint_manager=checkpoint_manager,
                        debug_mode=False,
                        with_noise=with_noise,
                    )
                    logger.info(
                        f"Validation completed at step "
                        f"{state_tracker.global_step} with noise {with_noise}"
                    )


@torch.no_grad()
def validation_epoch(
    model: nn.Module,
    valid_dataset: IterableDataset,
    valid_dataloader: Iterable,
    config: DictConfig,
    state_tracker: StateTracker,
    epoch: int,
    checkpoint_manager: SPMDCheckpointManager,
    debug_mode: bool = False,
    log_step: Optional[int] = None,
    with_noise: bool = False,
):
    """Run one full validation pass."""
    is_distributed = config.training.get("distributed", False)
    debug_rendezvous_every = int(
        config.training.get("debug_validation_rendezvous_every", 0) or 0
    )
    if not with_noise:
        model.eval()

    for difficulty_idx in range(len(config.masking.mlm_probability)):
        state_tracker.reset(source="validation")
        valid_dataset.noise_fn.mlm_probability_sampler.eval(idx=difficulty_idx)
        distributed_rendezvous(
            is_distributed,
            f"val_start_difficulty_{difficulty_idx}_noise_{int(with_noise)}",
        )
        pbar = tqdm(valid_dataloader, desc=f"Validation - Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar, start=1):
            batch_loss, num_correct, num_valid = validation_step(
                model=model,
                batch=batch,
                use_autocast=config.training.get("autocast", False),
            )
            state_tracker.update_loss(
                batch_loss, batch["labels"], source="validation"
            )
            state_tracker.update_perplexity(
                batch_loss, batch["labels"], source="validation"
            )
            state_tracker.update_accuracy_components(
                num_correct, num_valid, source="validation"
            )

            if (
                debug_rendezvous_every > 0
                and batch_idx % debug_rendezvous_every == 0
            ):
                distributed_rendezvous(
                    is_distributed,
                    "val_progress_"
                    f"difficulty_{difficulty_idx}_"
                    f"noise_{int(with_noise)}_"
                    f"batch_{batch_idx}",
                )

        stats = state_tracker.compute_statistics(source="validation")
        pbar.set_postfix(
            loss=f"{stats['loss']:.4f}",
            ppl=f"{stats['perplexity']:.2f}",
            acc=f"{stats['accuracy']:.4f}",
        )

        step_to_log = (
            log_step if log_step is not None else state_tracker.global_step
        )
        source = state_tracker.get_source("validation")
        logger.info(
            f"Validation loss: {source.total_loss}, perplexity: {source.total_perplexity}, accuracy: {source.total_accuracy}"
        )
        wandb_log_on_master(
            {
                f"val@{difficulty_idx}/loss_noise_{with_noise}": source.total_loss,
                f"val@{difficulty_idx}/perplexity_noise_{with_noise}": source.total_perplexity,
                f"val@{difficulty_idx}/accuracy_noise_{with_noise}": source.total_accuracy,
                f"val@{difficulty_idx}/epoch_noise_{with_noise}": epoch,
            },
            step=step_to_log,
        )
        distributed_rendezvous(
            is_distributed,
            f"val_end_difficulty_{difficulty_idx}_noise_{int(with_noise)}",
        )

    if config.training.save_strategy == "epoch" or debug_mode:
        checkpoint_manager.save(step=state_tracker.global_step)
        logger.info(
            f"Checkpoint saved at epoch {epoch} "
            f"(step {state_tracker.global_step})"
        )
    valid_dataset.noise_fn.mlm_probability_sampler.train()
    model.train()


# ── Top-level train loop ─────────────────────────────────────────────────────
def train(
    model: nn.Module,
    train_dataset: IterableDataset,
    train_collator,
    train_dataloader: Iterable,
    valid_dataset: IterableDataset,
    valid_dataloader: Iterable,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    config: DictConfig,
    checkpoint_manager: SPMDCheckpointManager,
    mlm_scheduler,
    num_steps_per_epoch: int,
    start_step: int = 0,
):
    """Main training loop across epochs."""
    state_tracker = StateTracker(
        sources=["train", "validation"],
        initial_step=0,
    )

    if config.training.get("debug_mode", False):
        logger.info("Running debug validation...")
        for with_noise in [False, True]:
            validation_epoch(
                model=model,
                valid_dataset=valid_dataset,
                valid_dataloader=valid_dataloader,
                config=config,
                state_tracker=state_tracker,
                epoch=0,
                checkpoint_manager=checkpoint_manager,
                debug_mode=True,
                with_noise=with_noise,
            )

    start_epoch = (
        (start_step // num_steps_per_epoch) + 1
        if start_step > 0 and num_steps_per_epoch > 0
        else 1
    )

    if start_step > 0:
        logger.info(f"Resuming from step {start_step} (epoch ~{start_epoch})")
        if (
            config.training.get("evaluate_after_resume", False)
            and config.training.eval_every > 0
            and start_step % config.training.eval_every == 0
        ):
            logger.info("Running evaluation after resumption...")
            for with_noise in [False, True]:
                validation_epoch(
                    model=model,
                    valid_dataset=valid_dataset,
                    valid_dataloader=valid_dataloader,
                    config=config,
                    state_tracker=state_tracker,
                    epoch=start_epoch - 1 if start_epoch > 1 else 0,
                    checkpoint_manager=checkpoint_manager,
                    debug_mode=False,
                    log_step=start_step,
                    with_noise=with_noise,
                )
                logger.info(
                    f"Validation completed after resumption with noise {with_noise}"
                )

    for epoch in range(start_epoch, config.training.epochs + 1):
        if config.dataset.packed and train_dataset is not None:
            train_dataset.epoch_index = epoch - 1
        if config.dataset.packed and train_collator is not None:
            steps_into_epoch = (
                start_step % num_steps_per_epoch
                if epoch == start_epoch and start_step > 0
                else 0
            )
            grad_accum_steps = config.training.get(
                "gradient_accumulation_steps", 1
            )
            train_collator.set_epoch(
                epoch_index=epoch - 1,
                batch_offset=steps_into_epoch * grad_accum_steps,
            )
        epoch_start_step = start_step if epoch == start_epoch else 0
        train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataset=valid_dataset,
            valid_dataloader=valid_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            state_tracker=state_tracker,
            epoch=epoch,
            checkpoint_manager=checkpoint_manager,
            mlm_scheduler=mlm_scheduler,
            start_step=epoch_start_step,
        )

        if epoch == start_epoch and train_dataset is not None:
            train_dataset.skip_examples = 0

        for with_noise in [False, True]:
            validation_epoch(
                model=model,
                valid_dataset=valid_dataset,
                valid_dataloader=valid_dataloader,
                config=config,
                state_tracker=state_tracker,
                epoch=epoch,
                checkpoint_manager=checkpoint_manager,
                debug_mode=False,
                with_noise=with_noise,
            )
            logger.info(
                f"Validation completed at epoch {epoch} with noise {with_noise}"
            )


def initialize_gradient_checkpointing(config, model: Ares):
    if config.model.get("gradient_checkpointing", False):
        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_module(layer)
        logger.info(
            f"XLA gradient checkpointing enabled: "
            f"{len(model.layers)} layers"
        )
    else:
        logger.info("XLA gradient checkpointing disabled")


# ── Entrypoint ────────────────────────────────────────────────────────────────
# Select a config variant with:  python train_expert_choice.py --config-name=<name>
@hydra.main(
    config_path="config",
    config_name="config_expert_choice_moe",
    version_base=None,
)
def main(config: DictConfig):
    is_resuming = config.training.get("resume_from_checkpoint", False)
    run_name = config.training.get("run_name", "ares-training")

    setup_logging(
        config.training.get("logging_dir", "logs"),
        run_name,
        resume=is_resuming,
        console_only=config.training.get("console_only_logging", False),
    )

    # ── XLA SPMD ──
    xr.use_spmd(auto=False)

    is_distributed = config.training.get("distributed", False)
    if is_distributed:
        init_spmd_process_group()
        logger.info(
            "Initialized SPMD process group for distributed checkpointing"
        )
        distributed_rendezvous(is_distributed, "after_process_group_init")

    device = xm.xla_device()
    set_seed(config.training.seed)

    num_devices = xr.global_runtime_device_count()
    logger.info(f"Number of devices: {num_devices}")

    # ── Training-step budget (needed by both LR and MLM schedulers) ──
    grad_accum_steps = config.training.get("gradient_accumulation_steps", 1)
    global_batch_size = (
        config.training.per_device_train_batch_size * num_devices
    )
    effective_batch_size = global_batch_size * grad_accum_steps
    logger.info(
        f"Global batch: {global_batch_size}, "
        f"grad_accum: {grad_accum_steps}, "
        f"effective batch: {effective_batch_size}"
    )

    # ── Tokenizer ──
    tokenizer = AresProteinTokenizer()

    # ── Dataset (created early to get exact step count for schedulers) ──
    if config.dataset.packed:
        train_dataset = PackedUniRef50Dataset(
            repo_id=config.dataset.repo_id,
            column_name=config.dataset.column_name,
            max_length=config.truncation.max_length,
            tokenizer=tokenizer,
            split="train",
            seed=config.dataset.seed,
            use_precomputed_bins=config.dataset.use_precomputed_bins,
            bins_repo_id=config.dataset.bins_repo_id,
        )
        num_examples_per_epoch = train_dataset.num_packed_examples
    else:
        from datasets import load_dataset as hf_load_dataset

        _raw_ds = hf_load_dataset(
            config.dataset.repo_id, split="train", streaming=False
        )
        num_examples_per_epoch = len(_raw_ds)
        del _raw_ds

    num_micro_batches_per_epoch = num_examples_per_epoch // global_batch_size
    num_steps_per_epoch = num_micro_batches_per_epoch // grad_accum_steps
    num_training_steps = num_steps_per_epoch * config.training.epochs
    warmup_steps = int(
        num_training_steps * config.training.get("warmup_ratio", 0.03)
    )

    logger.info(f"Examples per epoch: {num_examples_per_epoch:,}")
    logger.info(f"Steps per epoch: {num_steps_per_epoch:,}")
    logger.info(f"Total training steps: {num_training_steps:,}")
    logger.info(f"Warmup steps: {warmup_steps:,}")

    # ── Model ──
    model = create_model(config, vocab_size=len(tokenizer))
    initialize_gradient_checkpointing(config, model)

    # ── SPMD Mesh + FSDPv2 ──
    cols = 1
    rows = num_devices // cols
    mesh_shape = (rows, cols)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    logger.info(f"SPMD mesh: {num_devices} devices, shape {mesh_shape}")

    transformer_policy = functools.partial(
        wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=[EncoderLayer],
    )
    model = FSDPv2(
        model,
        mesh=mesh,
        shard_output=xla_sharding.shard_output,
        auto_wrap_policy=transformer_policy,
    )

    # ── Autocast ──
    use_autocast = config.training.get("autocast", False)
    logger.info(
        f"Autocast: {'enabled (bf16)' if use_autocast else 'disabled'}"
    )

    # ── Verify bf16 autocast ──
    if config.training.get("verify_bf16", False) and use_autocast:
        dummy_input = torch.randint(0, 10, (1, 8), device=device)
        with torch.no_grad(), _amp_ctx(True):
            dummy_out = model(dummy_input)
        ir_text = torch_xla._XLAC._get_xla_tensors_text([dummy_out.logits])
        has_bf16 = "bf16" in ir_text.lower() or "bfloat16" in ir_text.lower()
        has_f32 = "f32" in ir_text.lower() or "float32" in ir_text.lower()
        logger.info(f"XLA IR bf16={has_bf16}, f32={has_f32}")
        if has_bf16:
            logger.info("CONFIRMED: bf16 is active in XLA IR")
        else:
            logger.warning("No bf16 in XLA IR — autocast may not be working")
        del dummy_input, dummy_out
        torch_xla.sync()

    # ── Optimizer (must be created AFTER FSDPv2 wrapping) -- LR scheduler ──
    optimizer = create_optimizer(model=model, config=config)

    if is_distributed and config.training.get("prime_optimizer", False):
        prime_distributed_optimizer(optimizer)
        distributed_rendezvous(is_distributed, "after_prime_optimizer")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=warmup_steps,
        num_cycles=0.5,
    )

    # ── Checkpoint manager ──
    logger.info(
        f"Using checkpoint directory: {config.training.checkpoint_dir}"
    )
    checkpoint_manager = SPMDCheckpointManager(
        checkpoint_dir=config.training.checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        save_interval=1,
        max_to_keep=config.training.get("checkpoint_keep_last_k", 3),
        async_checkpointing=False,
        chkpt_on_preemption=True,
    )
    logger.info("Using SPMDCheckpointManager")

    # ── MpDeviceLoader (XLA data sharding + prefetch) ──
    mlm_scheduler = create_mlm_scheduler(config, num_training_steps)
    sampler = MLMProbabilitySampler(
        mlm_probs=list(config.masking.mlm_probability),
        masking_probs=list(config.masking.masking_probability),
        mutation_probs=list(config.masking.mutation_probability),
        scheduler=mlm_scheduler,
    )
    corruptor = SequenceCorruptor(
        tokenizer=tokenizer,
        mlm_probability_sampler=sampler,
    )

    # ── Resume ──
    start_step = 0
    is_resuming = config.training.get("resume_from_checkpoint", False)
    if is_resuming:
        if is_distributed:
            restored_step = checkpoint_manager.restore_latest()
            start_step = restored_step if restored_step is not None else 0
            logger.info(
                f"Restored training from step {start_step} in distributed mode"
            )
            distributed_rendezvous(is_distributed, "after_restore_latest")
            torch_xla.sync()
        else:
            restored_step = checkpoint_manager.load_latest()
            start_step = restored_step if restored_step is not None else 0
            logger.info(
                f"Restored training from step {start_step} in single VM mode"
            )
        if start_step > 0:
            logger.info(f"Resumed training from step {start_step}")

    # ================= Data Loading ================= #
    steps_into_current_epoch = (
        start_step % num_steps_per_epoch if start_step > 0 else 0
    )
    skip_examples = steps_into_current_epoch * effective_batch_size
    if skip_examples > 0:
        logger.info(
            f"Will skip {skip_examples} examples in dataset "
            f"(epoch {start_step // num_steps_per_epoch}, "
            f"{steps_into_current_epoch}/{num_steps_per_epoch} steps into epoch, "
            f"effective batch size {effective_batch_size})"
        )

    if config.dataset.packed:
        train_dataset.skip_examples = skip_examples
    else:
        train_dataset = load_dataset(
            config, tokenizer, corruptor, "train", skip_examples, training=True
        )

    valid_dataset = load_dataset(
        config, tokenizer, corruptor, "validation", 0, training=False
    )
    distributed_rendezvous(is_distributed, "after_dataset_init")

    if config.dataset.packed:
        train_collator = PackedBatchMaskingCollator(
            tokenizer=tokenizer,
            mlm_probability_sampler=sampler,
            seed=config.dataset.seed,
        )
        valid_collator = PackedEvalCollator()
    else:
        train_collator = AresCollator()
        valid_collator = AresCollator()

    train_dataloader_base = load_dataloader(
        config=config,
        dataset=train_dataset,
        collator=train_collator,
        num_devices=num_devices,
        training=True,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor,
    )
    valid_dataloader_base = load_dataloader(
        config=config,
        dataset=valid_dataset,
        collator=valid_collator,
        num_devices=num_devices,
        training=False,
        num_workers=0,
        prefetch_factor=None,
    )

    input_sharding = xs.ShardingSpec(mesh, ("fsdp", None))
    train_dataloader = pl.MpDeviceLoader(
        train_dataloader_base, device, input_sharding=input_sharding
    )
    valid_dataloader = pl.MpDeviceLoader(
        valid_dataloader_base, device, input_sharding=input_sharding
    )
    logger.info("Wrapped DataLoaders with MpDeviceLoader")
    distributed_rendezvous(is_distributed, "after_dataloader_wrap")

    # ── Wandb ──
    init_wandb_distributed(
        config,
        resume=is_resuming,
        start_step=start_step,
        resume_wandb=False,
        run_on_master_only=True,
    )
    distributed_rendezvous(is_distributed, "after_wandb_init")

    # ── Train ──
    logger.info("Starting training...")
    distributed_rendezvous(is_distributed, "before_train_loop")
    train(
        model=model,
        train_dataset=train_dataset,
        train_collator=train_collator,
        train_dataloader=train_dataloader,
        valid_dataset=valid_dataset,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        checkpoint_manager=checkpoint_manager,
        mlm_scheduler=mlm_scheduler,
        num_steps_per_epoch=num_steps_per_epoch,
        start_step=start_step,
    )

    global_master_process_call(wandb.finish)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
