import functools

import torch
import torch_xla.distributed.spmd as xs  # type: ignore


@functools.lru_cache(maxsize=1000)
def get_spec(ndim: int):
    # Always shard the first dimension (Batch)
    # Replicate all other dimensions (Sequence, Hidden, Vocab, etc.)
    return ("fsdp",) + (None,) * (ndim - 1)


def shard_output(outputs, mesh):
    for t in outputs:
        if not isinstance(t, torch.Tensor):
            continue
        if t.ndim == 0:
            xs.mark_sharding(t, mesh, ())
            continue
        spec = get_spec(t.ndim)
        xs.mark_sharding(t, mesh, spec)


def shard_batch(batch, device, mesh):
    for k in batch.keys():
        batch[k] = batch[k].to(device, non_blocking=True)
        xs.mark_sharding(
            batch[k],
            mesh,
            get_spec(batch[k].ndim, initial_axes=("fsdp", None)),
        )
    return batch
