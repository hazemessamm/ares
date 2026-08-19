import torch
from torch import nn


def soft_capping(x, capping_value):
    return capping_value * torch.tanh(x / capping_value)


def sample_gumbel(shape, device=None, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, scale=1, dim=-1):
    scaled_gumbel_dist = (
        sample_gumbel(logits.size(), device=logits.device) * scale
    )
    y = logits + scaled_gumbel_dist
    return nn.functional.softmax(y, dim=dim)


def jitter_noise(x, noise_scale=0.01):
    """
    Adds jitter noise to the input tensor.

    The noise is sampled from a uniform distribution:
    U(-noise_scale, noise_scale). This is often used in Mixture of Experts
    models to improve load balancing by adding small amounts of noise to
    the router logits.

    Uses torch.rand_like (not torch.empty_like().uniform_()) so that the
    RNG state is properly tracked by gradient checkpointing.
    """
    noise = (2 * torch.rand_like(x) - 1) * noise_scale
    return x + noise
