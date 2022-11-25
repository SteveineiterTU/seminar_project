from inspect import isfunction

import torch
import numpy as np
from torch import nn


def exists(x):
    return x is not None


def default(value, default_value):
    if exists(value):
        return value
    return default_value() if isfunction(default_value) else default_value


def up_sample(dimension):
    return nn.ConvTranspose2d(dimension, dimension, 4, 2, 1)


def down_sample(dimension):
    return nn.Conv2d(dimension, dimension, 4, 2, 1)


def extract(alpha, tensor, x_shape):
    batch_size = tensor.shape[0]
    out = alpha.gather(-1, tensor.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(tensor.device)


def num_to_groups(number, divisor):
    groups = number // divisor
    remainder = number % divisor
    group_array = [divisor] * groups
    if remainder > 0:
        group_array.append(remainder)
    return group_array


# ======================================== Beta Schedulers =============================================================
def cosine_beta_schedule(time_steps, step_size=0.008):
    """cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = time_steps + 1
    x = torch.linspace(0, time_steps, steps)
    alphas_cumulative_product = (
        torch.cos(((x / time_steps) + step_size) / (1 + step_size) * np.pi * 0.5) ** 2
    )
    alphas_cumulative_product = alphas_cumulative_product / alphas_cumulative_product[0]
    _betas = 1 - (alphas_cumulative_product[1:] / alphas_cumulative_product[:-1])
    return torch.clip(_betas, 1e-4, 0.9999)


def linear_beta_schedule(time_steps):
    beta_start = 1e-4
    beta_end = 2e-2
    return torch.linspace(beta_start, beta_end, time_steps)


def quadratic_beta_schedule(time_steps):
    beta_start = 1e-4
    beta_end = 2e-2
    return torch.linspace(beta_start**0.5, beta_end**0.5, time_steps) ** 2


def sigmoid_beta_schedule(time_steps):
    beta_start = 1e-4
    beta_end = 2e-2
    _betas = torch.linspace(-6, 6, time_steps)
    return torch.sigmoid(_betas) * (beta_end - beta_start) + beta_start