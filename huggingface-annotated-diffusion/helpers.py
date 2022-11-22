from inspect import isfunction

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
