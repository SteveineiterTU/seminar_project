from torch import nn
import torch
from torch import nn
import math
from einops import rearrange

from helpers import exists


class Residual(nn.Module):
    # TODO rename fn
    def __init__(self, fn):  # Not sure what fn is supposed to be -> mby a forward network or something like that?
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPositionEmbeddings(nn.Module):
    """Concept from https://arxiv.org/abs/1706.03762"""

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, time):  # TODO des im debugger anschauen / durchdenken, line 25 why -1, line 16
        device = time.device
        half_dimension = self.dimension // 2
        embeddings = math.log(1e+4) / (half_dimension - 1)  # TODO ASK: Why do we use log(10000) here
        embeddings = torch.exp(torch.arange(half_dimension, device=device) * (-embeddings))
        embeddings = time[:, None] * embeddings[None, :]  # TODO warum None / was macht des
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dimension, out_dimension, groups=8):
        super().__init__()
        self.projection = nn.Conv2d(dimension, out_dimension, 3, padding=1)  # Not sure if proj means projection tho.
        self.normalize = nn.GroupNorm(groups, out_dimension)
        self.activation = nn.SiLU()

    def forward(self, x, scale_and_shift=None):
        x = self.projection(x)
        x = self.normalize(x)
        if exists(scale_and_shift):
            scale, shift = scale_and_shift
            x = x * (scale + 1) + shift
        x = self.activation(x)

        return x


class ResnetBlock(nn.Module):
    """Concept from https://arxiv.org/abs/1512.03385"""

    def __init__(self, dimension, out_dimension, *, time_embedding_dimension=None, groups=8):
        super().__init__()
        self.multi_layer_perceptron = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dimension, out_dimension))
            if exists(time_embedding_dimension)
            else None
        )

        self.first_block = Block(dimension, out_dimension, groups=groups)
        self.second_block = Block(out_dimension, out_dimension, groups=groups)
        self.residual_conv = nn.Conv2d(dimension, out_dimension, 1) if dimension != out_dimension else nn.Identity()

    def forward(self, x, time_embedding=None):
        hidden_layer = self.first_block(x)  # Not sure if h stands for hidden_layer tho
        if exists(self.multi_layer_perceptron) and exists(time_embedding):
            time_embedding = self.multi_layer_perceptron(time_embedding)
            hidden_layer = rearrange(time_embedding, "b c -> b c 1 1") + hidden_layer  # TODO anschauen was das macht
        hidden_layer = self.second_block(hidden_layer)

        return hidden_layer + self.residual_conv(x)


class ConvNextBlock(nn.Module):
    """Concept from https://arxiv.org/abs/2201.03545"""
    def __init__(self, dimension, out_dimension, *, time_embedding_dimension=None, multiply_factor=2,
                 should_be_normalized=True):
        super().__init__()
        self.multi_layer_perceptron = (
            nn.Sequential(nn.GELU(), nn.Linear(time_embedding_dimension, dimension))
            if exists(time_embedding_dimension)
            else None
        )

        self.dense_conv = nn.Conv2d(dimension, dimension, 7, padding=3, groups=dimension)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dimension) if should_be_normalized else nn.Identity(),
            nn.Conv2d(dimension, out_dimension * multiply_factor, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_dimension * multiply_factor),
            nn.Conv2d(out_dimension * multiply_factor, out_dimension, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(dimension, out_dimension, 1) if dimension != out_dimension else nn.Identity

    def forward(self, x, time_embedding=None):
        hidden_layer = self.dense_conv(x)
        if exists(self.multi_layer_perceptron) and exists(time_embedding):
            condition = self.multi_layer_perceptron(time_embedding)
            hidden_layer = hidden_layer + rearrange(condition, "b c -> b c 1 1")
        hidden_layer = self.net(hidden_layer)
        return hidden_layer + self.residual_conv(x)

