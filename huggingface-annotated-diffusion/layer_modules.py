import math

import torch
from einops import rearrange
from torch import nn, einsum

from helpers import exists


class Residual(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x, *args, **kwargs):
        return self.function(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dimension, function):
        super().__init__()
        self.function = function
        self.normalize = nn.GroupNorm(1, dimension)

    def forward(self, x):
        x = self.normalize(x)
        return self.function(x)


class SinusoidalPositionEmbeddings(nn.Module):
    """Concept from https://arxiv.org/abs/1706.03762"""

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    # TODO des im debugger anschauen / durchdenken, line 25 why -1, line 16
    def forward(self, time):
        device = time.device
        half_dimension = self.dimension // 2
        # TODO ASK: Why do we use log(10000) here? Simply the # of words we want to embedd?
        embeddings = math.log(1e4) / (half_dimension - 1)
        embeddings = torch.exp(
            torch.arange(half_dimension, device=device) * (-embeddings)
        )
        # TODO warum None / was macht des
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ======================================== ResNet/ConvNext Blocks ======================================================
class Block(nn.Module):
    def __init__(self, dimension, out_dimension, groups=8):
        super().__init__()
        # Not sure if proj means projection tho.
        self.projection = nn.Conv2d(dimension, out_dimension, 3, padding=1)
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

    def __init__(
        self, dimension, out_dimension, *, time_embedding_dimension=None, groups=8
    ):
        super().__init__()
        self.multi_layer_perceptron = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dimension, out_dimension))
            if exists(time_embedding_dimension)
            else None
        )

        self.first_block = Block(dimension, out_dimension, groups=groups)
        self.second_block = Block(out_dimension, out_dimension, groups=groups)
        self.residual_conv = (
            nn.Conv2d(dimension, out_dimension, 1)
            if dimension != out_dimension
            else nn.Identity()
        )

    def forward(self, x, time_embedding=None):
        hidden_layer = self.first_block(x)  # Not sure if h stands for hidden_layer tho
        if exists(self.multi_layer_perceptron) and exists(time_embedding):
            time_embedding = self.multi_layer_perceptron(time_embedding)
            # TODO anschauen was das macht:
            hidden_layer = rearrange(time_embedding, "b c -> b c 1 1") + hidden_layer
        hidden_layer = self.second_block(hidden_layer)

        return hidden_layer + self.residual_conv(x)


class ConvNextBlock(nn.Module):
    """Concept from https://arxiv.org/abs/2201.03545"""

    def __init__(
        self,
        dimension,
        out_dimension,
        *,
        time_embedding_dimension=None,
        multiply_factor=2,
        should_be_normalized=True
    ):
        super().__init__()
        self.multi_layer_perceptron = (
            nn.Sequential(nn.GELU(), nn.Linear(time_embedding_dimension, dimension))
            if exists(time_embedding_dimension)
            else None
        )

        self.dense_conv = nn.Conv2d(
            dimension, dimension, 7, padding=3, groups=dimension
        )
        self.net = nn.Sequential(
            nn.GroupNorm(1, dimension) if should_be_normalized else nn.Identity(),
            nn.Conv2d(dimension, out_dimension * multiply_factor, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_dimension * multiply_factor),
            nn.Conv2d(out_dimension * multiply_factor, out_dimension, 3, padding=1),
        )
        self.residual_conv = (
            nn.Conv2d(dimension, out_dimension, 1)
            if dimension != out_dimension
            else nn.Identity()
        )

    def forward(self, x, time_embedding=None):
        hidden_layer = self.dense_conv(x)
        if exists(self.multi_layer_perceptron) and exists(time_embedding):
            condition = self.multi_layer_perceptron(time_embedding)
            hidden_layer = hidden_layer + rearrange(condition, "b c -> b c 1 1")
        hidden_layer = self.net(hidden_layer)
        return hidden_layer + self.residual_conv(x)


# ======================================== Attention Layers ============================================================
class Attention(nn.Module):
    def __init__(self, dimension, heads=4, head_dimension=32):
        super().__init__()
        self.scale = head_dimension**-0.5
        self.heads = heads
        hidden_dimension = head_dimension * heads
        self.to_query_key_value = nn.Conv2d(
            dimension, hidden_dimension * 3, 1, bias=False
        )
        self.to_out = nn.Conv2d(hidden_dimension, dimension, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        query_key_vlue = self.to_query_key_value(x).chunk(3, dim=1)
        # TODO Check: what is this doing
        query, key, value = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            query_key_vlue,
        )
        query = query * self.scale

        # TODO Check: what is this doing
        elements_sum = einsum("b h d i, b h d j -> b h i j", query, key)
        elements_sum = elements_sum - elements_sum.amax(dim=-1, keepdim=True).detach()
        attention = elements_sum.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attention, value)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dimension, heads=4, head_dimension=32):
        super().__init__()
        self.scale = head_dimension**-0.5
        self.heads = heads
        hidden_dimension = head_dimension * heads
        self.to_query_key_value = nn.Conv2d(
            dimension, hidden_dimension * 3, 1, bias=False
        )
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dimension, dimension, 1), nn.GroupNorm(1, dimension)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        query_key_vlue = self.to_query_key_value(x).chunk(3, dim=1)
        query, key, value = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            query_key_vlue,
        )

        # TODO why dim = -2
        query = query.softmax(dim=-2)
        key = key.softmax(dim=-1)
        query = query * self.scale
        context = einsum("b h d n, b h e n -> b h d e", key, value)

        out = einsum("b h d e, b h d n -> b h e n", context, query)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
