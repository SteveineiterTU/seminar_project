import torch
from torch import nn

from helpers import exists, default, down_sample, up_sample
from functools import partial

from layer_modules import (
    ConvNextBlock,
    ResnetBlock,
    SinusoidalPositionEmbeddings,
    Residual,
    PreNorm, Attention, LinearAttention,
)


class UNet(nn.Module):
    def __init__(
        self,
        dimension,
        initialization_dimension=None,
        out_dimension=None,
        dimension_multiplies=(1, 2, 4, 8),
        channels=3,
        with_time_embedding=True,
        resnet_block_groups=8,
        use_conv_next=True,
        conv_next_multiplier=2,
    ):
        super().__init__()

        # ======================================== Determine Dimensions ================================================
        self.channels = channels
        initialization_dimension = default(
            initialization_dimension, dimension // 3 * 2
        )
        self.initialization_conv = nn.Conv2d(
            channels, initialization_dimension, 7, padding=3
        )
        # TODO check what *map returns
        dimensions = [
            initialization_dimension,
            *map(lambda m: dimension * m, dimension_multiplies),
        ]
        in_out = list(zip(dimensions[:-1], dimensions[1:]))
        if use_conv_next:
            block_class = partial(ConvNextBlock, multiply_factor=conv_next_multiplier)
        else:
            block_class = partial(ResnetBlock, groups=resnet_block_groups)

        # ======================================== Time Embeddings =====================================================
        if with_time_embedding:
            time_dimension = dimension * 4
            self.time_multi_layer_perceptron = nn.Sequential(
                SinusoidalPositionEmbeddings(dimension),
                nn.Linear(dimension, time_dimension),
                nn.GELU(),
                nn.Linear(time_dimension, time_dimension),
            )
        else:
            time_dimension = None
            self.time_multi_layer_perceptron = None

        # ======================================== Layers ==============================================================
        self.down_sample_layers = nn.ModuleList([])
        self.up_sample_layers = nn.ModuleList([])
        num_resolutions = len(in_out)

        for index, (in_dimension, _out_dimension) in enumerate(in_out):
            is_last = index >= (num_resolutions - 1)
            self.down_sample_layers.append(
                nn.ModuleList(
                    [
                        block_class(
                            in_dimension,
                            _out_dimension,
                            time_embedding_dimension=time_dimension,
                        ),
                        block_class(
                            _out_dimension,
                            _out_dimension,
                            time_embedding_dimension=time_dimension,
                        ),
                        Residual(
                            PreNorm(_out_dimension, LinearAttention(_out_dimension))
                        ),
                        down_sample(_out_dimension) if not is_last else nn.Identity(),
                    ]
                )
            )

        middle_dimension = dimensions[-1]
        self.middle_first_block = block_class(
            middle_dimension,
            middle_dimension,
            time_embedding_dimension=time_dimension,
        )
        self.middle_attention = Residual(
            PreNorm(middle_dimension, Attention(middle_dimension))
        )
        self.middle_second_block = block_class(
            middle_dimension,
            middle_dimension,
            time_embedding_dimension=time_dimension,
        )

        for index, (in_dimension, _out_dimension) in enumerate(reversed(in_out[1:])):
            is_last = index >= (num_resolutions - 1)
            self.up_sample_layers.append(
                nn.ModuleList(
                    [
                        block_class(
                            _out_dimension * 2,
                            in_dimension,
                            time_embedding_dimension=time_dimension,
                        ),
                        block_class(
                            in_dimension,
                            in_dimension,
                            time_embedding_dimension=time_dimension,
                        ),
                        Residual(PreNorm(in_dimension, LinearAttention(in_dimension))),
                        up_sample(in_dimension) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dimension = default(out_dimension, channels)
        self.final_conv = nn.Sequential(
            block_class(dimension, dimension), nn.Conv2d(dimension, out_dimension, 1)
        )

    def forward(self, x, time):
        x = self.initialization_conv(x)
        t = (
            self.time_multi_layer_perceptron(time)
            if exists(self.time_multi_layer_perceptron)
            else None
        )
        h = []

        # ======================================== Down Sample =========================================================
        for (
            first_block,
            second_block,
            attention,
            _down_sample,
        ) in self.down_sample_layers:
            x = first_block(x, t)
            x = second_block(x, t)
            x = attention(x)
            h.append(x)
            x = _down_sample(x)

        # ======================================== Bottleneck ==========================================================
        x = self.middle_first_block(x, t)
        x = self.middle_attention(x)
        x = self.middle_second_block(x, t)

        # ======================================== Up Sample ===========================================================
        for first_block, second_block, attention, _up_sample in self.up_sample_layers:
            x = torch.cat((x, h.pop()), dim=1)
            x = first_block(x, t)
            x = second_block(x, t)
            x = attention(x)
            x = _up_sample(x)

        return self.final_conv(x)
