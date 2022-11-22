# My implementation of https://huggingface.co/blog/annotated-diffusion.

import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

from helpers import exists, default

