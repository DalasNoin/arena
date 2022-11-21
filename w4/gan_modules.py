import torch as t
from typing import Union, Tuple
from torch import nn
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from fancy_einsum import einsum
import os
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset
import wandb
import utils

def conv_transpose1d_minimal(x: t.Tensor, weights: t.Tensor, padding: int=0) -> t.Tensor:
    '''Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    # reverse kernel
    weight_mod = t.flip(weights, dims=(2,))
    weight_mod = rearrange(weight_mod, "a b c -> b a c")
    x_width = x.shape[2]
    # change input with padding
    kernel_width = weights.shape[2]
    padding_size = kernel_width -1 - padding
    x_mod = t.zeros((x.shape[0], x.shape[1],x.shape[2]+2*padding_size), dtype=x.dtype)
    x_mod[..., padding_size:padding_size + x_width] = x
    # apply conv1d, todo: maybe use own implementation
    result = t.nn.functional.conv1d(x_mod, weight_mod)
    return result

utils.test_conv_transpose1d_minimal(conv_transpose1d_minimal)

def fractional_stride_1d(x, stride: int = 1):
    '''Returns a version of x suitable for transposed convolutions, i.e. "spaced out" with zeros between its values.
    This spacing only happens along the last dimension.

    x: shape (batch, in_channels, width)

    Example: 
        x = [[[1, 2, 3], [4, 5, 6]]]
        stride = 2
        output = [[[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]]
    '''
    x_mod = t.zeros((x.shape[0], x.shape[1],(x.shape[2]-1)*stride + 1), dtype=x.dtype)
    x_mod[..., ::stride] = x
    return x_mod



def conv_transpose1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    x = fractional_stride_1d(x,stride=stride)
    # reverse kernel
    result = conv_transpose1d_minimal(x=x,weights=weights, padding=padding)
    return result



IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, Tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

def fractional_stride_2d(x, stride_h: int, stride_w: int):
    '''
    Same as fractional_stride_1d, except we apply it along the last 2 dims of x (width and height).
    x.shape = (batch, in_channels, height, width)
    '''

    x_mod = t.zeros((x.shape[0], x.shape[1],(x.shape[2]-1)*stride_h + 1,(x.shape[3]-1)*stride_w + 1), dtype=x.dtype)
    x_mod[..., ::stride_h, ::stride_w] = x
    return x_mod

def conv_transpose2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv_transpose2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    x = fractional_stride_2d(x, stride_h=stride_h, stride_w=stride_w)
    weight_mod = t.flip(weights, dims=(2,3))
    weight_mod = rearrange(weight_mod, "a b h w -> b a h w")
    x_height, x_width = x.shape[2], x.shape[3]
    # create new tensor of padded size
    kernel_height, kernel_width = weights.shape[2], weights.shape[3]
    padding_height_total = kernel_height - 1 - padding_h
    padding_width_total = kernel_width - 1 - padding_w
    x_mod = t.zeros((x.shape[0], 
                        x.shape[1],
                        x.shape[2]+2*padding_height_total, 
                        x.shape[3]+2*padding_width_total),
                        dtype=x.dtype)

    # move values to new tensor
    x_mod[..., padding_height_total:padding_height_total + x_height, 
            padding_width_total:padding_width_total+x_width] = x

    # apply conv2d, todo: maybe use own implementation
    result = t.nn.functional.conv2d(x_mod, weight_mod)
    return result
    



class ConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.ConvTranspose2d with bias=False.

        Name your weight field `self.weight` for compatibility with the tests.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = force_pair(kernel_size)
        self.stride = force_pair(stride)
        self.padding = force_pair(padding)

        # from steamlit: sqrt(k)=sqrt(1/(out_channels×kernel_width×kernel_height))
        sqrt_k = (1 / (self.out_channels*self.kernel_size[0]*self.kernel_size[1])) ** 0.5

        # self.weight = nn.Parameter(2*k*t.rand(in_channels, out_channels, *self.kernel_size) - k)
        self.weight = nn.Parameter(t.empty((in_channels, out_channels, *self.kernel_size)).uniform_(-sqrt_k,sqrt_k))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Apply conv transpose 2d
        """
        result = conv_transpose2d(x=x, weights=self.weight, stride=self.stride, padding=self.padding)
        return result



class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        # (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return t.tanh(x)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        # if x > 0: x else: x * negative_slope
        return nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
    
    def extra_repr(self) -> str:
        return f"LeakyReLU negative_slope={self.negative_slope}"



class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        # 1 / (1 + exp(-x))
        return t.sigmoid(x)


if __name__=="__main__":
    utils.test_fractional_stride_1d(fractional_stride_1d)
    utils.test_conv_transpose1d(conv_transpose1d)
    utils.test_conv_transpose2d(conv_transpose2d)
    utils.test_ConvTranspose2d(ConvTranspose2d)
    utils.test_Tanh(Tanh)
    utils.test_LeakyReLU(LeakyReLU)
    utils.test_Sigmoid(Sigmoid)
    utils.test_conv_transpose1d_minimal(conv_transpose1d_minimal)
