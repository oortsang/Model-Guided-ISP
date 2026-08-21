# MMGUBlock.py
# Measurement misfit gradient update block
# Corresponds to the CNN component of the HPS-CNN model's refinement blocks in the paper
from typing import Tuple, List, Callable
from collections import OrderedDict
import logging

import torch

from src.data.datasets import TupleLinearData

class CNN2DCartesian(torch.nn.Module):
    """Basic 2D CNN meant for Cartesian-space objects with ReLU activation
    """
    def __init__(
        self,
        N_x: int,
        N_y: int,
        N_cnn_layers: int,
        c2d_input: int,
        c2d_hidden: int,
        c2d_output: int,
        kernel_size: int,
        skip_connection: bool = True,
    ):
        """CNN2DCartesian setup
        Inputs shape:
            (N_batch, c2d_input, N_x, N_y)
        Outputs shape:
            (N_batch, c2d_output, N_x, N_y)
        The second axis acts as the channel dimension

        Parameters:
            N_x (int): number of gridpoints on the x axis
            N_y (int): number of gridpoints on the y axis
            N_cnn_layers (int): number of cnn layers
                There is ReLU activation after each layer except the last one
            c2d_input (int): number of channels expected in the input
            c2d_hidden (int): number of channels in the hidden layers
            c2d_output (int): number of channels in the output
            kernel_size (int): side length of the convolutional kernel in number of
                pixels; currently the padding code expects this to be an odd number
            skip_connection (bool): if this flag is set, the last channel
                of the input is added to the output

        """
        super().__init__()
        self.N_x = N_x
        self.N_x = N_x
        self.N_cnn_layers = N_cnn_layers
        self.c2d_in       = c2d_input
        self.c2d_hidden   = c2d_hidden
        self.c2d_out      = c2d_output
        self.kernel_size  = kernel_size
        self.skip_connection = skip_connection
        self.relu = torch.nn.ReLU()

        padding_2d = int(kernel_size / 2 - 1) + 1
        self.channel_dims = [c2d_input] + (N_cnn_layers-1) * [c2d_hidden] + [c2d_output]
        self.cnn2d_layers = torch.nn.ParameterList()
        for li in range(N_cnn_layers):
            dim_in  = self.channel_dims[li]
            dim_out = self.channel_dims[li+1]
            new_layer = torch.nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=kernel_size,
                padding=padding_2d,
                padding_mode="zeros"
            )
            torch.nn.init.kaiming_normal_(new_layer.weight, nonlinearity="relu")
            self.cnn2d_layers.append(new_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the conv2d on a cartesian object
        Input:  (N_batch, in_channels,  N_x, N_y)
        Output: (N_batch, out_channels, N_x, N_y)
        """
        in_shape = x.shape
        if self.skip_connection:
            add_slice = x[:, -1, None, ...] # broadcast over out_channels
        for i, layer_i in enumerate(self.cnn2d_layers):
            x = layer_i(x)
            if i+1 < self.N_cnn_layers:
                x = self.relu(x)
        out = x
        tmp_shape = x.shape
        if self.skip_connection:
            out = out + add_slice
        # print(f"Conv2DCart: {in_shape} -> {x.shape} -> {out.shape} (add_slice: {add_slice.shape})")
        return out

class MMGUBlock(torch.nn.Module):
    """Measurement misfit gradient update block (the learned part of the HPS-CNN architecture)
    Inspired by recursive linearization by using measurement misfit gradients
    gamma = (DF[qhat]^*(dk-F[qhat]))
    along with qhat to update the estimate
    """
    def __init__(
        self,
        N_x: int,
        N_y: int,
        N_cnn_layers: int,
        c2d_hidden: int,
        kernel_size: int,
        c2d_input: int = 2,
        skip_connection: bool = True,
        learn_cnn_scale: bool = False,
        init_cnn_scale: float = 1.0,
    ):
        """MMGU setup
        Inputs shape:
            (N_batch, c2d_input, N_x, N_y)
        The second axis acts as the channel dimension.
        Outputs shape:
            (N_batch, N_x, N_y)

        Parameters:
            N_x (int): number of gridpoints on the x axis
            N_y (int): number of gridpoints on the y axis
            N_cnn_layers (int): number of cnn layers
                There is ReLU activation after each layer except the last one
            c2d_hidden (int): number of channels in the hidden layers
            kernel_size (int): side length of the convolutional kernel in number of
                pixels; currently the padding code expects this to be an odd number
            c2d_input (int): number of channels expected in the input, usually expected to be 3
            skip_connection (bool): if this flag is set, the last channel
                of the input is added to the output
            use_cnn_scale (bool): can choose whether to use a learnable scaling parameter
        """
        super().__init__()
        self.relu = torch.nn.ReLU()

        # Set up the CNN blocks
        self.cnn2d = CNN2DCartesian(
            N_x=N_x,
            N_y=N_y,
            N_cnn_layers=N_cnn_layers,
            c2d_input=c2d_input,
            c2d_hidden=c2d_hidden,
            c2d_output=1,
            kernel_size=kernel_size,
            skip_connection=skip_connection,
        )
        self.learn_cnn_scale = learn_cnn_scale
        self.cnn_scale = (
            torch.nn.Parameter(torch.tensor(init_cnn_scale, dtype=torch.float32))
            if learn_cnn_scale
            else init_cnn_scale # just use as a fixed scalar otherwise
        )

    def forward(self, x_tup: Tuple[torch.Tensor], preprocessed: bool=True) -> torch.Tensor:
        """Take in x_tup=([qhat, Dfh_diff], dk), where DFh_diff=DF[qhat]^*(dk-F[qhat])
        If preprocessed==False then assume x_tup=(qhat, dk)
        the q-space object will contain Dfh_diff as
        q_space_obj shape: (N_batch, (3?), N_x, N_y)
        qhat shape: (N_batch, N_x, N_y)
        Dfh_diff shape: (N_batch, 2, N_x, N_y) # 2 for complex values

        dk shape:   (N_batch, N_r, N_s, 2) # or perhaps N_s, N_r
        output shape:   (N_batch, 1, N_x, N_y)

        Next step: process via conv2d
        """
        q_space_obj, dk = x_tup
        cnn2d_input = q_space_obj
        cnn2d_update = self.cnn2d(cnn2d_input).squeeze(1)
        cnn2d_output = q_space_obj[:, -1] + self.cnn_scale * cnn2d_update
        return cnn2d_output


def load_MMGUBlock_from_state_dict(
    state_dict: OrderedDict,
    summary_results_dd: dict,
) -> MMGUBlock:
    """Load a MMGUBlock from the state dict and summary results dictionary"""
    N_x = summary_results_dd["n_x_vals"]
    N_cnn_layers   = summary_results_dd["n_cnn_layers_2d"]
    N_cnn_channels = summary_results_dd["n_cnn_channels_2d"]
    kernel_size = summary_results_dd["kernel_size_2d"]
    learn_cnn_scale = summary_results_dd.get("learn_cnn_scale", False)
    init_cnn_scale  = summary_results_dd.get("init_cnn_scale", 1.0)

    new_mmgublock_model = MMGUBlock(
        N_x,
        N_x,
        N_cnn_layers=N_cnn_layers,
        c2d_hidden=N_cnn_channels,
        kernel_size=kernel_size,
        c2d_input=2,
        skip_connection=True,
        # hss=hss,
        learn_cnn_scale=learn_cnn_scale,
        init_cnn_scale=init_cnn_scale,
    )
    new_mmgublock_model.load_state_dict(state_dict=state_dict, strict=False)
    return new_mmgublock_model
