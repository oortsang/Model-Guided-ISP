"""
This file implements the forward and inverse neural networks for wave
scattering described in Fan and Ying "Solving Inverse Wave Scattering with
Deep Learning" 2019. We are calling it FYNet

The modifications to the architecture are that instead of using BCR-Net,
we use more 1D convolutions.

The MFISNet-Fused model is a strict superset of FYNetInverse, so this
interface simply points to MFISNet-Fused.
"""

import torch
from src.utils.conv_ops import (
    conv_in_fourier_space,
)
from src.models.MFISNet_Fused import MFISNet_Fused

import logging

class FYNetInverse(MFISNet_Fused):
    """FYNetInverse: the single-frequency case of MFISNet_Fused (N_freqs=1).

    This class used to have its own independent implementation; it is now a
    thin wrapper around MFISNet_Fused so the two share one (better-tested,
    better-maintained) set of parameter-initialization and forward logic.
    The only real work here is adapting the input shape: FYNetInverse's
    callers pass (batch, N_M, N_H, 2) with no explicit frequency axis, while
    MFISNet_Fused expects (batch, N_freqs, N_M, N_H, 2).
    """
    def __init__(
        self,
        N_h: int,
        N_rho: int,
        c_1d: int,
        c_2d: int,
        w_1d: int,
        w_2d: int,
        N_cnn_1d: int,
        N_cnn_2d: int,
        init_mode: str = None,
        **kwargs: dict,
    ) -> None:
        """The inverse NN described in section 2 of FY19. NN inputs have shape
        (batch, N_M, N_H, 2) and outputs have shape (batch, N_theta, N_rho),
        and we assume N_M == N_theta.

        Args:
            N_h  (int): Number of h grid points in (m, h) coordinates
            N_rho (int): Number of radial grid points in polar coordinates
            c_1d (int): Number of channels for 1d conv
            c_2d (int): Number of channels for 2d conv
            w_1d (int): Width of the 1d conv kernel
            w_2d (int): Width of the 2d conv kernel
            N_cnn_1d (int): Number of 1d conv layers
            N_cnn_2d (int): Number of 2d conv layers
            init_mode (str): choose which mode to use for initializing parameters
                Options:
                    [original, uniform-with-old-scale, normal-with-old-scale, he-normal]
            Miscellaneous keyword arguments (passed through to MFISNet_Fused):
                train_inputs_mean, train_inputs_std, train_outputs_mean, train_outputs_std:
                    enable input/output scaling when present, same as MFISNet_Fused
        """
        super().__init__(
            N_h=N_h,
            N_rho=N_rho,
            N_freqs=1,
            c_1d=c_1d,
            c_2d=c_2d,
            w_1d=w_1d,
            w_2d=w_2d,
            N_cnn_1d=N_cnn_1d,
            N_cnn_2d=N_cnn_2d,
            big_init=True,
            init_mode=init_mode,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the inversion model. Maps from wave fields to scattering
        objects.

        Args:
            x (torch.Tensor): Has shape (batch, N_M, N_H, 2)

        Returns:
            torch.Tensor: Has shape (batch, N_theta, N_rho)
        """
        return super().forward(x.unsqueeze(1))

    def __repr__(self) -> str:
        s = f"FYNetInverse model with {self.N_cnn_1d} 1D CNN layers, {self.N_cnn_2d} 2D CNN layers"
        s += f" 1D channel dim {self.c_1d}, 2D channel dim {self.c_2d}. 1D Convs performed with"
        s += f" {self.w_1d} modes, and 2D conv is performed with kernels of size ({self.w_2d}x{self.w_2d})."
        return s

def load_FYNetInverse_from_state_dict(
    state_dict: dict,
) -> FYNetInverse:
    """Sets up a FYNetInverse model from the given state dictionary.
    Mirrors load_MFISNet_Fused_from_state_dict, but N_freqs is always 1
    since that's fixed internally by FYNetInverse.
    """
    # First, compute hyperparameters from the state dictionary
    layer_dims = {key:tuple(val.shape) for (key, val) in state_dict.items()}
    parameter_keys = list(state_dict.keys())

    N_freqs = 1
    N_cnn_1d = sum("conv_1d_layers" in key for key in parameter_keys)
    N_cnn_2d = sum(("conv_2d_layers" in key) and ("weight" in key) for key in parameter_keys)

    (c_1d_int, c_1d_in, w_1d) = layer_dims["conv_1d_layers.0"]
    (c_1d_out, _c_1d_int, _w_1d) = layer_dims[f"conv_1d_layers.{N_cnn_1d-1}"] # last conv1d layer

    (c_2d_int, c_2d_in, w_2d, _) = layer_dims["conv_2d_layers.0.weight"]
    (c_2d_out, c_2d_int, _, _) = layer_dims[f"conv_2d_layers.{N_cnn_1d-1}.weight"]

    N_h  = c_1d_in // (2 * N_freqs)
    c_1d = c_1d_int
    c_2d = c_2d_int
    N_rho = c_1d_out

    extra_kwargs = {}
    if "train_inputs_mean" in state_dict.keys():
        extra_kwargs["train_inputs_mean"] = state_dict["train_inputs_mean"]
        extra_kwargs["train_inputs_std"] = state_dict["train_inputs_std"]
    if "train_outputs_mean" in state_dict.keys():
        extra_kwargs["train_outputs_mean"] = state_dict["train_outputs_mean"]
        extra_kwargs["train_outputs_std"] = state_dict["train_outputs_std"]

    # Next, initialize a model
    new_fynet_inverse_model = FYNetInverse(
        N_h=N_h,
        N_rho=N_rho,
        c_1d=c_1d,
        c_2d=c_2d,
        w_1d=w_1d,
        w_2d=w_2d,
        N_cnn_1d=N_cnn_1d,
        N_cnn_2d=N_cnn_2d,
        **extra_kwargs,
    )

    # Load in the values
    new_fynet_inverse_model.load_state_dict(state_dict=state_dict, strict=False)
    return new_fynet_inverse_model

class FYNetForward(torch.nn.Module):
    """FYNetForward for use with inputs/outputs that have already been cast to real dtypes
    Variable naming convention switched to agree with FYNetInverse and MFISNet-Fused.
    """
    def __init__(
        self,
        N_h: int,
        N_rho: int,
        c_1d: int,
        w_1d: int,
        N_cnn_1d: int,
        N_channels_out: int,
        polar_padding: bool = True,
        init_mode: str = None,
    ) -> None:
        """The Forward NN described in section 2 of FY19. NN inputs have shape
        (batch, N_theta, N_rho) and outputs have shape (batch, N_M, N_H, 2),
        and we assume N_M == N_theta.

        Only uses the conv1d portions

        Args:
            N_h  (int): Number of h grid points in (m, h) coordinates
            N_rho (int): Number of radial grid points in polar coordinates
            c_1d (int): Number of channels for 1d conv
            w_1d (int): Width of the 1d conv kernel
            N_cnn_1d (int): Number of 1d conv layers
            N_cnn_2d (int): Number of 2d conv layers

            init_mode (str): choose which mode to use for initializing parameters
                Options:
                    [original, uniform-with-old-scale, normal-with-old-scale, he-normal]
        """
        super().__init__()

        # Assume N_theta=N_m and is not required to be stored explicitly
        self.N_h = N_h
        self.N_rho = N_rho
        self.c_1d = c_1d
        self.w_1d = w_1d
        self.N_cnn_1d = N_cnn_1d
        self.N_channels_out = N_channels_out

        init_mode = init_mode.lower() if init_mode is not None else "original"
        self.init_mode = init_mode
        self.weight_dtype = torch.complex64
        self.real_dtype   = torch.float32

        padding_1d = int(self.w_1d / 2 - 1) + 1

        self.c_1d_in  = self.N_rho
        self.c_1d_int = self.c_1d # internal layers
        self.c_1d_out = (self.N_channels_out * self.N_h * 2)

        cnn1d_freq_h_axis_dims = [
            self.c_1d_in, # Input
            *((self.N_cnn_1d-1)*[self.c_1d_int]), # Interior
            self.c_1d_out,  # Final layer of 1D section
        ]
        logging.info(f"(conv1d) freq/h axis has {cnn1d_freq_h_axis_dims} channels")

        self.conv_1d_layers = torch.nn.ParameterList([])
        for li in range(self.N_cnn_1d):
            h_dim_in    = cnn1d_freq_h_axis_dims[li]
            h_dim_out   = cnn1d_freq_h_axis_dims[li+1]
            old_scaling = 2 / h_dim_in
            scale_he = torch.sqrt(torch.tensor(2 / h_dim_in, dtype=self.real_dtype))

            if init_mode == "original":
                scaling = old_scaling
                rand_fn = torch.rand
            elif init_mode == "uniform-with-old-scale":
                scaling = old_scaling
                rand_fn = lambda x: 2*torch.rand(x)-1
            elif init_mode == "normal-with-old-scale":
                scaling = old_scaling
                rand_fn = torch.randn
            elif init_mode == "he-normal":
                rand_fn  = torch.randn
                scaling  = scale_he
            else:
                raise ValueError(
                    f"FYNetForwardReal.__init__ received init_mode={init_mode} "
                    f"which is not in the recognized list of options: [original, "
                    f"uniform-with-old-scale, normal-with-old-scale, he-normal]",
                )
            new_params = scaling * rand_fn(
                h_dim_out,
                h_dim_in,
                self.w_1d,
                dtype=self.weight_dtype
            )
            self.conv_1d_layers.append(new_params)

        param_shapes = [p.shape for p in self.parameters()]
        param_numels = [p.numel() for p in self.parameters()]
        logging.info(f"FYNet-Forward contains parameters with sizes {param_shapes} "
                     f"for a total of {sum(param_numels)} parameters")

        self.relu = torch.nn.ReLU()

    def __repr__(self: None) -> str:
        s = f"FYNetForwardReal model with {self.N_cnn_1d} layers, channel dimension"
        s += f" {self.c1d}, and kernels with # freq modes: {self.w1d}"
        return s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the inversion model. Maps from wave fields to scattering
        objects. Here's a list of the intermediate shapes of the data
        In (N_batch, N_theta, self.N_rho) # Input shape
        -> (N_batch, self.N_rho, N_m) # Transpose so we can conv over the theta/m axis
        -> (N_batch, self.c_1d, N_m)  # Within the conv1d layers
        -> (N_batch, N_channels_out * N_h * 2, N_m) # Output after the conv1d layers
        -> (N_batch, N_channels_out, N_h, 2, N_m) # Split axes
        -> (N_batch, N_channels_out, N_m, N_h, 2) # Permute the axes

        Note that N_m = N_theta

        Args:
            x (torch.Tensor): Has shape  (batch, N_theta, N_rho)

        Returns:
            torch.Tensor: Has shape (N_batch, N_channels_out, N_m, N_h, 2)
        """
        N_batch = x.shape[0]
        N_m = x.shape[-2]

        conv1d_input = torch.transpose(x, -2, -1)

        conv1d_internal = conv1d_input
        for li, layer in enumerate(self.conv_1d_layers):
            conv1d_internal = conv_in_fourier_space(conv1d_internal, layer).real
            if li < self.N_cnn_1d - 1:
                conv1d_internal = self.relu(conv1d_internal)
        conv1d_output = conv1d_internal

        tmp = conv1d_output.reshape(N_batch, self.N_channels_out, self.N_h, 2, N_m)
        output = tmp.permute(0, 1, 4, 2, 3)

        return output
