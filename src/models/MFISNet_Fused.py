# MFISNet-Fused
# Inspired by FYNet but uses one adjoint network
# to take in all input frequencies

"""
This file implements a neural network meant as a baseline for our approach to
use machine learning methods for inverse scattering in the case where
measurements are available at multiple different frequencies.

This approach is similar to the approach outlined in FYNet.py, but
this model is extended to accept multi-frequency data by treating
data at each frequency as a different data channel. The 1D convolutions
allow for interactions between the different frequencies, and all the
data is merged into a single channel for the 2D conv filtering step.
Like FYNet.py, 2D convolutions always use polar padding.

The forward and inverse versions of the neural network
to recover scattering objects from the measured wavefield patterns.
The architecture is based on Fan and Ying's "Solving Inverse Wave Scattering
with Deep Learning" from 2019, with the BCR-Net replaced with 1D convolutions
as described in the main FYNet.py function.
"""
import torch
import logging
from src.utils.conv_ops import (
    conv_in_fourier_space,
    apply_conv_with_polar_padding,
)

class MFISNet_Fused(torch.nn.Module):
    def __init__(
        self,
        N_h: int,
        N_rho: int,
        N_freqs: int,
        c_1d: int,
        c_2d: int,
        w_1d: int,
        w_2d: int,
        N_cnn_1d: int,
        N_cnn_2d: int,
        big_init: bool = True,
        init_mode: str = None,
        **kwargs: dict,
    ) -> None:
        """
        Baseline comparison for a multi-frequency approach using similar network architecture.

        Based off the inverse NN described in section 2 of FY19. NN inputs have shape
        (batch, N_freqs, N_M, N_H, 2) and outputs have shape (batch, N_theta, N_rho),
        and we assume N_M == N_theta.

        Args:
            N_h   (int): number of h grid points
            N_rho (int): number of rho grid points
            N_freqs (int): number of frequencies used in the input data
            c_1d (int): Number of channels for 1d conv
            c_2d (int): Number of channels for 2d conv
            w_1d (int): Kernel size for 1d conv filter
            w_2d (int): Kernel size for 2d conv filter
            N_cnn_1d (int): Number of 1d conv layers
            N_cnn_2d (int): Number of 2d conv layers
            big_init (bool): decide whether to use the bigger initialization on the 1D conv layers
            init_mode (bool): whether to initialize the values as non-negative
        """
        super().__init__()

        self.N_h = N_h
        self.N_rho = N_rho
        self.N_freqs = N_freqs
        self.c_1d = c_1d
        self.c_2d = c_2d
        self.w_1d = w_1d
        self.w_2d = w_2d
        self.N_cnn_1d = N_cnn_1d
        self.N_cnn_2d = N_cnn_2d

        self.big_init = big_init
        # self.nonneg_init = nonneg_init
        init_mode = init_mode.lower() if init_mode is not None else "original"
        self.init_mode = init_mode
        self.forward_network_bool = False
        self.weight_dtype = torch.complex64
        self.real_dtype   = torch.float32

        torch_scalar_zero = torch.tensor(0, dtype=self.real_dtype)
        torch_scalar_one  = torch.tensor(1, dtype=self.real_dtype)
        self.scale_inputs = "train_inputs_mean" in kwargs.keys()
        self.scale_outputs = "train_outputs_mean" in kwargs.keys()
        if self.scale_inputs:
            train_inputs_mean = kwargs.get("train_inputs_mean", torch_scalar_zero)
            train_inputs_std  = kwargs.get("train_inputs_std", torch_scalar_one)
        else:
            train_inputs_mean = torch_scalar_zero
            train_inputs_std  = torch_scalar_one
        if self.scale_outputs:
            train_outputs_mean = kwargs.get("train_outputs_mean", torch_scalar_zero)
            train_outputs_std  = kwargs.get("train_outputs_std", torch_scalar_one)
        else:
            train_outputs_mean = torch_scalar_zero
            train_outputs_std  = torch_scalar_one
        self.train_inputs_mean  = torch.nn.Parameter(train_inputs_mean, requires_grad=False)
        self.train_inputs_std   = torch.nn.Parameter(train_inputs_std, requires_grad=False)
        self.train_outputs_mean = torch.nn.Parameter(train_outputs_mean, requires_grad=False)
        self.train_outputs_std  = torch.nn.Parameter(train_outputs_std, requires_grad=False)

        assert (
            self.w_2d % 2
        ), "I can't figure out how to do padding for kernel sizes divisible by 2"

        padding_1d = int(self.w_1d / 2 - 1) + 1
        padding_2d = int(self.w_2d / 2 - 1) + 1

        c_1d_in  = self.N_freqs * (self.N_h * 2)
        c_1d_int = self.c_1d # internal layers
        c_1d_out = self.N_rho
        c_2d_in  = 1
        c_2d_int = self.c_2d
        c_2d_out = 1

        logging.info(f"c_1d_in={c_1d_in}, c_1d_int: {c_1d_int}, c_1d_out: {c_1d_out}")
        logging.info(f"c_2d_in={c_2d_in}, c_2d_int: {c_2d_int}, c_2d_out: {c_2d_out}")

        # Save parameter values for later reference
        self.c_1d_in  = c_1d_in
        self.c_1d_int = c_1d_int
        self.c_1d_out = c_1d_out
        self.c_2d_in  = c_2d_in
        self.c_2d_int = c_2d_int
        self.c_2d_out = c_2d_out

        ### Initialize parameter values ###
        ## 1D CNN
        # First layer
        # Dimensions along the freq/h axis
        cnn1d_freq_h_axis_dims = [
            c_1d_in, # Input
            *((self.N_cnn_1d-1)*[c_1d_int]), # Interior
            c_1d_out,  # Final layer of 1D section
        ]
        logging.info(f"During the 1D Conv stage, freq/h axis has {cnn1d_freq_h_axis_dims} channels")

        self.conv_1d_layers = torch.nn.ParameterList([])
        for li in range(self.N_cnn_1d):
            h_dim_in    = cnn1d_freq_h_axis_dims[li]
            h_dim_out   = cnn1d_freq_h_axis_dims[li+1]
            old_scale_big   = 2 / h_dim_in
            old_scale_small = 1 / (h_dim_in * h_dim_out)
            old_scaling = old_scale_big if big_init else old_scale_small
            scale_he = torch.sqrt(torch.tensor(2 / h_dim_in, dtype=self.real_dtype))

            if init_mode == "original":
                scaling = old_scaling
                rand_fn = torch.rand
            elif init_mode == "uniform-with-old-scale":
                scaling = old_scaling
                rand_fn = lambda *args, **kwargs: 2*torch.rand(*args,**kwargs)-1
            elif init_mode == "normal-with-old-scale":
                scaling = old_scaling
                rand_fn = torch.randn
            elif init_mode == "he-normal":
                rand_fn  = torch.randn
                scaling  = scale_he
            else:
                raise ValueError(
                    f"MFISNet_Fused.__init__ received init_mode={init_mode} "
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

        ## 2D CNN
        # 2D Conv over (rho, theta) but use frequencies as channels
        # Dimensions along the h axis
        cnn2d_freq_h_axis_dims = [
            c_2d_in, # Input
            *((self.N_cnn_2d-1)*[c_2d_int]), # Interior
            c_2d_out, # Final layer
        ]
        logging.info(f"During the 2D Conv stage, freq/h axis has {cnn2d_freq_h_axis_dims} channels")
        self.conv_2d_layers = torch.nn.ParameterList([])
        for li in range(self.N_cnn_2d):
            freq_dim_in  = cnn2d_freq_h_axis_dims[li]
            freq_dim_out = cnn2d_freq_h_axis_dims[li+1]

            new_layer = torch.nn.Conv2d(
                in_channels=freq_dim_in,
                out_channels=freq_dim_out,
                kernel_size=self.w_2d,
                padding=padding_2d,
                padding_mode="zeros"
            )
            if init_mode == "he-normal":
                torch.nn.init.kaiming_normal_(new_layer.weight, nonlinearity="relu")
            self.conv_2d_layers.append(new_layer)

        param_shapes = [p.shape for p in self.parameters()]
        param_numels = [p.numel() for p in self.parameters()]
        logging.info(f"# MFISNet-Fused contains parameters with sizes {param_shapes} "
                     f"for a total of {sum(param_numels)} parameters")

        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the inversion model. Maps from wave fields to scattering
        objects. Here's a list of the intermediate shapes of the data
        In (N_batch, N_freqs, N_m, N_h, 2) # Input shape
        -> (N_batch, N_freqs, N_m, N_h * 2) # Collapse the real and imag parts of the array into the H dimension
        -> (N_batch, N_freqs * N_h * 2, N_m) # Reshape to perform 1D convolution along the M dimension
        -> (N_batch, N_freqs * self.c_1d, N_m) # Channel dimension becomes whatever the specified self.c_1d is
        -> (N_batch, N_freqs * self.N_rho, N_m) # Last conv1d block outputs self.N_rho channels
        -> (N_batch, N_freqs * self.N_rho, N_m) # Reshape to have a new channel dimension and do 2d conv on the final two dimensions
        -> (N_batch, N_freqs * self.c_2d, self.N_rho, N_m) # During conv2d, the 2nd axis becomes the channel dimension
        -> (N_batch, 1, self.N_rho, N_m) # Last conv2d block gets rid of the channel dimension
        -> (N_batch, self.N_rho, N_m) # Collapse down the singleton channel
        -> (N_batch, N_m, self.N_rho) # Transpose to meet output shape requirements
        Note that N_m = N_theta

        Args:
            x (torch.Tensor): Has shape (N_batch, N_freqs, N_m, N_h, 2)

        Returns:
            torch.Tensor: Has shape (batch, N_theta, N_rho)
        """
        N_batch = x.shape[0]
        N_m = x.shape[-3]
        N_h = x.shape[-2]
        N_freqs = self.N_freqs

        if self.scale_inputs:
            x = (x - self.train_inputs_mean) / self.train_inputs_std

        x = x.reshape((N_batch, N_freqs, N_m, 2*N_h))
        x = x.permute((0, 1, 3, 2)) # Send to (N_batch, N_freqs, 2*N_h, N_m)
        x = x.reshape((N_batch, N_freqs * 2*N_h, N_m))

        # First, 1D convolution across the N_M dimension.
        for kernel_weights in self.conv_1d_layers:
            # logging.info(f"x.shape: {x.shape}, kernel_weights.shape: {kernel_weights.shape}")
            x = conv_in_fourier_space(x, kernel_weights).real
            x = self.relu(x)
        # Resulting x shape: (N_batch, N_freqs*N_rho, N_m) or (N_batch, N_rho, N_m)
        # Can just use (N_batch, self.num_channels_mid, N_m)

        x = x.view((N_batch, -1, self.N_rho, N_m))

        # 2D convolutions in the (N_M, N_H) plane
        for i, conv_2d_layer in enumerate(self.conv_2d_layers):
            x = apply_conv_with_polar_padding(conv_2d_layer, x)
            if i == self.N_cnn_2d - 1:
                # Skip the last relu
                break
            else:
                x = self.relu(x)

        out = x.view((N_batch, -1, N_m))
        out = out.permute((0, 2, 1)) # (N_batch, N_theta, N_rho)
        if self.scale_inputs:
            out = self.train_outputs_std * out + self.train_outputs_mean
        return out

    def __repr__(self) -> str:
        s = f"MFISNet-Fused model with {self.N_cnn_1d} 1D CNN layers, {self.N_cnn_2d} 2D CNN layers"
        s += f" 1D channel dim {self.c_1d}, 2D channel dim {self.c_2d}. 1D Convs performed with"
        s += f" {self.w_1d} modes, and 2D conv is performed with kernels of size ({self.w_2d}x{self.w_2d})."
        return s

def load_MFISNet_Fused_from_state_dict(
    state_dict: dict,
    N_freqs: int,
) -> MFISNet_Fused:
    """Sets up a MFISNet-Fused model from the given state dictionary and number of frequencies"""
    # First, compute hyperparameters from the state dictionary
    layer_dims = {key:tuple(val.shape) for (key, val) in state_dict.items()}
    parameter_keys = list(state_dict.keys())

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
    new_mfisnet_fused_model = MFISNet_Fused(
        N_h=N_h,
        N_rho=N_rho,
        N_freqs=N_freqs,
        c_1d=c_1d,
        c_2d=c_2d,
        w_1d=w_1d,
        w_2d=w_2d,
        N_cnn_1d=N_cnn_1d,
        N_cnn_2d=N_cnn_2d,
        big_init=True, # just use this as a default value but it doesn't really matter,
        **extra_kwargs,
    )

    # Load in the values
    new_mfisnet_fused_model.load_state_dict(state_dict=state_dict, strict=False)
    return new_mfisnet_fused_model
