# MFISNet-Refinement-Block
# Implements a single per-frequency refinement block for MFISNet-Refinement
# (MRef): given the previous frequency level's estimate and the current
# frequency's data (in the form of [Fk[q], Fk[q]-dk, dk]), produces an
# updated estimate.
# Also contains a dataset type, TupleLinearData, for use with this setup.

import torch
import logging
from typing import List

from src.models.MFISNet_Fused import MFISNet_Fused
from src.models.FYNet import FYNetForward

from src.utils.conv_ops import (
    apply_conv_with_polar_padding,
)

NUM_D_MH_CHANNELS = 3 # for debugging purposes use just 1 channel for the d_mh input

class TupleLinearData(torch.utils.data.Dataset):
    def __init__(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        y: torch.Tensor,
        y_orig: torch.Tensor = None
    ) -> None:
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.y_orig = y_orig if y_orig is not None else y
        logging.info(
            "Initialized a TupleData instance with X shape: %s and y shape: %s",
            (self.X1.shape, self.X2.shape,),
            self.y.shape,
        )
        self.n_samples = X2.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # (OOT 6/4/2024) Gives two copies of the target because the training
        # loop seems to expect a filtered and final version of the sample
        # (OOT 9/8/2024) Wrap the two inputs to accommodate inputs of different shapes
        # return TupleWrapper(self.X1[idx], self.X2[idx]), self.y[idx], self.y[idx]
        return [self.X1[idx], self.X2[idx]], self.y[idx], self.y_orig[idx]

class KLayer2DCNN(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_in_channels: int,
        n_out_channels: int,
        n_feature_channels: int,
        kernel_size: int,
        skip_connection: bool = False,
        init_mode: str = None,
        angular_axis_last: bool = False,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_feature_channels = n_feature_channels
        self.kernel_size = kernel_size
        self.skip_connection = skip_connection

        init_mode = init_mode.lower() if init_mode is not None else "original"
        self.init_mode = init_mode

        padding_2d = int(kernel_size / 2 - 1) + 1
        self.cnn_layers = torch.nn.ParameterList()

        channel_dims = [
            n_in_channels, # Input (=1)
            *((n_layers-1)*[n_feature_channels]), # Interior
            n_out_channels, # Final layer (=1)
        ]
        logging.info(f"During the k-layer 2D CNN stage, the numbers of channels are {channel_dims}")
        self.cnn_layers = torch.nn.ParameterList([])
        for li in range(n_layers):
            channel_dim_in  = channel_dims[li]
            channel_dim_out = channel_dims[li+1]

            new_layer = torch.nn.Conv2d(
                in_channels=channel_dim_in,
                out_channels=channel_dim_out,
                kernel_size=self.kernel_size,
                padding=padding_2d,
                padding_mode="zeros"
            )
            if init_mode == "he-normal":
                torch.nn.init.kaiming_normal_(new_layer.weight, nonlinearity="relu")
            self.cnn_layers.append(new_layer)

        self.relu = torch.nn.ReLU()
        self.angular_axis_last = angular_axis_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Assume x has shape (batch, self.n_in_channels, X, Y)
        Output will have shape (batch, self.n_out_channels, X, Y)

        If skip_connection is specified, then the slice (:, :-1, :, :)
        will be added to the output
        """
        if self.skip_connection:
            add_slice = x[:, -1].unsqueeze(1)

        for i, layer_i in enumerate(self.cnn_layers):
            x = apply_conv_with_polar_padding(
                layer_i,
                x,
                angular_axis_last=self.angular_axis_last
            )
            if i+1 < self.n_layers:
                x = self.relu(x)
        out = x

        if self.skip_connection:
            out = out + add_slice

        return out

class MFISNet_Refinement_Block(torch.nn.Module):
    """A single per-frequency refinement block of MFISNet-Refinement (MRef).
    Uses an FYNet variant (MFISNet-Fused) along with a 2D CNN to perform the
    update step (plus a skip connection).
    """
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
        big_init: bool=True,
        init_mode: str = None,
        use_cnns_2d: str = None,
        embedding_mode: str = None,
        N_emb_channels_out: int = 0,
        use_pred_d_mh: bool=True,
    ) -> None:
        """Initialize the MFISNet_Refinement_Block object
        The setup is as follows:
        1. Optional embedding block that maps information from q-hat to the FYNet component
        2. FYNet block that takes in [Fk[q], Fk[q]-dk, dk], as well as the embedded data if used
            Optionally skips the 2D CNN layers since there are 2D CNN layers in the udpate block
            that also have access to the existing estimate qhat
        3. "update cnn" that performs a 2D CNN with a) the output of the FYNet block and b) the
            estimate qhat from the previous block
        4. Add the update to the incoming prediction, and output the result
        """
        super().__init__()

        # Select which 2D CNNs to use
        use_cnns_2d = use_cnns_2d if use_cnns_2d is not None else "both"
        # embedding_mode = embedding_mode.lower() if embedding_mode is not None else "none"
        embedding_mode = "none" if (embedding_mode is None or N_emb_channels_out==0) \
            else embedding_mode.lower()
        self.use_embedding_block = embedding_mode in ["fynet-forward",]
        self.embedding_mode = embedding_mode

        self.use_fynet_cnn_2d  = use_cnns_2d in ["fynet", "both"]
        self.use_update_cnn_2d = use_cnns_2d in ["update", "both"]
        self.N_fynet_cnn_2d  = N_cnn_2d if self.use_fynet_cnn_2d else 0
        self.N_update_cnn_2d = N_cnn_2d if self.use_update_cnn_2d else 0
        self.N_cnn_1d = N_cnn_1d
        logging.info(f"Received use_cnns_2d={use_cnns_2d}, so use_fynet_cnn_2d = {self.use_fynet_cnn_2d} and use_update_cnn_2d={self.use_update_cnn_2d}")
        logging.info(f"Channel dimensions: cnn_1d: {self.N_cnn_1d}, FYNet 2D CNN: {self.N_fynet_cnn_2d}, Update 2D CNN: {self.N_update_cnn_2d}")

        # Save numbers
        self.N_h = N_h
        self.N_rho = N_rho
        self.N_freqs = N_freqs
        self.c_1d = c_1d
        self.c_2d = c_2d
        self.w_1d = w_1d
        self.w_2d = w_2d
        self.N_emb_channels_out = N_emb_channels_out

        if self.use_embedding_block:
            if self.embedding_mode == "fynet-forward":
                self.embedding_block = FYNetForward(
                    N_h,
                    N_rho,
                    c_1d,
                    w_1d,
                    N_cnn_1d,
                    N_channels_out=N_emb_channels_out,
                    init_mode=init_mode,
                )
            else:
                raise ValueError(f"Only accepting embedding_mode='fynet-forward' or 'none' at the moment.")

        self.use_pred_d_mh = use_pred_d_mh
        if self.use_pred_d_mh:
            num_fynet_channels_in = N_emb_channels_out + NUM_D_MH_CHANNELS*N_freqs
        else:
            num_fynet_channels_in = N_emb_channels_out + N_freqs

        self.fynet_block = MFISNet_Fused(
            N_h,
            N_rho,
            num_fynet_channels_in,
            c_1d,
            c_2d,
            w_1d,
            w_2d,
            N_cnn_1d,
            self.N_fynet_cnn_2d,
            big_init,
            init_mode=init_mode,
        )
        logging.info(f"The FYNet block (as MFISNet-Fused) has {self.fynet_block.N_cnn_2d} 2D CNN layers")

        # Note: always uses polar padding
        self.update_block = KLayer2DCNN(
            n_layers=self.N_update_cnn_2d,
            n_in_channels=2,
            n_out_channels=1,
            n_feature_channels=c_2d,
            kernel_size=w_2d,
            skip_connection=True,
        )
        logging.info(f"KLayer2DCNN has {self.update_block.n_layers} layers")

    def forward(self, x_tup: List) -> torch.Tensor:
        """Perform a MFISNet_Fused forward call on the input_d_mh portion of the x_tup argument
        Assumes shapes:
            [pred_q, input_d_mh] = x_tup
            pred_q: (N_batch, N_theta, N_rho)
                predicted scatterer from the previous frequency level
            input_d_mh: (N_batch, NUM_D_MH_CHANNELS, N_m, N_h, 2 (real/imag))
                some combination of (d_mh, F(pred_q), d_mh-F(pred_q))
        Returns:
            output: (N_batch, N_theta, N_rho)
        """
        pred_q, input_d_mh = x_tup # .elems

        # Prepare the inputs for FYNet
        # Run the embedding block if it is in use and insert its outputs into the FYNet inputs
        if self.use_embedding_block:
            emb_output = self.embedding_block(pred_q)
            fynet_input = torch.concatenate([
                emb_output,
                input_d_mh,
            ], dim=1)
        else:
            fynet_input = input_d_mh
        # Run FYNet
        intermediate_result = self.fynet_block(fynet_input)

        # After FYNet, run the update cnn to clean things up and stitch in the
        # current estimate's information
        if self.use_update_cnn_2d:
            # If using the update cnn
            cnn_input = torch.stack([intermediate_result, pred_q], dim=1)
            cnn_output = self.update_block(cnn_input).squeeze(1)
            output = cnn_output # Owen's code already adds on pred_q
        else:
            # If not using the update cnn, add the results directly
            output = intermediate_result + pred_q
        return output

    def __repr__(self) -> str:
        c_1d = self.fynet_block.c_1d
        c_2d = self.fynet_block.c_2d
        w_1d = self.fynet_block.w_1d
        w_2d = self.fynet_block.w_2d
        N_cnn_1d = self.fynet_block.N_cnn_1d
        s = (
            f"MFISNet-Refinement-Block model with {N_cnn_1d} "
            f"1D CNN layers ({c_1d} channels, {w_1d} modes), "
            f"{self.N_fynet_cnn_2d} 2D CNN layers "
            f"({c_2d} channels {c_2d} with {w_2d}x{w_2d} kernels). "
            f"Also contains a KLayer2DCNN with {self.N_update_cnn_2d} layers "
            f"sending {NUM_D_MH_CHANNELS} channels to 1 through {c_2d} intermediate channels. "
            f"Using embedding block: {self.use_embedding_block} (channel width: {self.N_emb_channels_out})"
        )
        return s

def load_MFISNet_Refinement_Block_from_state_dict(
    state_dict: dict,
    N_freqs: int,
    epoch_results_dd: dict,
    N_h: int,
    use_pred_d_mh: bool = True,
) -> MFISNet_Refinement_Block:
    """Sets up a MFISNet_Refinement_Block model from the given state dictionary
    and number of frequencies

    Args:
        state_dict (dict): state dictionary holding the model weights
        N_freqs (int): number of frequencies in use
        epoch_results_dd (dict): the logging information from the results file,
            at the selected epoch
        N_h (int): number of grid points for h
        use_pred_d_mh (bool): whether the model will take pred_d_mh as one of the inputs

    Returns:
        new_mfisnet_refinement_block_model (MFISNet_Refinement_Block): the model loaded with the weights
    """
    logging.info(f"epoch_results_dd: {epoch_results_dd.keys()}")
    logging.info(f"state_dict: {state_dict.keys()}")

    # First, compute hyperparameters from the state dictionary / results dict
    N_rho = epoch_results_dd["n_rho_vals"]
    c_1d = epoch_results_dd["n_cnn_channels_1d"]
    c_2d = epoch_results_dd["n_cnn_channels_2d"]
    w_1d = epoch_results_dd["kernel_size_1d"]
    w_2d = epoch_results_dd["kernel_size_2d"]
    N_cnn_1d = epoch_results_dd["n_cnn_1d"]
    N_cnn_2d = max(epoch_results_dd["n_update_cnn_2d"], epoch_results_dd["n_fynet_cnn_2d"])
    use_cnns_2d = epoch_results_dd["use_cnns_2d"]
    use_pred_d_mh = epoch_results_dd.get("use_pred_d_mh", use_pred_d_mh)
    if "embedding_mode" in epoch_results_dd.keys():
        embedding_mode = epoch_results_dd["embedding_mode"]
        N_emb_channels_out = epoch_results_dd["n_emb_channels_out"]
    else:
        embedding_mode = "none"
        N_emb_channels_out = 0

    logging.info(f"Loading MFISNet_Refinement_Block with the following settings...")
    logging.info(f"N_h={N_h}")
    logging.info(f"N_rho={N_rho}")
    logging.info(f"N_freqs={N_freqs}")
    logging.info(f"c_1d={c_1d}")
    logging.info(f"c_2d={c_2d}")
    logging.info(f"w_1d={w_1d}")
    logging.info(f"w_2d={w_2d}")
    logging.info(f"N_cnn_1d={N_cnn_1d}")
    logging.info(f"N_cnn_2d={N_cnn_2d}")
    logging.info(f"N_emb_channels_out={N_emb_channels_out}")
    logging.info(f"use_pred_d_mh={use_pred_d_mh}")

    # Next, initialize a model
    new_mfisnet_refinement_block_model = MFISNet_Refinement_Block(
        N_h=N_h,
        N_rho=N_rho,
        N_freqs=N_freqs,
        c_1d=c_1d,
        c_2d=c_2d,
        w_1d=w_1d,
        w_2d=w_2d,
        N_cnn_1d=N_cnn_1d,
        N_cnn_2d=N_cnn_2d,
        big_init=True, # just use this as a default value but it doesn't really matter
        use_cnns_2d=use_cnns_2d,
        embedding_mode=embedding_mode,
        N_emb_channels_out=N_emb_channels_out,
        use_pred_d_mh=use_pred_d_mh,
    )

    # Load in the values
    new_mfisnet_refinement_block_model.load_state_dict(state_dict=state_dict)
    return new_mfisnet_refinement_block_model
