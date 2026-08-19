# MFISNet-PDE-Solver-Refinement-v1
# Packages MFISNet_Fused and MFISNet-Refinement but with a little extra wrapping
# to handle the new inputs in the form of [Fk[q], Fk[q]-dk, dk]
# Also contains a new dataset type, TupleLinearData, for use with the new setup

import torch
import logging
from typing import List
import re

from src.models.MFISNet_Fused import MFISNet_Fused, load_MFISNet_Fused_from_state_dict
from src.models.MFISNet_Refinement import KLayer2DCNN
from src.models.FYNet import FYNetInverse, FYNetForward

from src.utils.conv_ops import (
    conv_in_fourier_space,
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

class MFISNet_pde_solver_refinement_v1(torch.nn.Module):
    """MFISNet variant using the output of a PDE Solver to perform
    progressive refinement steps. Uses an FYNet variant (MFISNet-Fused)
    along with a 2D CNN to perform the update step (plus a skip connection).
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
        merge_middle_freq_channels: bool,
        big_init: bool=True,
        polar_padding: bool = False,
        init_mode: str = None,
        use_cnns_2d: str = None,
        embedding_mode: str = None,
        N_emb_channels_out: int = 0,
        set_c1d_per_freq: bool=True,
        use_pred_d_mh: bool=True,
    ) -> None:
        """Initialize the MFISNet_pde_solver_refinement_v1 object
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
                    polar_padding=polar_padding,
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
            merge_middle_freq_channels,
            big_init,
            polar_padding,
            init_mode=init_mode,
            set_c1d_per_freq=set_c1d_per_freq,
        )
        logging.info(f"The FYNet block (as MFISNet-Fused) has {self.fynet_block.N_cnn_2d} 2D CNN layers")

        # Try adding a KLayer2DCNN
        # Note - always uses polar padding
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
            f"PDE Solver Refinement model with {N_cnn_1d} "
            f"1D CNN layers ({c_1d} channels, {w_1d} modes), "
            f"{self.N_fynet_cnn_2d} 2D CNN layers "
            f"({c_2d} channels {c_2d} with {w_2d}x{w_2d} kernels). "
            f"Also contains a KLayer2DCNN with {self.N_update_cnn_2d} layers "
            f"sending {NUM_D_MH_CHANNELS} channels to 1 through {c_2d} intermediate channels. "
            f"Using embedding block: {self.use_embedding_block} (channel width: {self.N_emb_channels_out})"
        )
        return s

def load_MFISNet_pde_solver_refinement_v1_from_state_dict(
    state_dict: dict,
    N_freqs: int,
    epoch_results_dd: dict,
    N_h: int,
    use_pred_d_mh: bool = True,
) -> MFISNet_pde_solver_refinement_v1:
    """Sets up a MFISNet_pde_solver_refinement_v1 model from the given state dictionary
    and number of frequencies; Also currently seems to require the polar padding

    Args:
        state_dict (dict): state dictionary holding the model weights
        N_freqs (int): number of frequencies in use
        epoch_results_dd (dict): the logging information from the results file,
            at the selected epoch
        N_h (int): number of grid points for h
        use_pred_d_mh (bool): whether the model will take pred_d_mh as one of the inputs

    Returns:
        new_mfisnet_psr_v1_model (MFISNet_pde_solver_refinement_v1): the model loaded with the weights
    """
    # Compensate for the inclusion of multiple inputs per frequency
    # global NUM_D_MH_CHANNELS

    logging.info(f"epoch_results_dd: {epoch_results_dd.keys()}")
    logging.info(f"state_dict: {state_dict.keys()}")

    # First, compute hyperparameters from the state dictionary
    layer_dims = {key: tuple(val.shape) for (key, val) in state_dict.items()}
    parameter_keys = list(state_dict.keys())

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
    mmfc = epoch_results_dd["merge_middle_freq_channels"]
    polar_padding = epoch_results_dd["polar_padding"]
    set_c1d_per_freq = epoch_results_dd["set_c1d_per_freq"] \
        if "set_c1d_per_freq" in epoch_results_dd.keys() else False

    logging.info(f"Loading MFISNet_pde_solver_refinement_v1 with the following settings...")
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
    logging.info(f"set_c1d_per_freq={set_c1d_per_freq}")
    logging.info(f"use_pred_d_mh={use_pred_d_mh}")

    # Next, initialize a model
    new_mfisnet_psr_v1_model = MFISNet_pde_solver_refinement_v1(
        N_h=N_h,
        N_rho=N_rho,
        N_freqs=N_freqs,
        c_1d=c_1d,
        c_2d=c_2d,
        w_1d=w_1d,
        w_2d=w_2d,
        N_cnn_1d=N_cnn_1d,
        N_cnn_2d=N_cnn_2d,
        merge_middle_freq_channels=mmfc,
        big_init=True, # just use this as a default value but it doesn't really matter
        polar_padding=polar_padding,
        use_cnns_2d=use_cnns_2d,
        embedding_mode=embedding_mode,
        N_emb_channels_out=N_emb_channels_out,
        set_c1d_per_freq=set_c1d_per_freq,
        use_pred_d_mh=use_pred_d_mh,
    )

    # Load in the values
    new_mfisnet_psr_v1_model.load_state_dict(state_dict=state_dict)
    return new_mfisnet_psr_v1_model
