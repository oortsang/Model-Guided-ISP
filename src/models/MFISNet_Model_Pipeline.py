# MFISNet_Model_Pipeline.py
# For combining several models into one pipeline
# Takes care of the PDE Solver calls and coordinate transforms if needed
# Also handles MFISNet-Refinement where each layer may have different
# hyperparameters

import numpy as np
import torch
import logging
from typing import List, Dict, Callable
import time

from solvers.integral_equation.helmholtz_solver_bicgstab import (
    setup_bicgstab_solver,
    HelmholtzSolverBicgstab,
    NP_CDTYPE, TORCH_CDTYPE, TORCH_RDTYPE
)
from solvers.integral_equation.helmholtz_solver_gradients import (
    PytorchPDESolver,
)

from src.data.data_transformations import (
    prep_polar_padder,
    polar_pad_and_apply,
    prep_conv_interp_2d,
    prep_rs_to_mh_interp,
    apply_interp_2d,
    get_scale_factor,
    CONST_RHO_PRIME,
    CONST_THETA_PRIME,
    prepare_polar_to_cart
)
# from src.training_utils.make_predictions import prepare_polar_to_cart

from src.models.MFISNet_Refinement_Block import (
    MFISNet_Refinement_Block,
    load_MFISNet_Refinement_Block_from_state_dict,
)
from src.models.MFISNet_Fused import (
    MFISNet_Fused,
    load_MFISNet_Fused_from_state_dict,
)

from src.utils.vram_info import get_memory_info, free_vram

# JAX/HPS stuff
import jax
import jax.numpy as jnp
import jaxlib
from solvers.hps.wave_scattering import (
    HPSScatteringSolver,
    PytorchHPSSolver,
    setup_hps_scattering_solver,
)


class MFISNet_Model_Pipeline(torch.nn.Module):
    """A class that allows for the combination of multiple
    MFISNet-Fused or MFISNet-PDE-Solver-Refinement objects
    """
    def __init__(
        self,
        blocks: List,
        solvers: List,
        freq_list: List[float],
        use_solver: bool = True,
        pde_solver_config: Dict = None,
        block_types: List[str] = None,
        N_x: int = None,
        N_rho: int = None,
        N_theta: int = None,
        N_h: int = None,
        x_vals: np.ndarray = None,
        rho_vals: np.ndarray = None,
        theta_vals: np.ndarray = None,
        h_vals: np.ndarray = None,
    ):
        """Initialize the Model Pipeline object
        Parameters:
            blocks (list of pytorch models): a list of neural network blocks,
                intended to be one per frequency used
            solvers (list of HelmholtzSolverBicgstab objects):
                Differentiable PDE solvers for each frequency
                Could optionally set this up inside the function in the future?
                ***Note this is one element shorter than the solvers list, since the PDE solver is
                    never used at the base frequency***
            freq_list (list of floats): the list of angular frequencies k of the expected data
            pde_solver_config (dictionary): configuration dictionary for the PDE solvers
                For now, just share the same configuration at every frequency.
                Refer to the HelmholtzSolverBicgstab code for the relevant keys.
            block_types (list of strings): optionally indicate what each type of block is
                for later display/debugging convenience...
            N_x, N_rho, N_theta, N_h (all ints):
                grid dimensions for use setting up the coordinate transforms
                these are for use with the PDE solver
            x_vals, rho_vals, theta_vals, h_vals (all np.ndarrays):
                grid point values; can be used to override the default grids
        """
        super().__init__()

        self.N_x     = N_x
        self.N_rho   = N_rho
        self.N_theta = N_theta
        self.N_m     = N_theta
        self.N_h     = N_h

        self.blocks  = torch.nn.ParameterList(blocks)
        self.solvers = solvers
        self.freq_list = freq_list
        self.N_freqs   = len(freq_list)
        self.use_solver = use_solver
        self.pde_solver_config = pde_solver_config if pde_solver_config is not None else dict()
        self.pde_solver_type = self.pde_solver_config.get("solver_type", None)
        self.block_types = block_types

        # Set up coordinate transform tools
        # but only if using the solver
        if self.use_solver:
            # Grids to use...
            self.h_vals = np.linspace(-np.pi/2, np.pi/2, N_h, endpoint=False) \
                if h_vals is None else h_vals
            # self.m_vals = np.linspace(-pi, pi, N_theta, endpoint=False) \
            #     if m_vals is None else m_vals
            self.theta_vals = np.linspace(0, 2*np.pi, N_theta, endpoint=False) \
                if theta_vals is None else theta_vals
            self.rho_vals = np.linspace(0, 0.575, N_rho, endpoint=False) \
                if rho_vals is None else rho_vals
            self.x_vals = np.linspace(-0.5, 0.5, N_x, endpoint=False) \
                if theta_vals is None else theta_vals

            # First, (r,s) to (m,h)
            conv_rs_to_m, conv_rs_to_h = prep_rs_to_mh_interp(
                self.theta_vals,  # r grid points
                self.theta_vals,  # s grid points
                N_theta,
                len(self.h_vals),
                a_neg_half=True,
            )
            self.torch_rs_to_m = torch.tensor(
                conv_rs_to_m.todense(), dtype=TORCH_CDTYPE, requires_grad=False,
            )
            self.torch_rs_to_h = torch.tensor(
                conv_rs_to_h.todense(), dtype=TORCH_CDTYPE, requires_grad=False,
            )

            def rs_to_mh_fn(d_rs_i):
                # nonlocal N_m, N_h, torch_rs_to_m, torch_rs_to_h
                return apply_interp_2d(
                    self.torch_rs_to_m, self.torch_rs_to_h, d_rs_i
                ).reshape(self.N_m, self.N_h)

            d_mh_scale_factor = get_scale_factor(CONST_RHO_PRIME, CONST_THETA_PRIME)

            self.rs_to_mh_fn = rs_to_mh_fn
            self.d_mh_scale_factor = d_mh_scale_factor

            # Second, polar to cart
            # Code based on the prepare_polar_to_cart function in
            # src/training_utils/make_predictions.py
            center = np.zeros(2)
            data_grid_xy = (
                np.array(np.meshgrid(self.x_vals, self.x_vals))
                .transpose(1, 2, 0)
                .reshape(N_x**2, 2)
            )
            cart_grid_radii = np.sqrt(data_grid_xy[:, 0] ** 2 + data_grid_xy[:, 1] ** 2)
            cart_grid_thetas = np.mod(
                np.arctan2(data_grid_xy[:, 1] - center[1], data_grid_xy[:, 0] - center[0]),
                2 * np.pi,
            )
            self.cart_grid_polar_coords = np.array([cart_grid_thetas, cart_grid_radii]).T

            padded_rho_vals, polar_padder = prep_polar_padder(
                self.rho_vals, N_theta, dim=1, with_torch=True
            )
            polar_to_x, polar_to_y = prep_conv_interp_2d(
                self.theta_vals,
                padded_rho_vals,
                self.cart_grid_polar_coords,
                bc_modes=("periodic", "extend"),
                a_neg_half=True,
            )
            self.torch_polar_to_x = torch.tensor(
                polar_to_x.todense(), dtype=TORCH_RDTYPE,
            )
            self.torch_polar_to_y = torch.tensor(
                polar_to_y.todense(), dtype=TORCH_RDTYPE,
            )
            def my_polar_to_cart(polar_data: torch.Tensor, self_obj=self) -> torch.Tensor:
                """Helper function to send polar grid data to cartesian grid data
                Note that this helper function also takes care of reshaping
                """
                nonlocal self
                in_shape = polar_data.shape
                res = polar_pad_and_apply(
                    polar_padder,
                    self.torch_polar_to_x,
                    self.torch_polar_to_y,
                    polar_data.reshape(-1, self.N_theta, self.N_rho),
                    batched=True,
                ).reshape(
                    *in_shape[:-2], self.N_x, self.N_x
                    # -1, self.N_x, self.N_x
                )
                return res

            self.polar_to_cart_fn = my_polar_to_cart

    def to(self, device):
        """Override the base to() function to make sure the
        interpolation operators are moved properly
        """
        moved_model = super(MFISNet_Model_Pipeline, self).to(device)

        for i in range(len(moved_model.blocks)):
            moved_model.blocks[i] = moved_model.blocks[i].to(device)

        if "torch_rs_to_m" in self.__dict__.keys():
            moved_model.torch_rs_to_m = moved_model.torch_rs_to_m.to(device)
            moved_model.torch_rs_to_h = moved_model.torch_rs_to_h.to(device)

        if "torch_polar_to_x" in self.__dict__.keys():
            moved_model.torch_polar_to_x = moved_model.torch_polar_to_x.to(device)
            moved_model.torch_polar_to_y = moved_model.torch_polar_to_y.to(device)

        return moved_model

    def forward(self, dk_stack: torch.Tensor, return_tmp_vals: bool = False) -> torch.Tensor:
        """Forward pass of the pipeline
        Expected input shape: (N_batch, N_freqs, N_m, N_2, 2)
        Assumes that the first block is always an fynet-type block

        Sketch of the function:
        Iterate over each of the frequencies:
            For each scattering object in the batch, run the PDE solver if needed
            Run the block corresponding to the current frequency
        Output the final block's outputs, and optionally include the temporary values
        from each frequency

        Args:
            dk_stack (torch.Tensor): input measurement data with shape ~(N_batch, N_freqs, N_m, N_h, 2)
            return_tmp_vals (bool): determine whether to also return the qhat and Fk[qhat] values from each frequency
        Returns:
            qhat_polar_t (torch.Tensor): the predicted scattering objects for the batch
            qhat_polar_list (if return_tmp_vals=True; torch.Tensor): a list of the provisional estimates (in polar)
                stacked into a single pytorch tensor
            pde_output_rs_list (if return_tmp_vals=True; torch.Tensor): a list of the PDE outputs (in (r,s) coordinates
                and without the constant scaling factor used in our (m, h) arrays)
                stacked into a single pytorch tensor
        """
        # First, do an FYNet operation for the base frequency
        qhat_polar_t = self.blocks[0](dk_stack[:, 0, ...])
        qhat_polar_list = [qhat_polar_t.clone()]
        pde_output_rs_list = []


        # Then, iterate for each frequency
        for t in range(1, self.N_freqs):
            # Run the PDE solver if needed
            dkt = dk_stack[:, t, ...]
            PDESolverFunc = (
                PytorchPDESolver
                if self.pde_solver_config.get("solver_type", "ls") == "ls"
                else PytorchHPSSolver
            )
            if self.use_solver:
                # 1. Convert from polar to cartesian
                # Do the first part with CDTYPE, which may be double precision
                # qhat_cart_t = self.polar_to_cart_fn(qhat_polar_t.to(TORCH_RDTYPE))
                # 2. Apply the PDE solver to each scattering object...
                pde_output_rs_t_list = []
                pde_output_mh_t_list = []
                # for i, qhat_t_i in enumerate(qhat_cart_t):
                for i, qhat_polar_t_i in enumerate(qhat_polar_t):
                    qhat_t_i = self.polar_to_cart_fn(qhat_polar_t_i.to(TORCH_RDTYPE))
                    t0 = time.perf_counter()
                    pde_output_rs_t_i = PDESolverFunc.apply(
                        qhat_t_i, self.solvers[t-1], self.pde_solver_config
                    )
                    t1 = time.perf_counter()

                    # Convert from (r,s) to (m,h), which also involves a constant factor
                    # and saving a real tensor in single precision
                    pde_output_mh_t_i = torch.view_as_real(
                        self.d_mh_scale_factor * self.rs_to_mh_fn(pde_output_rs_t_i)
                    ).to(torch.float32)

                    # convert back to float32 since it will be fed to the neural network later
                    pde_output_rs_t_list.append(pde_output_rs_t_i)
                    pde_output_mh_t_list.append(pde_output_mh_t_i)
                pde_output_rs_list.append(torch.stack(pde_output_rs_t_list, dim=0))
                pde_output_mh_t = torch.stack(pde_output_mh_t_list, dim=0)

                # Prepare the inputs for the FYNet-like block in the refinement block
                extra_inputs_t = torch.stack(
                    [pde_output_mh_t, pde_output_mh_t-dkt, dkt],
                    dim=1
                )
                input_kt = [qhat_polar_t, extra_inputs_t]
            else:
                # If not using the PDE solver, just prepare the inputs
                input_kt = [qhat_polar_t, dkt[:, np.newaxis, ...]]

            # Run the current block
            qhat_polar_tp1 = self.blocks[t](input_kt)

            # At the end of this frequency step...
            # Save the provisional estimate to the list and update the current estimate
            qhat_polar_list.append(qhat_polar_tp1.detach().clone())
            qhat_polar_t = qhat_polar_tp1

        # Prepare extra return values if requested
        # (these are the qhat estimates (in polar) and PDE outputs (in rs coordinates))
        if return_tmp_vals:
            res = (
                qhat_polar_t,
                torch.stack(qhat_polar_list, dim=1),
                torch.stack(pde_output_rs_list, dim=1) if len(pde_output_rs_list)>0 else None,
            )
        else:
            res = qhat_polar_t
        return res

    # def __str__(self):
    #     """Use for printing out information"""

def save_MFISNet_Model_Pipeline_by_block(
    model_pipeline: MFISNet_Model_Pipeline,
    block_fp_list: List[str],
):
    """Helper function to save each of the blocks from the MFISNet_Model_Pipeline
    to a collection of state dictionaries.
    Note that each of the constituent blocks is saved independently.
    Args:
        model_pipeline (MFISNet_Model_Pipeline): the model in question
        block_fp_list (list of strings): a list of the file paths where each block should be stored
    """
    for i, block_fp in enumerate(block_fp_list):
        block = model_pipeline.blocks[i]
        # block_type = model_pipeline.block_types[i]
        torch.save(block.state_dict(), block_fp)


def load_MFISNet_Model_Pipeline_from_state_dict(
    selected_hyperparams_dict: Dict,
    device: torch.cuda.device,
    N_x: int,
    N_h: int = None, # optional unless there is only FYNet
    pde_solver_config: Dict = None,
    prepare_half_grid: bool = True,
    rho_vals: np.ndarray = None,
    **kwargs,
) ->  MFISNet_Model_Pipeline:
    """Load the MFISNet_Model_Pipeline object based on the selected hyperparameter dictionary
    If needed, this function also prepares PDE solvers
    The MFISNet_Model_Pipeline already prepares coordinate transform code if needed.

    Expected contents include:
    - hyperparameters
    - block type (e.g., "fynet" or "mpsr" or "mref")
    - each block's parameter file path
    Args:
        selected_hyperparams_dict (dict): the selected hyperparameters, from one of the results yaml files
        device (torch.cuda.device): the device to use in loading the model and setting up the PDE solvers
        N_x (int): number of grid points in the spatial domain
        N_h (int): number of grid points for h in the wave field measurements
        pde_solver_config (dict): the settings used to call the PDE solver
        prepare_half_grid (bool): whether to prepare the PDE solver for the half-sized grid for use warm-starting
            the linear system solves later on
        rho_vals (np.ndarray): polar grid values, to allow overriding the default range
        Miscellaneous keyword arguments:
            spatial_domain_max (float): the largest grid point of the spatial domain, which is assumed
                to be symmetric about zero; defaults to 0.5 (so the domain is [-0.5, 0.5] x [-0.5, 0.5])
            receiver_radius (float): radius of the receiver ring, for use with the PDE solver
    Returns:
        model_pipeline (MFISNet_Model_Pipeline): a model containing a series of blocks, loaded from disk, along with
            PDE solvers if needed
    """
    N_freqs = len([k for k in selected_hyperparams_dict.keys() if "freq_idx" in k])
    freq_list = []

    N_rho = selected_hyperparams_dict["freq_idx_1"]["n_rho_vals"]
    N_theta = selected_hyperparams_dict["freq_idx_1"]["n_theta_vals"]
    N_m = N_theta
    blocks = []
    solvers = []
    use_solver = False
    block_types = []
    jax_device = kwargs.get("jax_device", jax.devices()[0])

    # Iterate through the different entries
    for fi in range(1, 1+N_freqs):
        key = f"freq_idx_{fi}"
        shd_fi = selected_hyperparams_dict[key]
        # First, identify whether this is fynet, mpsr, or mref
        # shd_fi["block_type"]
        block_type = shd_fi["block_type"].strip().lower()
        model_fp = shd_fi["central_model_fp"]
        state_dict = torch.load(model_fp, map_location=device)
        # Assume that the nu_list is a string of a singleton list
        # like '[4.0]'
        nu_val = float(shd_fi["source_nu_list"].strip("[]"))
        k = nu_val * 2*np.pi
        freq_list.append(k)
        block_types.append(block_type)

        # Next, load the block according to type as block_fi
        if block_type == "fynet":
            # Load FYNet / MFISNet-Fused
            block_fi = load_MFISNet_Fused_from_state_dict(
                state_dict,
                1, # N_freqs -- always using it as 1 in this case
            )
        elif block_type == "mpsr" or block_type == "psr" \
             or block_type == "pde-solver-refinement":
            # Prepare the solver
            use_solver = True
            # Set up any PDE solvers, if necessary
            spatial_domain_max = kwargs.get("spatial_domain_max", 0.5)
            receiver_radius = kwargs.get("receiver_radius", 100)
            N_h = shd_fi["n_h_vals"]

            # Call the setup function and add to the list of solvers
            logging.info(f"Starting to set up the differentiable solver")
            # Check which solver type to use...
            solver_type = pde_solver_config["solver_type"]
            logging.info(f"Requested solver type {solver_type}")
            if solver_type == "hps":
                solver_obj = setup_hps_scattering_solver(
                    N_x, spatial_domain_max, k, receiver_radius,
                    hps_sd_int_mat_dir=pde_solver_config["hps_sd_mat_dir"],
                    hps_l=pde_solver_config["hps_l"],
                    hps_p=pde_solver_config["hps_p"],
                    hps_comp_domain_factor=pde_solver_config["hps_comp_domain_factor"],
                    device=jax_device,
                )
                logging.info(f"Finished setting up the HPS solver")
            else:
                solver_obj = setup_bicgstab_solver(
                    N_x, spatial_domain_max, nu_val, receiver_radius, device=device,
                    prepare_half_grid=True,
                )
                logging.info(f"Finished setting up the differentiable LS solver")
            solvers.append(solver_obj)

            # Load PDE-Solver-Refinement (uses the same block class as MFISNet-Refinement)
            block_fi = load_MFISNet_Refinement_Block_from_state_dict(
                state_dict,
                1, # N_freqs -- always using it as 1 in this case
                epoch_results_dd=shd_fi,
                N_h=shd_fi["n_h_vals"],
                use_pred_d_mh=True,
            )
        elif block_type == "mref" or block_type == "mfisnet-refinement":
            # Load MFISNet-Refinement
            block_fi = load_MFISNet_Refinement_Block_from_state_dict(
                state_dict,
                1, # N_freqs -- always using it as 1 in this case
                epoch_results_dd=shd_fi,
                N_h=shd_fi["n_h_vals"],
                use_pred_d_mh=False,
            )
        else:
            raise KeyError(
                f"block_type {block_type} not recognized; expecting "
                f"one of fynet, mpsr, and mref"
            )
        blocks.append(block_fi)

    # Assemble everything
    # already have blocks and solvers and freq_list
    model_pipeline = MFISNet_Model_Pipeline(
        blocks,
        solvers,
        freq_list,
        use_solver=use_solver,
        pde_solver_config=pde_solver_config,
        block_types=block_types,
        N_x = N_x,
        N_rho = N_rho,
        N_theta = N_theta,
        N_h = N_h,
        rho_vals = rho_vals,
    ).to(device)

    return model_pipeline
