# Fine-tuning the MFISNet_Model_Pipeline object
# Updates in an end-to-end training regime
# and expects each block to be already trained to some degree

# Mostly standard imports
import numpy as np
import torch
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import wandb
import shutil, psutil
import os, sys, glob
import time
import logging
import copy
from typing import Tuple, Callable, Dict
import argparse

# PDE solver
from solvers.integral_equation.HelmholtzSolverDifferentiable import (
    setup_differentiable_solver,
    HelmholtzSolverDifferentiable,
    PytorchPDESolver,
    NP_CDTYPE, TORCH_CDTYPE, TORCH_RDTYPE,
)

# Models/dataset
from src.models.MFISNet_pde_solver_refinement_v1 import (
    MFISNet_pde_solver_refinement_v1,
    load_MFISNet_pde_solver_refinement_v1_from_state_dict,
)
# from train_MFISNet_Fused import setup_single_dataset, LinearData

from src.models.MFISNet_Model_Pipeline import (
    MFISNet_Model_Pipeline,
    save_MFISNet_Model_Pipeline_by_block,
    load_MFISNet_Model_Pipeline_from_state_dict,
)
# Data loading/preparation
from src.data.add_noise import add_noise_to_d
from src.data.data_io import load_hdf5_to_dict, load_multifreq_dataset
from src.data.data_naming_constants import (
    Q_POLAR,
    Q_CART,
    D_MH,
    D_RS,
    Q_POLAR_LPF,
    Q_CART_LPF,
    NU_SF,
    OMEGA_SF,
    KEYS_FOR_TRAINING_SAMPLES_ALL,
    FREQ_DEPENDENT_KEYS,
    TRUNCATABLE_KEYS,
)
from src.data.data_transformations import (
    prep_polar_padder,
    polar_pad_and_apply,
    prep_conv_interp_2d,
    prep_rs_to_mh_interp,
    apply_interp_2d,
    # get_scale_factor,
    # CONST_RHO_PRIME,
    # CONST_THETA_PRIME,
    CONST_D_MH_SCALE_FACTOR,
    prepare_polar_to_cart,
)
from src.data.datasets import (
    FullData,
    setup_dataset_full,
)

# Training
from src.training_utils.train_loop import (
    evaluate_losses_on_dataloader,
    evaluate_losses_on_dataloader_with_cartesian,
)
from src.training_utils.loss_functions import MSEModule
# from src.training_utils.make_predictions import prepare_polar_to_cart

# Misc. utilities
# from src.utils.plotting_utils import plot_row
from src.utils.vram_info import (
    get_memory_info,
    free_vram,
    get_vram_total_mb,
    vram_mb_to_frac,
)

from src.utils.logging_utils import (
    write_result_to_file,
    find_best_epoch,
    load_field_in_yaml_file,
    load_yaml_to_dict,
    save_dict_to_yaml,
    FMT, TIMEFMT,
    hash_dict
)

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    bool_choices = ["true", "false"]

    ### Data loading/saving options ###
    parser.add_argument("--data_input_nus", type=str, nargs="+")
    parser.add_argument(
        "--data_dir_base",
        type=str,
        help="For the reference dataset, indicate the directory containing all the "
        "measurement folders corresponding to the relevant frequencies and data subsets",
    )
    parser.add_argument("--truncate_num", type=int)
    parser.add_argument("--truncate_num_val", type=int)
    parser.add_argument("--log_train_subset_frac", type=float, default=1.0)
    # Caution -- will need to apply noise to d_rs first if the PDE solver is used,
    # then d_mh later
    parser.add_argument(
        "--noise_to_signal_ratio", default=None, type=float
    )
    parser.add_argument("--use_noise_seed", choices=["true", "false"], default="false")
    parser.add_argument("--noise_seed_list_train",  type=int, nargs="*", default=None)
    parser.add_argument("--noise_seed_list_val",  type=int, nargs="*", default=None)

    parser.add_argument(
        "--out_train_results_fp",
        type=str,
        help="File path to the saved logging information",
    )
    parser.add_argument(
        "--out_model_dir",
        type=str,
        help="Directory where to put updated model parameters",
    )
    parser.add_argument(
        "--in_central_results_fp",
        type=str,
        help="The results yaml containing relevant model paths and hyperparameters",
    )
    parser.add_argument(
        "--out_central_results_fp",
        type=str,
        help="Generate a centralized results yaml file similar to in_central_results_fp",
    )

    # Option to use smoothed targets or not
    # alternate option
    parser.add_argument(
        "--use_targets", choices=["original", "smoothed", "legacy"], default="legacy",
        help=(
            "Set target as original or smoothed; alternatively, set to legacy to use "
            "either --used_smoothed_targets or --use_original_targets."
        )
    )
    parser.add_argument("--use_smoothed_targets", default=False, action="store_true")
    parser.add_argument("--use_original_targets", action="store_false", dest="use_smoothed_targets")


    ### Optimization settings ###
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument(
        "--lr_init_base", type=float, default=1.0,
        help="Base initial learning rate (for batch size 16); adjusted based on the batch size"
    )
    parser.add_argument(
        "--weight_decay_base", type=float, default=0.0,
        help="Base weight-decay rate (for batch size 16); adjusted based on the batch size"
    )
    parser.add_argument("--eta_min_base", type=float, default=1e-04)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=98283)
    parser.add_argument("--d_rs_loss_weight", type=float, default=0.0)
    # not really an optimization parameter but could potentially affect the run
    # in the future...

    ### Register PDE-Solver options as well (for use with PSR blocks) ###
    parser.add_argument(
        "--pde_fwd_linsys_solver",
        choices=["bicgstab", "gmres"], default="bicgstab"
    )
    parser.add_argument(
        "--pde_adj_linsys_solver",
        choices=["bicgstab", "gmres", "same-as-fwd"], default="same-as-fwd"
    )
    parser.add_argument("--pde_fwd_rtol", type=float, default=1e-2)
    parser.add_argument("--pde_adj_rtol", type=float, default=1e-2)
    parser.add_argument("--pde_fwd_use_half_grid", choices=bool_choices, default="true")
    parser.add_argument("--pde_adj_use_half_grid", choices=bool_choices, default="false")
    parser.add_argument("--pde_fwd_half_grid_tol_ratio", type=float, default=0.5)
    parser.add_argument("--pde_adj_half_grid_tol_ratio", type=float, default=0.5)
    parser.add_argument("--pde_max_iter", type=int, default=1000)
    parser.add_argument("--pde_spatial_domain_max", type=float, default=0.5)
    parser.add_argument("--pde_receiver_radius", type=float, default=100)
    parser.add_argument("--pde_batch_size", type=int, default=100)
    parser.add_argument("--pde_restart", type=int, default=10)

    # HPS settings
    parser.add_argument(
        "--pde_solver_type", choices=["ls", "hps"], default="ls",
        help="Solver type: Lippmann-Schwinger ('ls') or Hierarchical Poincare-Steklov ('hps'). "
        "Defaults to the Lippmann-Schwinger solver."
    )
    parser.add_argument("--pde_hps_l", type=int, help="Number of HPS quadtree levels")
    parser.add_argument(
        "--pde_hps_p", type=int,
        help="Polynomial order of leaf-level Chebyshev grid"
    )
    parser.add_argument(
        "--pde_hps_comp_domain_factor", type=float, default=1.,
        help="How much larger of a computational domain to use for the HPS quadtree"
    )
    parser.add_argument(
        "--pde_hps_sd_mat_dir", type=str,
        default="rlc_data/HPS_SD_matrices",
        help="Directory where to find the interior S and D scattering matrix files",
    )
    parser.add_argument(
        "--jax_mem_alloc_mb", type=int,
        default=3072,
        help="Amount of VRAM to allocate to JAX, in MB"
    )

    # Extras
    ### Logging options ###
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--n_batches_per_log", type=int, default=50)
    parser.add_argument("--n_epochs_per_log",  type=int, default=0)
    parser.add_argument("--save_all_model_weights", choices=bool_choices, default="true")
    parser.add_argument("--selection_field", default="eval_rel_l2")
    parser.add_argument("--selection_mode", default="min", choices=["min", "max"])
    parser.add_argument("--verbose_level", default=0, type=int)

    # Weights and Biases setup
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, help="The W&B entity")
    parser.add_argument(
        "--wandb_mode", choices=["offline", "online", "disabled"], default="offline"
    )

    a = parser.parse_args()

    # Parse boolean arguments + misc. arguments that need extra processing
    a.pde_fwd_use_half_grid = (a.pde_fwd_use_half_grid == "true")
    a.pde_adj_use_half_grid = (a.pde_adj_use_half_grid == "true")
    a.save_all_model_weights = (a.save_all_model_weights == "true")

    # Override unless use_targets="legacy"
    if a.use_targets == "smoothed":
        a.use_smoothed_targets = True
    elif a.use_targets == "original":
        a.use_smoothed_targets = False

    if a.pde_adj_linsys_solver == "same-as-fwd":
        a.pde_adj_linsys_solver = a.pde_fwd_linsys_solver
    return a

def train_pipeline_e2e(
    model_pipeline: MFISNet_Model_Pipeline,
    train_dloader,
    eval_dloader,
    loss_function,
    log_function: Callable,
    device: torch.cuda.Device,
    solver_obj: HelmholtzSolverDifferentiable,
    pde_solver_config: Dict,
    polar_to_cart_fn: Callable,
    rs_to_mh_fn: Callable,
    d_rs_loss_weight: float = 1,
    lr_init: float = 1e-4,
    eta_min: float = 1e-4,
    weight_decay: float = 0,
    log_every_n: int = 5,
    log_by_epoch: bool = True,
    num_epochs: int  = 1,
    verbose_level: int = 0,
) -> MFISNet_Model_Pipeline:
    """Fine tune the MFISNet-Model_Pipeline model; technically I was advised to call this "end-to-end training" after writing the code
    in order to avoid ambiguities with finetuning in sense of using a different dataset/data distribution or an alternate sense having
    to do with selecting (hyper)parameters.

    Note that, at the moment, the PDE solver is quite slow, so the logging is done based on batches rather than epochs

    Arguments:
        model_pipeline (MFISNet_Model_Pipeline): the base model, with weights loaded
        train_dloader (torch.utils.data.DataLoader): data loader for the training set
        eval_dloader (torch.utils.data.DataLoader): data loader for the evaluation/validation set
        loss_function (MSEModule or some other loss function module): the loss function object that computes
            the loss function but also helps to computer other error statistics
        log_function (Callable): a function that performs the logging tasks periodically and evaluates the model
        device (torch.cuda.device): the cuda device to use for this training process
        solver_obj (HelmholtzSolverDifferentiable): the differentiable PDE solver in case the d_rs loss term is in use
        polar_to_cart_fn (Callable): coordinate transform that moves an object from the polar grid to the cartesian grid
        rs_to_mh_fn (Callable): coordinate transform that moves an object from the (r, s) grid to the (m, h) grid
        d_rs_loss_weight (float): weight for the d_rs loss term
        lr_init (float): initial learning rate
        eta_min (float): minimum learning rate
        weight_decay (float): weight decay
        log_every_n (int): the number of batches between logging
        num_epochs (int): the number of epochs to train for
        verbose_level (int): level of verbosity for the outputs
    Returns:
        model_pipeline (MFISNet_Model_Pipeline): the updated model
    """
    model_pipeline = model_pipeline.train()

    optimizer = torch.optim.AdamW(
        model_pipeline.parameters(),
        lr_init,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=eta_min,
    )

    logging.info(f"Starting training...")
    t0 = time.perf_counter()
    model_pipeline = model_pipeline.to(device)
    epoch_stagger = 0
    num_batches_per_epoch = len(train_dloader)
    logged_batch_idcs = []
    train_polar_rel_errs = []
    train_cart_rel_errs = []
    eval_polar_rel_errs = []
    eval_cart_rel_errs = []
    # Helper functions to deal with the fact that the datasets may contain different
    # quantities, depending on the setting that we are working in
    def move_helper(obj, target):
        if isinstance(obj, list):
            out = [elem.to(target) for elem in obj]
        else:
            out = obj.to(target)
        return out
    def unpack_helper(db):
        if isinstance(db, list):
            out1 = db[0]
            out2 = db[1]
        else:
            out1 = db
            out2 = None
        return out1, out2

    for epoch in range(num_epochs):
        logging.info(f"({time.perf_counter()-t0:.2f}s) Epoch {epoch}")
        get_memory_info()
        for batch_idx, data_batch in (enumerate(train_dloader, start=0)):
            # logging.info(f"({time.perf_counter()-t0:.2f}s) Batch {batch_idx};")
            # import pdb; pdb.set_trace()
            data_batch = [move_helper(db_entry, device) for db_entry in data_batch]
            x_mh, x_rs = unpack_helper(data_batch[0])
            y_p, y_c   = unpack_helper(data_batch[1])
            yf_p, yf_c = unpack_helper(data_batch[2])

            optimizer.zero_grad()

            pde_solve_time = 0
            t1 = time.perf_counter()

            # Call the model and unpack the extras (temporary values)
            all_outputs = model_pipeline(x_mh, return_tmp_vals=True)
            model_output = all_outputs[0]
            extra_outputs_qhat  = all_outputs[1]
            extra_outputs_Fqhat = all_outputs[2] # if model_pipeline.use_solver else None # redundant

            # Compute the loss function
            # First, the MSE of the q outputs
            q_loss_term = loss_function(model_output, y_p)
            loss_val = q_loss_term
            # Next, add on the PDE solver loss terms if relevant
            qhat_list = []
            Fk_qhat_list = []
            PDESolverFunc = (
                PytorchPDESolver
                if pde_solver_config.get("solver_type", "ls") == "ls"
                else PytorchHPSSolver
            )
            if d_rs_loss_weight != 0:
                d_rs_loss_term = 0
                for i in range(model_output.shape[0]):
                    # Convert the prediction to polar, then run the PDE solver
                    qhat_i = polar_to_cart_fn(model_output[i]).squeeze(0)
                    Fk_q_pred_i = PDESolverFunc.apply(
                        qhat_i.to(TORCH_CDTYPE),
                        solver_obj,
                        pde_solver_config
                    )

                    # Compute the MSE in scattered wave measurement space
                    d_rs_loss_term += torch.mean(torch.abs(
                        torch.view_as_real(Fk_q_pred_i.to(torch.cfloat)) - x_rs[i, -1]
                    )**2)
                    qhat_list.append(qhat_i.detach().cpu().numpy())
                    Fk_qhat_list.append(Fk_q_pred_i.detach().cpu().numpy())
                loss_val = loss_val + d_rs_loss_weight * d_rs_loss_term

                if verbose_level >= 1:
                    logging.info(
                        f"({time.perf_counter()-t0:.2f}s) Batch {batch_idx}; "
                        f" batch_loss = {loss_val:.3e} = {q_loss_term:.3e}+"
                        f"{d_rs_loss_weight:.3e}*{d_rs_loss_term:.3e}"
                    )
            else:
                if verbose_level >= 2:
                    logging.info(
                        f"({time.perf_counter()-t0:.2f}s) Batch {batch_idx}; "
                        f"batch loss = {loss_val:.3e}"
                    )

            # In case of NaN, save information to disk
            if torch.isnan(loss_val):
                logging.info(f"loss val is nan!")
                # Save the extra outputs as well as x, y, y_final... also
                # find a random place to save it
                tmp_seed = int(float(str(time.perf_counter())[::-1]))
                rng = np.random.default_rng(tmp_seed)
                tag = "0x"+"".join([hex(x)[2:] for x in rng.integers(256, size=6)])
                date_str = time.strftime("%Y-%m-%d_%H-%M_", time.localtime())
                debug_fp = f"scratch_dir/{date_str}_debug_solver_nan_{tag}.npz"
                debug_contents = {
                    "x_mh": x_mh.detach().cpu().numpy(),
                    "y_p": y_p.detach().cpu().numpy(),
                    "y_final": yf_p.detach().cpu().numpy(),
                    "model_output": model_output.detach().cpu().numpy(),
                    "extra_outputs": extra_outputs_qhat.detach().cpu().numpy(),
                    "qhat_list": np.array(qhat_list),
                    "Fk_qhat_list": np.array(Fk_qhat_list),
                }
                msg = (
                    f"(epoch {epoch}, batch {batch_idx}) NaN encountered in the loss val. "
                    f"See {debug_fp} file for the inputs/outputs causing this error."
                )
                print(msg)
                logging.info(msg)
                np.savez(
                    debug_fp,
                    **debug_contents,
                )
                raise RuntimeError(msg)

            # Evaluate the backward pass and take the optimization step
            # Skip for the very first sample though
            loss_val.backward()
            is_first_batch = (batch_idx==0 and epoch == 0)

            # For the first batch, make sure to do the logging first
            # so that we get a better idea of the baseline we're starting from
            if not is_first_batch:
                optimizer.step()

            # Run the logging function and also print out the running errors since the code may not get to the end
            # Batch-based logging mode
            is_last_batch = (epoch+1 == num_epochs and (batch_idx+1) ==  len(train_dloader))
            if (not log_by_epoch) and ((batch_idx % log_every_n == 0) or is_last_batch):
                # do some logging stuff
                logging.info(f"({time.perf_counter()-t0:.2f}s, batch {batch_idx}) Logging!")
                res = log_function(
                    model_pipeline,
                    epoch,
                    batch_idx,
                    num_epochs,
                )
                train_polar_rel_errs.append(res[0])
                train_cart_rel_errs.append(res[1])
                eval_polar_rel_errs.append(res[2])
                eval_cart_rel_errs.append(res[3])
                logged_batch_idcs.append(num_batches_per_epoch*epoch + batch_idx)
                logging.info(
                    f"Current logging status...\n"
                    f"logged_batch_idcs = {logged_batch_idcs}\n"
                    f"train_polar_rel_errs = {train_polar_rel_errs}\n"
                    f"train_cart_rel_errs = {train_cart_rel_errs}\n"
                    f"eval_polar_rel_errs = {eval_polar_rel_errs}\n"
                    f"eval_cart_rel_errs = {eval_cart_rel_errs}"
                )
                logging.info(f"({time.perf_counter()-t0:.2f}s) logging done")

            # Take the first step after logging
            if is_first_batch:
                optimizer.step()

        # Epoch-based logging mode
        is_last_epoch = (epoch+1 == num_epochs)
        if log_by_epoch and (epoch % log_every_n == 0) or is_last_epoch:
                # do some logging stuff
                logging.info(f"({time.perf_counter()-t0:.2f}s, Epoch {epoch}) Logging!")
                res = log_function(
                    model_pipeline,
                    epoch,
                    batch_idx,
                    num_epochs,
                )
                train_polar_rel_errs.append(res[0])
                train_cart_rel_errs.append(res[1])
                eval_polar_rel_errs.append(res[2])
                eval_cart_rel_errs.append(res[3])
                logged_batch_idcs.append(num_batches_per_epoch*epoch + batch_idx)
                # logged_batch_idcs.append(epoch) # I think this would be confusing though
                logging.info(
                    f"Current logging status...\n"
                    f"logged_batch_idcs = {logged_batch_idcs}\n"
                    f"train_polar_rel_errs = {train_polar_rel_errs}\n"
                    f"train_cart_rel_errs = {train_cart_rel_errs}\n"
                    f"eval_polar_rel_errs = {eval_polar_rel_errs}\n"
                    f"eval_cart_rel_errs = {eval_cart_rel_errs}"
                )
                logging.info(f"({time.perf_counter()-t0:.2f}s) logging done")

        # Update the learning rate and continue to the next epoch
        scheduler.step()

    plotting_info = (
        logged_batch_idcs,
        train_polar_rel_errs,
        train_cart_rel_errs,
        eval_polar_rel_errs,
        eval_cart_rel_errs,
    )
    return model_pipeline, plotting_info

# # For backwards compatibility...
# fine_tune_pipeline_blocks = train_pipeline_e2e

def main(
    args: argparse.Namespace,
    return_model: bool = False,
) -> None:
    """Driver code to fine-tune MFISNet-Model-Pipeline objects
    1. Load data
    2. Prepare data and additional tools
        a. select smoothed/original targets
        b. select the grids
        c. set up the datasets
        d. set up auxiliary objects like coordinate transformations and PDE solvers if necessary
        - prepare datasets and dataloaders
    3. Load NN
    4. Set up the logging function
    5. Train NN
    6. Save info to disk
        a. Select the information corresponding to the best epoch
        b. Save the model weights
        c. Save the log information
    """
    # ** try with double-precision **
    # global TORCH_CDTYPE, TORCH_RDTYPE, NP_CDTYPE
    # TORCH_CDTYPE = torch.cdouble
    # TORCH_RDTYPE = torch.double
    # NP_CDTYPE = np.cdouble

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    batch_size_adjust_factor = (args.batch_size / 16)
    lr_init_eff = args.lr_init_base * batch_size_adjust_factor
    eta_min_eff = args.eta_min_base * batch_size_adjust_factor
    weight_decay_eff = args.weight_decay_base * batch_size_adjust_factor
    logging.info(
        f"Adjusting the learning rate by a factor of {batch_size_adjust_factor} "
        f" since batch_size={args.batch_size} (vs. the default 16) "
        f"Using lr_init_eff={lr_init_eff}, eta_min_eff={eta_min_eff}, "
        f"and weight_decay_eff={weight_decay_eff} (c.f.: lr_init_base={args.lr_init_base}, "
        f"eta_min_base={args.eta_min_base}, and weight_decay={args.weight_decay_base})."
    )

    use_pde_solver_loss = (args.d_rs_loss_weight != 0)
    convert_d_rs = (args.noise_to_signal_ratio != 0) and use_pde_solver_loss
    noise_location = "d_rs" if convert_d_rs else "d_mh"

    out_all_models_dir = os.path.join(args.out_model_dir, "all_models")
    if os.path.isdir(args.out_model_dir):
        # If it exists, then remove everything in it...
        # shutil.rmtree(args.out_model_dir) # skip for now to avoid bad mistakes...
        model_files = glob.glob(f"{args.out_model_dir}/model_params_f*.pickle")
        for mf in model_files:
            os.remove(mf)
        if os.path.isdir(out_all_models_dir):
            shutil.rmtree(out_all_models_dir)
    else:
        os.mkdir(args.out_model_dir)
    # Create an extra directory if requested by save_all_model_weights...
    if args.save_all_model_weights:
        if not os.path.isdir(out_all_models_dir):
            os.mkdir(out_all_models_dir)
    # Clear out the results file if one already exists...
    if os.path.exists(args.out_train_results_fp):
        os.remove(args.out_train_results_fp)

    #########################################################
    ### 1. Data loading ###

    data_dir_base  = args.data_dir_base
    str_nu_list = (
        args.data_input_nus
    )  # nu in string form (to preserve decimals properly)
    nu_list = [float(str_nu) for str_nu in str_nu_list]
    N_freqs = len(nu_list)
    logging.info(f"data_dir_base: {data_dir_base}")
    logging.info(f"nu values received: {str_nu_list}")

    train_files = [
        os.path.join(data_dir_base, f"train_measurements_nu_{nu}") for nu in str_nu_list
    ]
    eval_files = [
        os.path.join(data_dir_base, f"val_measurements_nu_{nu}") for nu in str_nu_list
    ]

    logging.info(f"Loading training dataset")
    logging.info(
        f"Attempting to load the following folders: {train_files}"
    )

    if args.use_noise_seed:
        eff_noise_seed_list_train = args.noise_seed_list_train
        eff_noise_seed_list_val   = args.noise_seed_list_val
    else:
        eff_noise_seed_list_train = None
        eff_noise_seed_list_val   = None

    if args.noise_to_signal_ratio != 0:
        logging.info(f"Using seed as {eff_noise_seed_list_train} for the training set")
        logging.info(f"Using seed as {eff_noise_seed_list_val} for the val set")
    else:
        logging.info(f"Not adding noise!")


    train_dd, train_meta_dd = load_multifreq_dataset(
        train_files,
        truncate_num=args.truncate_num,
        # key_replacement=key_replacement,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to=noise_location,
        nan_mode="skip",
        load_cart=True,
        noise_seed=eff_noise_seed_list_train,
    )
    def kv_shrinker(key, val):
        """Little helper function to see the shapes of entires in a dictionary"""
        if isinstance(val, np.ndarray):
            if val.size > 1:
                return f"{key}<shape>", val.shape
            else:
                return key, val.item()
        elif hasattr(val, "__len__") and len(val) > 1:
            return f"{key}<len>", len(val)
        else:
            return key, val

    train_dd_short = dict(kv_shrinker(k, v) for (k, v) in train_dd.items())
    logging.info(f"train_dd has entries with shapes: {train_dd_short}")

    logging.info(f"Loading evaluation dataset")
    logging.info(
        f"Attempting to load the following folders: {eval_files}"
    )
    eval_dd, eval_meta_dd = load_multifreq_dataset(
        eval_files,
        truncate_num=args.truncate_num_val,
        # key_replacement=key_replacement,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to=noise_location,
        nan_mode="skip",
        load_cart=True,
        noise_seed=eff_noise_seed_list_val,
    )
    eval_dd_short = dict(kv_shrinker(k, v) for (k, v) in eval_dd.items())
    logging.info(f"eval_dd has entries with shapes: {eval_dd_short}")

    ### 2. Data processing/transformation ###
    # 2a. Prepare the q_polar targets
    train_q_polar_orig = train_dd[Q_POLAR]
    eval_q_polar_orig  = eval_dd[Q_POLAR]
    train_q_cart_orig  = train_dd[Q_CART]
    eval_q_cart_orig   = eval_dd[Q_CART]
    if args.use_smoothed_targets:
        logging.info(f"Using smoothed targets for training and validation")
        train_q_polar = train_dd[Q_POLAR_LPF][:, -1, ...]
        eval_q_polar  = eval_dd[Q_POLAR_LPF][:, -1, ...]
        train_q_cart  = train_dd[Q_CART_LPF][:, -1, ...]
        eval_q_cart   = eval_dd[Q_CART_LPF][:, -1, ...]
    else:
        logging.info(f"Using original targets for training and validation")
        train_q_polar = train_q_polar_orig
        eval_q_polar  = eval_q_polar_orig
        train_q_cart  = train_q_cart_orig
        eval_q_cart   = eval_q_cart_orig

    if not use_pde_solver_loss:
        train_d_mh  = train_dd[D_MH]
        eval_d_mh   = eval_dd[D_MH]
        train_d_rs  = None
        eval_d_rs   = None
    else:
        train_d_rs  = train_dd[D_RS]
        eval_d_rs   = eval_dd[D_RS]
        if not convert_d_rs:
            train_d_mh  = train_dd[D_MH]
            eval_d_mh   = eval_dd[D_MH]
        else:
            # To do: set up the (r,s)-to-(m,h) transforms and then apply them to the d_rs data
            raise NotImplementedError(
                f"Converting the noise from d_rs to d_mh is not yet implemented :( Please fix..."
            )

    # 2b. Fetch the relevant grids and dimensions
    rho_vals = train_dd["rho_vals"]
    theta_vals = train_dd["theta_vals"]
    h_vals = train_dd["h_vals"]
    omega_vals = train_dd["omega_sf"]
    x_vals = train_dd["x_vals"]

    N_x = x_vals.shape[0]
    N_rho = rho_vals.shape[0]
    N_h = h_vals.shape[0]
    N_theta = theta_vals.shape[0]
    N_m = N_theta
    N_train = train_q_polar.shape[0]
    N_eval = eval_q_polar.shape[0]

    # 2c. Next... run the "setup_dataset" function (may require re-organizing)
    # train_dset = setup_single_dataset_with_d_rs(
    train_dset = setup_dataset_full(
        train_q_polar,
        train_q_cart,
        train_d_mh,
        q_polar_orig=train_q_polar_orig,
        q_cart_orig=train_q_cart_orig,
        wave_field_rs=train_d_rs,
    )
    # eval_dset = setup_single_dataset_with_d_rs(
    eval_dset = setup_dataset_full(
        eval_q_polar,
        eval_q_cart,
        eval_d_mh,
        q_polar_orig=eval_q_polar_orig,
        q_cart_orig=eval_q_cart_orig,
        wave_field_rs=eval_d_rs,
    )
    logging.info(f"Finished loading data. N_train={N_train}, N_eval={N_eval}")

    # Send to the data loader
    train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, num_workers=1, prefetch_factor=2)
    eval_dloader  = torch.utils.data.DataLoader(eval_dset,  batch_size=args.batch_size, num_workers=1, prefetch_factor=2)

    # Create a subset of the training set for faster logging
    rng = np.random.default_rng(args.seed+17934)
    subset_size = int(args.log_train_subset_frac * args.truncate_num)
    train_subset_idcs = rng.choice(args.truncate_num, size=subset_size, replace=False)
    train_subset = torch.utils.data.Subset(
        dataset=train_dset,
        indices=train_subset_idcs,
    )
    train_subset_dloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, num_workers=1, prefetch_factor=2)

    # 2d. Set up the auxiliary objects (coordinate transforms, solver object)
    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training on device: %s", device)
    # Prepare for polar to cartesian transformations
    polar_to_cart_fn = prepare_polar_to_cart(
        x_vals, theta_vals, rho_vals, conv_op_device=device
    )
    logging.info(f"Finished setting up cartesian to polar interpolation")

    if use_pde_solver_loss:
        # Set up the solver at the last frequency
        nu_last = nu_list[-1]
        prepare_half_grid = (args.pde_fwd_use_half_grid or args.pde_adj_use_half_grid)
        solver_obj = setup_differentiable_solver(
            N_x, args.pde_spatial_domain_max, nu_last, args.pde_receiver_radius,
            device=device,
            prepare_half_grid=prepare_half_grid,
        )
        pde_solver_config = {
            "fwd_linsys_solver": args.pde_fwd_linsys_solver,
            "adj_linsys_solver": args.pde_adj_linsys_solver,
            "rtol": args.pde_fwd_rtol,
            "adj_rtol": args.pde_adj_rtol,
            "use_half_grid": args.pde_fwd_use_half_grid,
            "half_grid_tol_ratio": args.pde_fwd_half_grid_tol_ratio,
            "adj_use_half_grid": args.pde_adj_use_half_grid,
            "adj_half_grid_tol_ratio": args.pde_adj_half_grid_tol_ratio,
            "batch_size": args.pde_batch_size,
            "max_iter": args.pde_max_iter,
            "restart": args.pde_restart,
            # placeholder values until I get the motivation to update the templates
            "convergence_by_dir": True,
            # leave these values for now to avoid extra outputs
            "error_unless_converged": False,
            "verbose": False,
            "report_status": False,
            "_solve_Helmholtz_inv_msg": False,
        }
        logging.info(f"Finished setting up the PDE solver for use with PDE solver loss")

        # Prepare for (r,s) to (m,h) transformations
        conv_rs_to_m, conv_rs_to_h = prep_rs_to_mh_interp(
            theta_vals,  # r grid points
            theta_vals,  # s grid points
            N_theta,
            len(h_vals),
            a_neg_half=True,
        )
        torch_rs_to_m = torch.tensor(
            conv_rs_to_m.todense(), dtype=TORCH_CDTYPE, requires_grad=False, device=device,
        )
        torch_rs_to_h = torch.tensor(
            conv_rs_to_h.todense(), dtype=TORCH_CDTYPE, requires_grad=False, device=device,
        )
        def rs_to_mh_fn(d_rs_i):
            nonlocal N_m, N_h, torch_rs_to_m, torch_rs_to_h
            return apply_interp_2d(
                torch_rs_to_m, torch_rs_to_h, d_rs_i
            ).reshape(N_m, N_h)
        logging.info(f"Finished setting up (r,s) to (m,h) interpolation")
    else:
        # Define dummy variables to avoid errors
        solver_obj = None
        pde_solver_config = dict()
        # polar_to_cart_fn = None
        rs_to_mh_fn = None
        logging.info(f"Skipping the PDE-solver loss setup steps...")


    ### 3. Model loading ###
    ### Prepare for NN training ###
    # Get the hyperparameters first
    logging.info(f"Starting to load the model")
    selected_model_hyperparams_dd = load_yaml_to_dict(args.in_central_results_fp)
    internal_pde_solver_config = {
        "solver_type": args.pde_solver_type,
        # HPS settings
        "hps_l": args.pde_hps_l,
        "hps_p": args.pde_hps_p,
        "hps_comp_domain_factor": args.pde_hps_comp_domain_factor,
        "hps_sd_mat_dir": args.pde_hps_sd_mat_dir,
        "hps_save_tree": (args.batch_size==1), # only save if the batch size is 1
        # LS settings
        "fwd_linsys_solver": args.pde_fwd_linsys_solver,
        "adj_linsys_solver": args.pde_adj_linsys_solver,
        "rtol": args.pde_fwd_rtol,
        "adj_rtol": args.pde_adj_rtol,
        "use_half_grid": args.pde_fwd_use_half_grid,
        "half_grid_tol_ratio": args.pde_fwd_half_grid_tol_ratio,
        "adj_use_half_grid": args.pde_adj_use_half_grid,
        "adj_half_grid_tol_ratio": args.pde_adj_half_grid_tol_ratio,
        "batch_size": args.pde_batch_size,
        "max_iter": args.pde_max_iter,
        "restart": args.pde_restart,
        # leave these values for now to avoid extra outputs
        "error_unless_converged": False,
        "verbose": False,
        "report_status": False,
        "_solve_Helmholtz_inv_msg": False,
    }
    model_pipeline = load_MFISNet_Model_Pipeline_from_state_dict(
        selected_model_hyperparams_dd,
        device,
        N_x=N_x,
        pde_solver_config=internal_pde_solver_config,
        prepare_half_grid=True,
        rho_vals=rho_vals,
    ).to(device)
    logging.info(f"Finished loading the model!")

    ### 4. Set up the logging function ###
    N_epochs = args.n_epochs
    loss_module_0 = MSEModule()
    polar_loss_fn_dd = {
        "mse": loss_module_0.mse,
        "psnr": loss_module_0.psnr,
        "rel_l2": loss_module_0.relative_l2_error,
        "final_mse": loss_module_0.mse_against_final,
        "final_psnr": loss_module_0.psnr_against_final,
        "final_rel_l2": loss_module_0.relative_l2_error_against_final,
    }
    cart_loss_fn_dd = {
        "cart_mse": loss_module_0.mse,
        "cart_psnr": loss_module_0.psnr,
        "cart_rel_l2": loss_module_0.relative_l2_error,
        "cart_final_mse": loss_module_0.mse_against_final,
        "cart_final_psnr": loss_module_0.psnr_against_final,
        "cart_final_rel_l2": loss_module_0.relative_l2_error_against_final,
    }
    current_best_info = {
        "epoch": None,
        "batch_idx": None,
        "model": None,
        "field_val": None,
        "train_log_dd": None,
    }

    id_hash = hash_dict(vars(args))
    epoch_stagger = 0  # Just a single training phase
    with wandb.init(
        id=id_hash,
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        # mode="disabled" if skip_wandb else args.wandb_mode,
        mode=args.wandb_mode,
        reinit=True,
        resume=None,
        settings=wandb.Settings(start_method="fork"),
    ) as wandbrun:
        # First, set up the logging function
        def log_function(model_0, epoch_local, batch_idx_local, N_epochs):
            nonlocal args
            nonlocal train_dloader, eval_dloader, train_subset_dloader
            nonlocal current_best_info, out_all_models_dir
            nonlocal use_pde_solver_loss, polar_to_cart_fn
            nonlocal polar_loss_fn_dd, cart_loss_fn_dd
            with torch.no_grad():
                # 1. Perform the evaluation
                weight_norm = torch.norm(
                    torch.cat([x.view(-1) for x in model_0.parameters()]), 2
                )
                outs = evaluate_losses_on_dataloader_with_cartesian(
                    # model_0, train_dloader,
                    model_0, train_subset_dloader,
                    polar_loss_fn_dd, cart_loss_fn_dd, polar_to_cart_fn,
                    device, incl_x_rs=use_pde_solver_loss,
                )
                polar_train_loss_dd, cart_train_loss_dd = outs
                train_loss_dd = {**polar_train_loss_dd, **cart_train_loss_dd}
                # train_loss_dd = evaluate_losses_on_dataloader(
                #     model_0, train_dloader, loss_fn_dd, device,
                #     incl_x_rs=use_pde_solver_loss,
                # )
                train_polar_rel_err = torch.mean(train_loss_dd["rel_l2"]).item()
                train_cart_rel_err  = torch.mean(train_loss_dd["cart_rel_l2"]).item()
                logging.info(
                    "(polar) Train MSE: {:.5e}, Train Rel L2: {:.5f}, Train PSNR: {:.3f}".format(
                        torch.mean(train_loss_dd["mse"]).item(),
                        torch.mean(train_loss_dd["rel_l2"]).item(),
                        torch.mean(train_loss_dd["psnr"]).item(),
                        # train_rel_l2_aaa,
                        # train_psnr_aaa,
                    )
                )
                logging.info(
                    "(cart)  Train MSE: {:.5e}, Train Rel L2: {:.5f}, Train PSNR: {:.3f}".format(
                        torch.mean(train_loss_dd["cart_mse"]).item(),
                        torch.mean(train_loss_dd["cart_rel_l2"]).item(),
                        torch.mean(train_loss_dd["cart_psnr"]).item(),
                        # train_rel_l2_aaa,
                        # train_psnr_aaa,
                    )
                )
                # eval_loss_dd = evaluate_losses_on_dataloader(
                #     model_0, eval_dloader, loss_fn_dd, device,
                #     incl_x_rs=use_pde_solver_loss,
                # )
                outs = evaluate_losses_on_dataloader_with_cartesian(
                    model_0, eval_dloader,
                    polar_loss_fn_dd, cart_loss_fn_dd, polar_to_cart_fn,
                    device, incl_x_rs=use_pde_solver_loss,
                )
                polar_eval_loss_dd, cart_eval_loss_dd = outs
                eval_loss_dd = {**polar_eval_loss_dd, **cart_eval_loss_dd}

                eval_polar_rel_err = torch.mean(eval_loss_dd["rel_l2"]).item()
                eval_cart_rel_err  = torch.mean(eval_loss_dd["cart_rel_l2"]).item()
                logging.info(
                    "   (polar) Val MSE: {:.5e}, Val Rel L2: {:.5f}, Val PSNR: {:.3f}".format(
                        torch.mean(eval_loss_dd["mse"]).item(),
                        torch.mean(eval_loss_dd["rel_l2"]).item(),
                        torch.mean(eval_loss_dd["psnr"]).item(),
                        # test_mse_aaa,
                        # test_rel_l2_aaa,
                        # test_psnr_aaa,
                    )
                )
                logging.info(
                    "   (cart)  Val MSE: {:.5e}, Val Rel L2: {:.5f}, Val PSNR: {:.3f}".format(
                        torch.mean(eval_loss_dd["cart_mse"]).item(),
                        torch.mean(eval_loss_dd["cart_rel_l2"]).item(),
                        torch.mean(eval_loss_dd["cart_psnr"]).item(),
                    )
                )
                logging.info("\tWeight L2 norm: {:.3e}".format(weight_norm.item()))

                # Memory usage check-in
                process = psutil.Process()
                logging.info(
                    f"Memory usage: {process.memory_info().rss>>20} MB"
                )  # this is not where the memory usage peaks
                if torch.cuda.is_available():
                    vram_free_bytes, vram_available_bytes = torch.cuda.mem_get_info()
                    vram_used_mb = (vram_available_bytes - vram_free_bytes) >> 20
                    logging.info(
                        f"Current VRAM usage: {vram_used_mb} MB / {vram_available_bytes>>20} MB"
                    )

                # 2. Collect the relevant information into the logs
                epoch_overall = epoch_local + epoch_stagger
                batch_idx_overall = len(train_dloader) * epoch_overall + batch_idx_local
                train_loss_entries = {
                    f"train_{k}": torch.mean(v).item() for (k, v) in train_loss_dd.items()
                }
                eval_loss_entries = {
                    f"eval_{k}": torch.mean(v).item() for (k, v) in eval_loss_dd.items()
                }
                train_log_dd = {
                    ##### These entries change each round... #####
                    # Epoch/batch information
                    "epoch": epoch_overall,
                    "batch_idx_local": batch_idx_local,
                    "batch_idx": batch_idx_overall,

                    # Evaluation metrics
                    **train_loss_entries,
                    **eval_loss_entries,
                    "weight_norm": weight_norm.item(),

                    ##### After this, the entries stay the same every round... #####

                    # Experiment info/dimensions
                    "n_train": N_train,
                    "n_eval": N_eval,
                    "n_freqs": N_freqs,
                    "n_rho_vals": N_rho,
                    "n_theta_vals": N_theta,
                    "n_h_vals": N_h,
                    "n_x_vals": N_x,

                    # Architecture info (minimal I guess)
                    "block_types": "-".join(model_0.block_types),

                    # Optimization info
                    "lr_init_base": args.lr_init_base,
                    "eta_min_base": args.eta_min_base,
                    "weight_decay_base": args.weight_decay_base,
                    "lr_init_eff": lr_init_eff,
                    "eta_min_eff": eta_min_eff,
                    "weight_decay_eff": weight_decay_eff,
                    "batch_size": args.batch_size,

                    # PDE solver-related settings
                    "use_pde_solver_loss": use_pde_solver_loss,
                    "d_rs_loss_weight": args.d_rs_loss_weight,
                    "pde_fwd_linsys_solver": args.pde_fwd_linsys_solver,
                    "pde_adj_linsys_solver": args.pde_adj_linsys_solver,
                    "pde_fwd_rtol": args.pde_fwd_rtol,
                    "pde_adj_rtol": args.pde_adj_rtol,
                    "pde_fwd_use_half_grid": args.pde_fwd_use_half_grid,
                    "pde_adj_use_half_grid": args.pde_adj_use_half_grid,
                    "pde_fwd_half_grid_tol_ratio": args.pde_fwd_half_grid_tol_ratio,
                    "pde_adj_half_grid_tol_ratio": args.pde_adj_half_grid_tol_ratio,
                    "pde_max_iter": args.pde_max_iter,
                    "pde_batch_size": args.pde_batch_size,
                    "pde_restart": args.pde_restart,
                    "pde_spatial_domain_max": args.pde_spatial_domain_max,
                    "pde_receiver_radius": args.pde_receiver_radius,

                    # Extra data
                    "hash": id_hash,
                    "source_nu_list": nu_list,
                }
                # # Append the evaluation metrics
                # for k, v in train_loss_dd.items():
                #     train_log_dd["train_" + k] = torch.mean(v).item()
                # for k, v in eval_loss_dd.items():
                #     train_log_dd["eval_" + k] = torch.mean(v).item()

                # 3. Write the log entry to disk and wandb; also save weights if relevant
                write_result_to_file(args.out_train_results_fp, **train_log_dd)

                if args.save_all_model_weights:
                    block_fp_list = [
                        os.path.join(
                            out_all_models_dir,
                            f"model_params_{freq_idx}_epoch_{epoch_overall}"
                            f"_batch_{batch_idx_local}.pickle"
                        )
                        for freq_idx in range(1, 1+N_freqs)
                    ]
                    logging.info(f"Preparing to save all model weights to {block_fp_list}...")
                    save_MFISNet_Model_Pipeline_by_block(model_pipeline, block_fp_list)


                # Track which model is the best according to the selection criterion
                recent_field_val = train_log_dd[args.selection_field]
                is_min_mode = args.selection_mode=="min"
                is_max_mode = args.selection_mode=="max"
                cbi_field_val = current_best_info["field_val"]
                if cbi_field_val is None or \
                   (is_min_mode and recent_field_val < cbi_field_val) or \
                   (is_max_mode and recent_field_val > cbi_field_val):
                   # Save to the current best model...
                    current_best_info["model"] = copy.deepcopy(model_0).to("cpu")
                    current_best_info["field_val"] = recent_field_val
                    current_best_info["epoch"] = epoch_overall
                    current_best_info["batch_idx"] = batch_idx_overall
                    current_best_info["train_log_dd"] = {**train_log_dd}
            errs = (
                train_polar_rel_err, train_cart_rel_err,
                eval_polar_rel_err, eval_cart_rel_err,
            )
            return errs

    ### 5. Train the model! ###
    t0 = time.perf_counter()
    log_by_epoch = (args.n_batches_per_log==0)
    log_every_n   = args.n_epochs_per_log if log_by_epoch else args.n_batches_per_log
    logging.info(f"~~~ log_by_epoch: {log_by_epoch} ~~~")
    logging.info(f"~~~ log_every_n: {log_every_n} ~~~")

    model_pipeline, extra_info = train_pipeline_e2e(
        model_pipeline,
        train_dloader,
        eval_dloader,
        # train_subset_dloader,
        loss_function=loss_module_0,
        log_function=log_function,
        device=device,
        solver_obj=solver_obj,
        pde_solver_config=pde_solver_config,
        polar_to_cart_fn=polar_to_cart_fn,
        rs_to_mh_fn=rs_to_mh_fn,
        d_rs_loss_weight=args.d_rs_loss_weight,
        lr_init=lr_init_eff,
        eta_min=eta_min_eff,
        weight_decay=weight_decay_eff,
        log_every_n=log_every_n,
        log_by_epoch=log_by_epoch,
        num_epochs=N_epochs,
        verbose_level=args.verbose_level,
    )
    logging.info(f"Optimization complete!") # easier to grep for
    logging.info(f"Fine-tuning finished after {time.perf_counter() - t0:.3f}s")
    logged_batch_idcs = extra_info[0]
    train_polar_rel_errs, train_cart_rel_errs = extra_info[1:3]
    eval_polar_rel_errs, eval_cart_rel_errs = extra_info[3:5]

    logging.info(f"N_train={N_train}")
    logging.info(f"N_eval={N_eval}")
    logging.info(
        f"logged_batch_idcs = {logged_batch_idcs}\n"
        f"train_polar_rel_errs = {train_polar_rel_errs}\n"
        f"train_cart_rel_errs = {train_cart_rel_errs}\n"
        f"eval_polar_rel_errs = {eval_polar_rel_errs}\n"
        f"eval_cart_rel_errs = {eval_cart_rel_errs}\n"
    )

    # 6. Save the model and results to disk
    # 6a. Select the relevant information
    cbi_epoch = current_best_info["epoch"]
    cbi_batch_idx = current_best_info["batch_idx"]
    cbi_field_val = current_best_info["field_val"]
    cbi_train_log_dd = current_best_info["train_log_dd"]
    logging.info(
        f"Selected the model from epoch {cbi_epoch}, batch {cbi_batch_idx}, with "
        f"field '{args.selection_field}' value {cbi_field_val} "
        f"(selection mode: {args.selection_mode})."
    )
    logging.info(f"Selected train log... {cbi_train_log_dd}")

    # 6b. Save the model weights to disk
    block_fp_list = [
        os.path.join(
            args.out_model_dir,
            f"model_params_{freq_idx}.pickle"
        )
        for freq_idx in range(1, 1+N_freqs)
    ]
    logging.info(f"Saving the best versions to {block_fp_list}")
    save_MFISNet_Model_Pipeline_by_block(model_pipeline, block_fp_list)

    smh_dd = selected_model_hyperparams_dd
    for fi in range(N_freqs):
        freq_idx = fi + 1
        smh_dd[f"freq_idx_{freq_idx}"]["central_model_fp"] = block_fp_list[fi]

    # 6c. Save the results to disk
    out_central_results_dd = {
        "e2e_info": {
            "block_fp_list": block_fp_list, # for convenience even though the model fps are already updated
            **cbi_train_log_dd,
        },
        **smh_dd,
    }
    logging.info(f"Writing to out_central_results_fp={args.out_central_results_fp}")
    save_dict_to_yaml(out_central_results_dd, args.out_central_results_fp)

    # Exit when finished, optionally returning the model
    logging.info("Finished!")
    if return_model:
        return model_pipeline
    return


if __name__ == "__main__":
    a = setup_args()

    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    if a.debug:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)

    logging.info(f"Received the following arguments: {a}")

    try:
        if a.jax_mem_alloc_mb == 0:
            logging.info(f"Skipping jax/hps loading! (jax_mem_alloc_mb==0)")
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.0"
        else:
            # Calculate the vram fraction required to have jax_mem_alloc_mb allocated to jax
            jax_mem_alloc_frac = (
                vram_mb_to_frac(a.jax_mem_alloc_mb)
                if a.jax_mem_alloc_mb is not None
                else 0.4
            )
            jax_mem_alloc_mb = int(jax_mem_alloc_frac * get_vram_total_mb())
            logging.info(f"JAX VRAM allocation: {jax_mem_alloc_frac:.3f} (in MB: {jax_mem_alloc_mb})")
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{jax_mem_alloc_frac:.3f}"
            import jax
            jax_device = jax.devices("gpu")[0]
            jax.config.update("jax_default_device", jax_device)
            jax.config.update("jax_enable_x64", True)
            jax_device = jax.devices("gpu")[0]
            logging.info(f"Note: Jax has been loaded :)")
            from solvers.hps.wave_scattering import PytorchHPSSolver
            logging.info(f"Note: HPS has been loaded :)")
    except:
        logging.info(f"Note: Jax has not been loaded due to an error :(")

    main(a)
