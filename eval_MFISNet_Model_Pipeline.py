# Evaluate the MFISNet_Model_Pipeline object and save the predictions

# Mostly standard imports
import numpy as np
import torch
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import shutil, psutil
import os, sys, glob
import time
import logging
import copy
import yaml
from typing import Tuple, Callable, Dict
import argparse
# import tqdm

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

from src.models.MFISNet_Model_Pipeline import (
    MFISNet_Model_Pipeline,
    save_MFISNet_Model_Pipeline_by_block,
    load_MFISNet_Model_Pipeline_from_state_dict,
)
from src.data.datasets import (
    FullData,
    setup_dataset_full,
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
    CONST_D_MH_SCALE_FACTOR,
    prepare_polar_to_cart,
)

# Training
from src.training_utils.train_loop import (
    evaluate_losses_on_dataloader,
    evaluate_losses_on_dataloader_with_cartesian,
)
from src.training_utils.loss_functions import MSEModule
from src.training_utils.make_predictions import make_preds_on_dataset

# Misc. utilities
# from src.utils.plotting_utils import plot_row
from src.utils.vram_info import get_memory_info, free_vram

from src.utils.logging_utils import (
    write_result_to_file,
    find_best_epoch,
    load_field_in_yaml_file,
    load_yaml_to_dict,
    save_dict_to_yaml,
    FMT, TIMEFMT,
    hash_dict
)

# try:
#     os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
#     import jax # also sets options in the __name__=="__main__" section
# except:
#     pass


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
    # Caution -- will need to apply noise to d_rs first if the PDE solver is used,
    # then d_mh later
    parser.add_argument(
        "--noise_to_signal_ratio", default=None, type=float
    )
    parser.add_argument("--use_noise_seed", choices=["true", "false"], default="false")
    parser.add_argument("--noise_seed_list",  type=int, nargs="*", default=None)

    parser.add_argument(
        "--in_central_results_fp",
        type=str,
        help="The results yaml containing relevant model paths and hyperparameters",
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
    parser.add_argument(
        "--eval_on_set",
        choices=["train", "val", "test"],
        default="test",
        help="Choose one of the train/val/test sets to evaluate on",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int, default=50,
    )
    parser.add_argument("--seed", default=None, type=int)  # seed bc we're using noise

    ### Register PDE-Solver options as well (for use with PSR blocks) ###
    parser.add_argument(
        "--use_pde_args",
        choices=bool_choices,
        default="false",
        help="Whether to use the PDE argument values rather than the settings found in "
            "the 'finetune_info' section of in_central_results_fp. Defaults to using "
            "the settings from disk unless the section is not found."
    )
    parser.add_argument(
        "--pde_solver_type", choices=["ls", "hps"], default="ls",
        help="Solver type: Lippmann-Schwinger ('ls') or Hierarchical Poincare-Steklov ('hps'). "
        "Defaults to the Lippmann-Schwinger solver."
    )
    parser.add_argument(
        "--pde_fwd_linsys_solver",
        choices=["bicgstab", "gmres"], default="bicgstab"
    )
    parser.add_argument("--pde_fwd_rtol", type=float, default=1e-2)
    parser.add_argument("--pde_fwd_use_half_grid", choices=bool_choices, default="true")
    parser.add_argument("--pde_fwd_half_grid_tol_ratio", type=float, default=0.5)
    parser.add_argument("--pde_max_iter", type=int, default=1000)
    parser.add_argument("--pde_spatial_domain_max", type=float, default=0.5)
    parser.add_argument("--pde_receiver_radius", type=float, default=100)
    parser.add_argument("--pde_batch_size", type=int, default=100)
    parser.add_argument("--pde_restart", type=int, default=10)
    # HPS Settings
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

    ### Logging options ###
    parser.add_argument(
        "--timing_run", default="false", choices=["true", "false"],
        help="Evaluates a second time without saving outputs for a better timing estimate"
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--verbose_level", default=1, type=int)

    parser.add_argument(
        "--output_predictions_dir",
        type=str,
        help="Point to the desired output predictions file",
    )
    parser.add_argument(
        "--output_summary_fp",
        type=str,
        help="Point to the desired output summary file",
    )
    parser.add_argument(
        "--samples_per_chunk",
        type=int,
        default=500,
        help="This is the 'shard_size' for make_predictions_on_dataset",
    )


    a = parser.parse_args()

    # Parse boolean arguments + misc. arguments that need extra processing
    a.pde_fwd_use_half_grid = (a.pde_fwd_use_half_grid == "true")
    a.use_pde_args = (a.use_pde_args == "true")
    a.use_noise_seed = (a.use_noise_seed == "true")
    a.timing_run = (a.timing_run == "true")

    # Override unless use_targets="legacy"
    if a.use_targets == "smoothed":
        a.use_smoothed_targets = True
    elif a.use_targets == "original":
        a.use_smoothed_targets = False

    return a


def main(
    args: argparse.Namespace,
    return_model: bool = False,
) -> None:
    """Driver code to evaluate MFISNet-Model-Pipeline objects
    1. Load data
    2. Prepare data and additional tools
        a. select smoothed/original targets
        b. select the grids
        c. set up the datasets
        d. set up auxiliary objects like coordinate transformations and PDE solvers if necessary
        - prepare datasets and dataloaders
    3. Load the NN correspondign to the best epoch
    4. Set up the loss function and save the predictions to disk
    5. Compute the error/performance statistics and save the results to disk
    """
    # ** try with double-precision **
    # global TORCH_CDTYPE, TORCH_RDTYPE, NP_CDTYPE
    # TORCH_CDTYPE = torch.cdouble
    # TORCH_RDTYPE = torch.double
    # NP_CDTYPE = np.cdouble

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_noise_seed and len(args.noise_seed_list) > 0:
        eff_noise_seed_list = args.noise_seed_list
    else:
        eff_noise_seed_list = None

    if args.noise_to_signal_ratio != 0:
        logging.info(f"Using seed as {eff_noise_seed_list}")
    else:
        logging.info(f"Not adding noise!")


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

    eval_set_name = args.eval_on_set
    eval_files = [
        os.path.join(data_dir_base, f"{eval_set_name}_measurements_nu_{nu}") for nu in str_nu_list
    ]

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

    eval_dd, eval_meta_dd = load_multifreq_dataset(
        eval_files,
        truncate_num=args.truncate_num,
        # key_replacement=key_replacement,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_mh",
        nan_mode="skip",
        load_cart=True,
        noise_seed=eff_noise_seed_list,
    )
    eval_dd_short = dict(kv_shrinker(k, v) for (k, v) in eval_dd.items())
    logging.info(f"eval_dd has entries with shapes: {eval_dd_short}")

    ### 2. Data processing/transformation ###
    # 2a. Prepare the q_polar targets
    eval_q_polar_orig  = eval_dd[Q_POLAR]
    eval_q_cart_orig   = eval_dd[Q_CART]
    if args.use_smoothed_targets:
        logging.info(f"Using smoothed targets for evaluation")
        eval_q_polar  = eval_dd[Q_POLAR_LPF][:, -1, ...]
        eval_q_cart   = eval_dd[Q_CART_LPF][:, -1, ...]
    else:
        logging.info(f"Using original targets for evaluation")
        eval_q_polar  = eval_q_polar_orig
        eval_q_cart   = eval_q_cart_orig

    eval_d_mh   = eval_dd[D_MH]

    # 2b. Fetch the relevant grids and dimensions
    rho_vals = eval_dd["rho_vals"]
    theta_vals = eval_dd["theta_vals"]
    h_vals = eval_dd["h_vals"]
    omega_vals = eval_dd["omega_sf"]
    x_vals = eval_dd["x_vals"]

    N_x = x_vals.shape[0]
    N_rho = rho_vals.shape[0]
    N_h = h_vals.shape[0]
    N_theta = theta_vals.shape[0]
    N_m = N_theta
    N_eval = eval_q_polar.shape[0]

    # 2c. Next... run the "setup_dataset" function (may require re-organizing)
    eval_dset = setup_dataset_full(
        eval_q_polar,
        eval_q_cart,
        eval_d_mh,
        q_polar_orig=eval_q_polar_orig,
        q_cart_orig=eval_q_cart_orig,
        # wave_field_rs=eval_d_rs,
    )
    logging.info(f"Finished loading data. N_{eval_set_name}={N_eval}")

    # Send to the data loader
    eval_dloader = torch.utils.data.DataLoader(eval_dset, batch_size=args.eval_batch_size)


    # 2d. Set up the auxiliary objects (coordinate transforms, solver object)
    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Evaluating on device: %s", device)
    # Prepare for polar to cartesian transformations
    polar_to_cart_fn = prepare_polar_to_cart(
        x_vals, theta_vals, rho_vals, conv_op_device=device
    )

    ### 3. Model loading ###
    ### Prepare for NN evaluation ###
    # Get the hyperparameters first
    logging.info(f"Starting to load the model")
    selected_model_hyperparams_dd = load_yaml_to_dict(args.in_central_results_fp)
    main_pde_solver_keys = [
        "solver_type",
        "fwd_linsys_solver",
        "fwd_rtol",
        "fwd_use_half_grid",
        "fwd_half_grid_tol_ratio",
        "batch_size",
        "max_iter",
        "restart",
        "hps_l",
        "hps_p",
        "hps_comp_domain_factor",
        "hps_sd_mat_dir",
    ]
    # Extract the pde solver configuration
    e2e_section = "finetune_info" if "finetune_info" in selected_model_hyperparams_dd.keys() \
        else "e2e_info" if "e2e_info" in  selected_model_hyperparams_dd.keys() \
        else None
    # if (not args.use_pde_args) and "finetune_info" in selected_model_hyperparams_dd.keys():
    if (not args.use_pde_args) and e2e_section is not None:
        finetune_info = selected_model_hyperparams_dd[e2e_section]
        selected_pde_solver_config = {
            key: finetune_info[f"pde_{key}"]
            for key in main_pde_solver_keys
        }
    else:
        # Grab from the args values
        args_vars = vars(args) # converts the contents of args to a dict
        selected_pde_solver_config = {
            key: args_vars[f"pde_{key}"]
            for key in main_pde_solver_keys
        }
    model_dir = os.path.split(
        selected_model_hyperparams_dd["freq_idx_1"]["central_model_fp"]
    )[0]
    internal_pde_solver_config = {
        # leave these values for now to avoid extra outputs
        **selected_pde_solver_config,
        "error_unless_converged": False,
        "verbose": False,
        "report_status": (args.verbose_level >= 2),
        "_solve_Helmholtz_inv_msg": False,
    }
    logging.info(f"Loaded PDE settings as: {internal_pde_solver_config}")
    print(f"selected_model_hyperparams_dd={selected_model_hyperparams_dd}")
    model_pipeline = load_MFISNet_Model_Pipeline_from_state_dict(
        selected_model_hyperparams_dd,
        device,
        N_x=N_x,
        pde_solver_config=internal_pde_solver_config,
        prepare_half_grid=True,
        rho_vals=rho_vals,
    ).to(device)
    logging.info(f"Finished loading the model!")

    ### 4. Set up the loss functions, then evaluate ###
    loss_module_0 = MSEModule()
    polar_loss_fn_dd = {
        "polar_mse": loss_module_0.mse,
        "polar_psnr": loss_module_0.psnr,
        "polar_rel_l2": loss_module_0.relative_l2_error,
        "polar_final_mse": loss_module_0.mse_against_final,
        "polar_final_psnr": loss_module_0.psnr_against_final,
        "polar_final_rel_l2": loss_module_0.relative_l2_error_against_final,
    }
    cart_loss_fn_dd = {
        "cart_mse": loss_module_0.mse,
        "cart_psnr": loss_module_0.psnr,
        "cart_rel_l2": loss_module_0.relative_l2_error,
        "cart_final_mse": loss_module_0.mse_against_final,
        "cart_final_psnr": loss_module_0.psnr_against_final,
        "cart_final_rel_l2": loss_module_0.relative_l2_error_against_final,
    }

    # Evaluate the model and save the outputs to disk...
    _, _, polar_eval_loss_dd, cart_eval_loss_dd = make_preds_on_dataset(
        model=model_pipeline,
        dloader=eval_dloader,
        experiment_info=eval_meta_dd,
        output_dir=args.output_predictions_dir,
        device=device,
        shard_size=args.samples_per_chunk,
        use_orig_idcs=True,
        to_cart_fn=polar_to_cart_fn,
        evaluate_outputs=True,
        polar_loss_fn_dict=polar_loss_fn_dd,
        cart_loss_fn_dict=cart_loss_fn_dd,
        incl_x_rs=True,
    )
    eval_loss_dd = {**polar_eval_loss_dd, **cart_eval_loss_dd}
    if args.timing_run:
        t0 = time.perf_counter()
        make_preds_on_dataset(
            model=model_pipeline,
            dloader=eval_dloader,
            experiment_info=eval_meta_dd,
            output_dir=None,
            device=device,
            shard_size=args.samples_per_chunk,
            use_orig_idcs=True,
            to_cart_fn=polar_to_cart_fn,
            evaluate_outputs=False,
            polar_loss_fn_dict=dict(),
            cart_loss_fn_dict=dict(),
            incl_x_rs=True,
        )
        t1 = time.perf_counter()
        logging.info(f"Timing run took {t1-t0:.3f}s for {args.truncate_num} samples")


    # eval_polar_rel_err = torch.mean(eval_loss_dd["polar_rel_l2"]).item()
    # eval_cart_rel_err  = torch.mean(eval_loss_dd["cart_rel_l2"]).item()

    # 5. Compute the relevant statistics and save to disk
    cart_rel_l2_mean = torch.mean(eval_loss_dd["cart_rel_l2"]).item()
    cart_rel_l2_std  = torch.std(eval_loss_dd["cart_rel_l2"]).item()
    cart_mse_mean    = torch.mean(eval_loss_dd["cart_mse"]).item()
    cart_mse_std     = torch.std(eval_loss_dd["cart_mse"]).item()
    cart_psnr_mean   = torch.mean(eval_loss_dd["cart_psnr"]).item()
    cart_psnr_std    = torch.std(eval_loss_dd["cart_psnr"]).item()

    logging.info(f"~~~Summary~~~")
    logging.info(f"MSE error: {cart_mse_mean:.3e}±{cart_mse_std:.3e}")
    logging.info(f"Rel l2 error: {cart_rel_l2_mean:.5f}±{cart_rel_l2_std:.5f}")
    logging.info(f"PSNR: {cart_psnr_mean:.5f}±{cart_psnr_std:.5f}")

    summary_errors_dict = {
        "cart_mse_mean": cart_mse_mean,
        "cart_mse_std": cart_mse_std,
        "cart_rel_l2_mean": cart_rel_l2_mean,
        "cart_rel_l2_std": cart_rel_l2_std,
        "cart_psnr_mean": cart_psnr_mean,
        "cart_psnr_std": cart_psnr_std,
    }

    summary_dict = {
        # Summary values
        **summary_errors_dict,
        # Metadata
        # "block_fp_list": block_fp_list,
        "model_dir": model_dir,
        "predictions_fp": args.output_predictions_dir,
    }

    os.makedirs(os.path.split(args.output_summary_fp)[0], exist_ok=True)
    with open(args.output_summary_fp, "w") as sfile:
        yaml.dump(summary_dict, sfile, default_flow_style=False)
    logging.info(f"Saved summary file to {args.output_summary_fp}")
    logging.info(f"Saved predictions to {args.output_predictions_dir}")
    logging.info(f"Finished!")


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

    # if "jax" in sys.modules:
    #     jax_device = jax.devices("gpu")[0]
    #     jax.config.update("jax_default_device", jax_device)
    #     jax.config.update("jax_enable_x64", True)
    #     logging.info(f"Note: Jax has been loaded :)")
    # else:
    #     logging.info(f"Note: Jax has not been loaded :(")

    main(a)
