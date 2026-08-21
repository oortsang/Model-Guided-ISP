"""
This script does the following:
1. Grabs the scattering object file
2. Generates wave fields for the scattering objects.
5. Makes the necessary tranformations from FY19.
5. Saves the data to disk.
"""

import argparse
import logging
import os
import sys
from typing import Dict
import numpy as np
import wandb
import torch
import time

from src.data.data_io import (
    load_single_dir_slice,
    save_dict_to_hdf5,
    load_hdf5_to_dict,
    load_field_in_hdf5,
    update_field_in_hdf5,
)
from src.data.data_naming_constants import (
    X_VALS,
    SAMPLE_COMPLETION,
    SAMPLE_PROGRESS,
    FILE_COMPLETION,
    Q_CART,
    Q_CART_LPF,
    Q_POLAR,
    Q_POLAR_LPF,
    D_RS,
    D_MH,
    NU_SF,
    OMEGA_SF,
    TRUNCATABLE_KEYS,
)
from src.training_utils.loss_functions import MSEModule

from src.utils.pipeline_utils import pretty_dict_to_str

# Jax versions are imported later
from src.utils.vram_info import get_memory_info, free_vram, vram_mb_to_frac, get_vram_total_mb
from src.data.add_noise import add_noise_to_d

import logging
FMT = "%(asctime)s:recursive-linearization: %(levelname)s - %(message)s"
TIMEFMT = "%Y-%m-%d %H:%M:%S"

Q_CART_DTYPE = np.float32


def setup_args() -> argparse.Namespace:
    """Sets up the arguments for use in the terminal
    """
    bool_choices = ["false", "true"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_scobj_dir",
        type=str,
        default=None,
        help=(
            "Reference scattering object dir, only used to evaluate the predictions at the end."
        ),
    )
    parser.add_argument(
        "--input_meas_dir",
        type=str,
        help=(
            "Used to grab the measurements, but input_scobj_dir is "
            "specified separately for increased flexibility"
        ),
    )
    parser.add_argument(
        "--init_scobj_dir",
        type=str,
        default=None,
        nargs="?", # skip argument to set as None
        help=(
            "Optionally set these scattering potentials as the initialization "
            "(in case we want to chain NN+GN/RL methods together)"
        ),
    )
    parser.add_argument("--output_scobj_dir", type=str)
    parser.add_argument(
        "--dset",
        type=str,
        help="Optionally handle the dir/{dset}_measurements... part of the paths for inputs/outputs",
    )
    parser.add_argument(
        "--scobj_dir_format",
        type=str,
        default="{0}_scattering_objs",
        help="Can override the default scobj directory formatting",
    )
    parser.add_argument(
        "--input_meas_dir_format",
        type=str,
        default="{0}_measurements_nu_{1}",
        help="Can override the default measurements directory formatting",
    )
    parser.add_argument(
        "--output_scobj_fp_format",
        type=str,
        default="scattering_objs_{0}.h5",
        help="Can override the default scattering objects formatting",
    )

    parser.add_argument(
        "--sample_idx_start",
        type=int,
        help="Specify the index of the first sample to process in the input_scobj_dir",
    )
    parser.add_argument(
        "--sample_idx_end",
        type=int,
        default=None,
        help="Specify the index of the last sample to process in the input_scobj_dir",
    )
    parser.add_argument(
        "--sample_idx_count",
        type=int,
        default=None,
        help="Specify the number of samples to load from input_scobj_dir; overrides sample_idx_end",
    ) # alternate option
    parser.add_argument("--data_input_nus", type=str, nargs="+")

    # parser.add_argument("--input_fp", type=str)
    # parser.add_argument("--output_fp", type=str)
    # parser.add_argument("--nu_source_freq", type=float)  # non-angular

    # HPS solver arguments
    parser.add_argument("--hps_l", type=int, help="Number of HPS quadtree levels")
    parser.add_argument(
        "--hps_p", type=int,
        help="Polynomial order of leaf-level Chebyshev grid"
    )
    parser.add_argument(
        "--hps_comp_domain_factor", type=float, default=1.,
        help="How much larger of a computational domain to use for the HPS quadtree"
    )
    parser.add_argument(
        "--hps_sd_mat_dir", type=str,
        default="HPS_SD_matrices",
        help="Directory where to find the interior S and D scattering matrix files",
    )
    parser.add_argument(
        "--jax_mem_alloc_mb", type=int,
        default=3072,
        help="Amount of VRAM to allocate to JAX, in MB"
    )

    # General arguments
    parser.add_argument("--receiver_radius", type=float, default=100.0)

    parser.add_argument("--write_every_n", type=int, default=1)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=(
            "Logging level for this script's own messages (and any third-party "
            "loggers). Note jaxhps only ever logs at DEBUG, so DEBUG will include "
            "its (verbose) internal messages; INFO will not."
        ),
    )

    # RecLin arguments
    parser.add_argument("--gn_iters_per_freq", type=int, default=1)
    parser.add_argument("--gn_step_size", type=float, default=1.0)
    parser.add_argument("--gn_eps", type=float, default=0.0)
    parser.add_argument(
        "--allow_increase_error", choices=bool_choices, default="false",
        help="If set to true, skips GN step that would increase measurement error"
    )
    parser.add_argument("--cg_rtol", type=float, default=1e-4)
    parser.add_argument("--cg_max_iters", type=int, default=20)
    parser.add_argument(
        "--fbp_override", choices=bool_choices, default="false",
        help="Override settings specifically for filtered back-projection",
    )
    parser.add_argument("--fbp_iters", type=int, default=1)
    parser.add_argument("--fbp_step_size", type=float, default=1.0)
    parser.add_argument("--fbp_eps", type=float, default=0.0)
    parser.add_argument(
        "--check_meas_err", choices=bool_choices, default="false",
        help="Check the error of Fk[q_ref] (for debugging)",
    )
    parser.add_argument(
        "--check_scobj_err", choices=bool_choices, default="false",
        help="Check the error of q-hat during RL",
    )
    parser.add_argument(
        "--overwrite_if_settings_change", choices=bool_choices, default="false",
        help="Check the error of q-hat during RL",
    )
    parser.add_argument(
        "--noise_to_signal_ratio", default=0.0, type=float
    )
    parser.add_argument("--use_noise_seed", choices=bool_choices, default="false")
    parser.add_argument("--noise_seed_list", type=int, nargs="*", default=None)
    parser.add_argument("--noise_norm_mode", choices=["l2", "inf"], default="inf")

    a = parser.parse_args()

    bool_args = [
        "check_meas_err",
        "check_scobj_err",
        "overwrite_if_settings_change",
        "fbp_override",
        "use_noise_seed",
        "allow_increase_error",
    ]
    for bool_arg in bool_args:
        str_val = getattr(a, bool_arg)
        setattr(a, bool_arg, str_val == "true")

    if a.sample_idx_count is not None:
        a.sample_idx_end = a.sample_idx_start + a.sample_idx_count
    elif a.sample_idx_end is not None:
        a.sample_idx_count = a.sample_idx_end - a.sample_idx_start
    else:
        raise ValueError(f"Did not receive a value for either sample_idx_end or sample_idx_count")

    return a

def create_or_get_rl_data_file(
    meas_dd: dict,
    out_fp: str,
    args: dict,
    extras: dict=None,
    q_cart_init: np.ndarray=None,
    overwrite_if_extras_change: bool=False,
) -> dict:
    """Creates a blank file with the desired settings and sets up empty fields
    (in case the target output does not yet exist)
    Parameters:
        scobj_dd (dict): scattering object dict
        out_fp (str): output datafile filepath
        args (dict): the rest of the arguments passed to this script
        q_cart_init (np.ndarray): initialization for q predictions
        overwrite_if_extras_change (bool): if this flag is set to True and
            the values of the entries in extras do not match what is in out_fp
            then this function will overwrite the existing file

    Output:
        None (simply creates the new file at out_fp and will raise an error if it exists already)
    """
    global Q_CART_DTYPE

    extras = extras if extras is not None else dict()
    if os.path.exists(out_fp):
        # raise ValueError(
        #     f"Output file {out_fp} exists; not overwriting it"
        # )
        rl_settings = load_hdf5_to_dict(out_fp)
        overwrite_out_fp = False
        if overwrite_if_extras_change:
            extras_entries_changed = not (
                all(ek in rl_settings.keys() for ek in extras.keys())
                and all(np.all(rl_settings[ek]==ev) for (ek, ev) in extras.items())
            )
            overwrite_out_fp = overwrite_out_fp or extras_entries_changed
        print(f"overwrite_if_extras_change={overwrite_if_extras_change}")
        print(f"overwrite_out_fp={overwrite_out_fp}")
        if not overwrite_out_fp:
            logging.info(f"Loading {out_fp}")
            return rl_settings
        logging.info(f"Extra settings have changed; will overwrite {out_fp}")
        out_extras = {ek: rl_settings[ek] for ek in extras.keys() if ek in rl_settings.keys()}
        out_extras_str = pretty_dict_to_str(out_extras, indent_width=4)
        new_extras_str = pretty_dict_to_str(extras, indent_width=4)
        logging.info(f"Old settings\n{out_extras_str}")
        logging.info(f"New settings\n{new_extras_str}")

    nu_sf      = np.array(list(map(float, args.data_input_nus)))
    omega_sf   = 2*np.pi * nu_sf
    x_vals     = meas_dd[X_VALS]

    N_k        = nu_sf.shape[0]
    N_x        = x_vals.shape[0]
    N_samples  = meas_dd[SAMPLE_COMPLETION].shape[0]

    q_cart_shape = (N_samples, N_x, N_x)
    q_cart_out = np.full(q_cart_shape, np.nan, dtype=Q_CART_DTYPE)

    # Have 1+N_k rather than N_k so we can store initialization too
    q_cart_tmp_shape = (N_samples, 1+N_k, N_x, N_x)
    q_cart_tmp = np.full(q_cart_tmp_shape, np.nan, dtype=Q_CART_DTYPE)

    # Convenience settings
    sample_progress   = np.zeros((N_samples, 1+N_k), dtype=bool)
    sample_completion = np.zeros(N_samples, dtype=bool)
    file_completion = np.array([False])

    # Set the initialization
    q_cart_tmp[:, 0] = 0 if q_cart_init is None else q_cart_init
    sample_progress[:, 0] = True

    new_settings = {
        X_VALS: x_vals,
        # Scattering objects
        Q_CART: q_cart_out,
        "q_cart_tmp": q_cart_tmp,
        # Frequency info
        NU_SF: nu_sf,
        OMEGA_SF: omega_sf,

        # Tracking progress
        SAMPLE_PROGRESS: sample_progress,
        SAMPLE_COMPLETION: sample_completion,
        FILE_COMPLETION: file_completion,

        # Can pass extra stuff to include
        **extras,
    }
    out_settings = {**meas_dd, **new_settings}
    os.makedirs(os.path.split(out_fp)[0], exist_ok=True)
    save_dict_to_hdf5(out_settings, out_fp)
    return out_settings

def evaluate_q_cart(
    ref_scobj_dir: str,
    q_cart_eval: np.ndarray,
    args: dict,
) -> dict:
    logging.info(f"Reference scobj dir: {ref_scobj_dir}")
    ref_scobj_dd = load_single_dir_slice(
        ref_scobj_dir,
        args.sample_idx_start,
        args.sample_idx_end,
        # Loads everything by default
        load_keys=[
            Q_CART,
            SAMPLE_COMPLETION,
        ],
        sample_keys=[
            Q_CART,
            SAMPLE_COMPLETION,
        ]
    )
    if not np.all(ref_scobj_dd[SAMPLE_COMPLETION]):
        raise ValueError(
            f"Not every reference scobj has been marked as complete! "
            f"Please double-check the directory {ref_scobj_dir} for "
            f"indices {args.sample_idx_start}:{args.sample_idx_end}."
        )
    q_cart_ref = ref_scobj_dd[Q_CART]
    N_samples  = q_cart_ref.shape[0]

    loss_module_0 = MSEModule()
    error_metric_keys  = ["mse", "psnr", "rel_l2"]
    error_metric_attrs = ["mse", "psnr", "relative_l2_error"]
    cart_loss_fn_dd = {
        **{
            f"cart_{k}": getattr(loss_module_0, f"{k_a}")
            for (k, k_a) in zip(error_metric_keys, error_metric_attrs)
        },
        **{
            f"cart_final_{k}": getattr(loss_module_0, f"{k_a}_against_final")
            for (k, k_a) in zip(error_metric_keys, error_metric_attrs)
        },
    }
    cart_out_dd = {
        loss_key: np.full(N_samples, np.nan, dtype=np.float64)
        for loss_key in cart_loss_fn_dd.keys()
    }
    for loss_key, loss_fn in cart_loss_fn_dd.items():
        cart_out_dd[loss_key][:] = loss_fn(
            torch.tensor(q_cart_eval),
            torch.tensor(q_cart_ref),
            torch.tensor(q_cart_ref),
        ).cpu().numpy()
        logging.info(f"{loss_key} values: {cart_out_dd[loss_key]}")

    logging.info(f"Aggregate numbers...")
    for loss_key, loss_vals in cart_out_dd.items():
        loss_mean  = loss_vals.mean()
        loss_stdev = loss_vals.std()
        logging.info(f"Overall {loss_key}: {loss_mean:.6e}±{loss_stdev:.3e}")

    return cart_out_dd


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

def rel_err_fn(x, ref, **kwargs):
    return jnp.linalg.norm(x-ref, **kwargs) / jnp.linalg.norm(ref)


def main(args: argparse.Namespace) -> None:
    ###########################################################################
    # Setup and finish processing arguments
    logging.info(f"Received arguments {vars(args)}")

    if args.use_noise_seed and len(args.noise_seed_list) > 0:
        eff_noise_seed_list = args.noise_seed_list
    else:
        eff_noise_seed_list = None

    if args.noise_to_signal_ratio != 0:
        logging.info(f"Using seed as {eff_noise_seed_list}")
    else:
        logging.info(f"Not adding noise!")

    ##### 1. Prepare input/output paths #####
    # Append dset_scattering_objs to the path names
    # kbar_init_str = args.data_input_nus[0]
    # Load the initial measurement directory just to set up the
    kbar_list = np.array([float(kbar) for kbar in args.data_input_nus])
    first_input_meas_dir = os.path.join(
        args.input_meas_dir,
        args.input_meas_dir_format.format(args.dset, args.data_input_nus[0])
    )
    logging.info(f"Initial input measurement dir: {first_input_meas_dir}")

    # Prepare the output directory
    output_scobj_dir = os.path.join(
        args.output_scobj_dir,
        args.scobj_dir_format.format(args.dset),
    )
    os.makedirs(args.output_scobj_dir, exist_ok=True)
    output_scobj_fp = os.path.join(
        output_scobj_dir,
        args.output_scobj_fp_format.format(args.sample_idx_start),
    )
    logging.info(f"Output file path: {output_scobj_fp}")

    # Prepare the reference scattering objects directory, but only use later
    if args.ref_scobj_dir is not None:
        ref_scobj_dir = os.path.join(
            args.ref_scobj_dir,
            args.scobj_dir_format.format(args.dset),
        )
    else:
        ref_scobj_dir = None
    logging.info(f"Reference scobj dir: {ref_scobj_dir}")


    ##### 2. Load initial measurement data  #####

    # Load the relevant scattering object data
    # Load measurement data as we go through it, since
    # it may not be necessary to load everything anyway
    first_input_meas_dd = load_single_dir_slice(
        first_input_meas_dir,
        args.sample_idx_start,
        args.sample_idx_end,
        # Loads everything by default
        ignore_keys=[
            "m_vals",
            "h_vals",
            D_MH,
            Q_POLAR,
            Q_CART_LPF,
            Q_POLAR_LPF,
        ],
        sample_keys=[
            D_RS,
            Q_CART,
            SAMPLE_COMPLETION,
            SAMPLE_PROGRESS,
        ]
    )
    first_input_meas_dd_short = dict(kv_shrinker(k,v) for (k,v) in first_input_meas_dd.items())
    logging.info(f"Received input_scobj_dd with keys/shapes {first_input_meas_dd_short}")

    # Verify that the loaded measurements are valid
    if not np.all(first_input_meas_dd[SAMPLE_COMPLETION]):
        raise ValueError(
            f"The loaded scattering objects have not all been marked as completed; "
            f"please double-check the files containing sample slice "
            f" {args.sample_idx_start}:{args.sample_idx_end}."
        )

    # Prepare the output file
    if not args.fbp_override:
        args.fbp_iters     = args.gn_iters_per_freq
        args.fbp_step_size = args.gn_step_size
        args.fbp_eps       = args.gn_eps
    extras = {
        "input_meas_dir_base": bytes(args.input_meas_dir, "utf-8"),
        "sample_idx_start": args.sample_idx_start,
        "sample_idx_end":   args.sample_idx_end,
        "sample_idx_count": args.sample_idx_count,
        "hps_l": args.hps_l,
        "hps_p": args.hps_p,
        "hps_comp_domain_factor": args.hps_comp_domain_factor,
        "gn_eps":            args.gn_eps,
        "gn_step_size":      args.gn_step_size,
        "gn_iters_per_freq": args.gn_iters_per_freq,
        "cg_rtol":           args.cg_rtol,
        "cg_max_iters":      args.cg_max_iters,
        "fbp_iters":         args.fbp_iters,
        "fbp_step_size":     args.fbp_step_size,
        "fbp_eps":           args.fbp_eps,
        "kbar_list":         kbar_list, # Check the frequency list too!
        "allow_increase_error": args.allow_increase_error,
    }
    if args.noise_to_signal_ratio != 0:
        extras["noise_level"] = args.noise_to_signal_ratio,

    q_cart_init = None # gets set to zero by default
    if args.init_scobj_dir is not None:
        init_scobj_dir = os.path.join(
            args.init_scobj_dir,
            # f"{args.dset}_scattering_objs",
            args.scobj_dir_format.format(args.dset),
        )
        logging.info(f"Init scobj dir: {init_scobj_dir}")
        init_scobj_dd = load_single_dir_slice(
            init_scobj_dir,
            args.sample_idx_start,
            args.sample_idx_end,
            # Loads everything by default
            load_keys=[Q_CART, SAMPLE_COMPLETION],
            sample_keys=[Q_CART, SAMPLE_COMPLETION],
        )
        q_cart_init = init_scobj_dd[Q_CART]

    # Load the reference objects if available
    # Only used for checking errors
    if ref_scobj_dir is not None:
        ref_scobj_dd = load_single_dir_slice(
            ref_scobj_dir,
            args.sample_idx_start,
            args.sample_idx_end,
            # Loads everything by default
            load_keys=[
                Q_CART,
                SAMPLE_COMPLETION,
            ],
            sample_keys=[
                Q_CART,
                SAMPLE_COMPLETION,
            ]
        )
        ref_q_cart = ref_scobj_dd[Q_CART]
    else:
        ref_q_cart = None

    output_scobj_dd = create_or_get_rl_data_file(
        first_input_meas_dd,
        output_scobj_fp,
        args,
        extras=extras,
        q_cart_init=q_cart_init,
        overwrite_if_extras_change=args.overwrite_if_settings_change,
    )
    output_scobj_dd_short = dict(kv_shrinker(k,v) for (k,v) in output_scobj_dd.items())
    logging.info(f"Loaded output_scobj_dd with keys/shapes {output_scobj_dd_short}")

    # Check the outputs for completion
    if output_scobj_dd[FILE_COMPLETION] == True:
        logging.info(f"The file {output_scobj_fp} has already been marked as complete!")
        _ = evaluate_q_cart(
            ref_scobj_dir,
            q_cart_eval=output_scobj_dd[Q_CART],
            args=args,
        )
        return

    # Extract values
    q_cart_tmp        = output_scobj_dd["q_cart_tmp"]
    sample_completion = output_scobj_dd[SAMPLE_COMPLETION]
    sample_progress   = output_scobj_dd[SAMPLE_PROGRESS]
    x_vals = output_scobj_dd[X_VALS]
    N_x = x_vals.shape[0]
    spatial_domain_max = np.max(np.abs(x_vals))

    # Setup for the HPS settings
    L = args.hps_l
    p = args.hps_p
    hps_cdf = args.hps_comp_domain_factor
    hpst_spatial_domain_max = hps_cdf * spatial_domain_max
    R = args.receiver_radius

    # Simplification... just set these values here
    N_r = N_x
    N_s = N_x
    hps_source_dirs   = jnp.pi/2-jnp.linspace(0, 2*jnp.pi, N_s, endpoint=False)
    hps_reciever_dirs = jnp.pi/2-jnp.linspace(0, 2*jnp.pi, N_r, endpoint=False)

    unif_domain_bounds = spatial_domain_max * np.array([-1., 1., -1., 1.])
    hpst_domain_bounds = hps_cdf * unif_domain_bounds
    hpst_root   = DiscretizationNode2D(*hpst_domain_bounds)
    hpst_domain = Domain(p=p, q=p-2, root=hpst_root, L=L)

    jax_device = jax.config.jax_default_device
    QtU = None
    UtQ = None

    for ti, kbar_str in enumerate(args.data_input_nus, start=1):
        # Internally switching to calling it kbar rather than nu
        # as in, k divided by 2pi
        kbar = float(kbar_str)
        k = kbar * 2*np.pi
        logging.info(f"Frequency k_{ti}/2pi={kbar_str}")

        # First, check whether we even need to set up the solver
        if np.all(sample_progress[:, ti]):
            logging.info(f"This frequency is already finished! Continuing...")
            continue

        # Set up the solver
        # logging.info(f"Setting up the solver...")
        SD_matrices_fp = get_SD_matrices_fp(
            kbar_str=kbar_str,
            L=L,
            p=p,
            domain_half_length=hpst_spatial_domain_max,
            SD_matrices_dir=args.hps_sd_mat_dir,
        )
        # logging.info(f"SD_matrices_fp={SD_matrices_fp}")
        S_int, D_int = load_SD_matrices(SD_matrices_fp)

        hss_kt = HPSScatteringSolver(
            L=L, p=p, N_x=N_x, k=k,
            S_int=S_int,
            D_int=D_int,
            # S_ext=S_ext, # optional
            # D_ext=D_ext, # optional
            N_r=N_r,
            N_s=N_s,
            unif_domain_bounds=unif_domain_bounds,
            quad_domain_bounds=hpst_domain_bounds,
            use_ItI=True,
            QtU=QtU,
            UtQ=UtQ
        )
        # Save for re-use
        QtU = hss_kt.QtU
        UtQ = hss_kt.UtQ

        # Load measurement data
        input_meas_kt_dir = os.path.join(
            args.input_meas_dir,
            # f"{args.dset}_measurements_nu_{kbar_str}",
            args.input_meas_dir_format.format(args.dset, kbar_str)
        )
        logging.info(f"Loading the measurement data from {input_meas_kt_dir}")
        input_meas_kt_dd = load_single_dir_slice(
            input_meas_kt_dir,
            args.sample_idx_start,
            args.sample_idx_end,
            load_keys=[
                D_RS,
                SAMPLE_COMPLETION,
                FILE_COMPLETION,
            ],
            sample_keys=[
                D_RS,
                SAMPLE_COMPLETION,
                SAMPLE_PROGRESS,
            ]
        )
        d_rs_kt = input_meas_kt_dd[D_RS]
        if args.noise_to_signal_ratio != 0:
            eff_noise_seed_list = args.noise_seed_list if args.use_noise_seed else None
            eff_noise_seed_ti   = eff_noise_seed_list[ti-1]
            logging.info(
                f"Note: adding noise at a relative level of {args.noise_to_signal_ratio} "
                f"(noise seed={eff_noise_seed_ti})"
            )
            d_rs_kt = add_noise_to_d(
                d_rs_kt,
                args.noise_to_signal_ratio,
                noise_seed=eff_noise_seed_ti,
                seed_mode="sequential",
                norm_mode=args.noise_norm_mode,
            )
        else:
            logging.info(f"Note: not adding noise")

        logging.info(f"Loaded d_rs with shape {d_rs_kt.shape}")

        N_samples = d_rs_kt.shape[0]
        chunk_idx_start = 0
        chunk_idx_end   = chunk_idx_start + args.write_every_n
        made_changes_in_chunk = False

        # Grab the settings for Gauss-Newton
        gn_iters = args.gn_iters_per_freq
        gn_step_size = args.gn_step_size
        gn_eps       = args.gn_eps
        cg_rtol      = args.cg_rtol
        cg_iters     = args.cg_max_iters
        if args.init_scobj_dir is None and ti == 1:
            logging.info(f"Filtered back-projection")
            if args.fbp_override:
                logging.info(f"Overriding settings...")
                gn_iters = args.fbp_iters
                gn_step_size = args.fbp_step_size
                gn_eps       = args.fbp_eps
        logging.info(
            f"(k/2pi={kbar_str}) will use {gn_iters} iters "
            f"with step size {gn_step_size:.3f} and eps {gn_eps:.3e}"
        )

        sample_times = []
        t0 = time.perf_counter()
        # t_after_first = None
        for i in range(N_samples):
            logging.info(f"Sample {i+1}")
            t_sample_start = time.perf_counter()
            curr_qi_t = q_cart_tmp[i, ti]
            curr_qi_t_valid = not np.any(np.isnan(curr_qi_t))
            already_done_i = sample_progress[i, ti] and curr_qi_t_valid
            if already_done_i:
                qi_t_unif = curr_qi_t
                logging.info(f"Already done!")
            else:
                # Check scattering potential error
                if ref_q_cart is not None and args.check_scobj_err:
                    qi_tm1_err = rel_err_fn(
                        q_cart_tmp[i,ti-1],
                        ref_q_cart[i],
                    )
                    logging.info(f"Incoming scattering potential error: {qi_tm1_err:.5e}")

                qi_tm1_unif = q_cart_tmp[i, ti-1]
                qi_tm1_hpst = UtQ.apply(qi_tm1_unif)
                # logging.info(f"qi_{{t-1}} nans? {np.any(np.isnan(qi_tm1_unif))}")

                qi_t_hpst, qi_t_hpst_gn_list = gauss_newton_loop_single_sample(
                    hss_kt,
                    d_rs_kt[i],
                    q_init=qi_tm1_hpst,
                    gn_iters=gn_iters,
                    gn_step_size=gn_step_size,
                    gn_eps=gn_eps,
                    cg_rtol=cg_rtol,
                    cg_iters=cg_iters,
                    verbosity=3,
                    allow_increase_error=args.allow_increase_error,
                )
                qi_t_unif = QtU.apply(qi_t_hpst)
                # qi_t_unif_gn_list = [QtU.apply(qi) for qi in qi_t_hpst_gn_list]
                made_changes_in_chunk = True

                # Check scattering potential error
                if ref_q_cart is not None and args.check_scobj_err:
                    qi_t_err = rel_err_fn(
                        qi_t_unif,
                        ref_q_cart[i],
                    )
                    logging.info(f"Scattering potential error: {qi_t_err:.5e}")

                # Check measurement error...
                if args.check_meas_err:
                    qi_t_solver = SharedSolver(
                        hss_kt,
                        qi_t_hpst,
                        # UtQ.apply(ref_q_cart[i]), # using ref scobj instead just to debug...
                    )
                    Fkt_q_hps = qi_t_solver.forward_exterior().T
                    Fkt_q_hps_err = rel_err_fn(
                        Fkt_q_hps,
                        d_rs_kt[i],
                    )
                    logging.info(f"Measurement error: {Fkt_q_hps_err:.5e}")
                # if t_after_first is None:
                #     t_after_first = time.perf_counter()
                sample_time = time.perf_counter() - t_sample_start
                sample_times.append(sample_time)


            # logging.info(f"Update sample_progress etc....")
            q_cart_tmp[i, ti] = qi_t_unif
            sample_progress[i, ti] = True

            vram_msg = get_memory_info_jax(jax_device, print_msg=False)
            logging.info(f"{vram_msg}")

            # logging.info(f"Consider clearing the jax caches after each sample if there are still problems")
            # jax.clear_caches()

            if (i+1) % args.write_every_n == 0 or (i+1==N_samples):
                if made_changes_in_chunk:
                    logging.info(f"Writing to disk!")
                    # Write the updates to disk
                    # Just to be safe, differentiate between the chunk slice
                    # with indices of the loaded arrays vs. the file's arrays
                    # For now they'll be the same
                    chunk_slice_loaded = np.s_[chunk_idx_start:chunk_idx_end, ti]
                    chunk_slice_file   = np.s_[chunk_idx_start:chunk_idx_end, ti]

                    update_field_in_hdf5(
                        "q_cart_tmp",
                        q_cart_tmp[chunk_slice_loaded],
                        output_scobj_fp,
                        chunk_slice_file,
                    )
                    update_field_in_hdf5(
                        SAMPLE_PROGRESS,
                        sample_progress[chunk_slice_loaded],
                        output_scobj_fp,
                        chunk_slice_file,
                    )
                else:
                    logging.info(f"Skipping write to disk as nothing in the chunk has changed")

                # Update chunk-tracking variables
                chunk_idx_start = min(N_samples, args.write_every_n+chunk_idx_start)
                chunk_idx_end   = min(N_samples, args.write_every_n+chunk_idx_end)
                made_changes_in_chunk = False

        t1 = time.perf_counter()
        # freq_time_total = t1-t0
        N_proc = len(sample_times)
        freq_time_total = np.sum(sample_times)
        freq_time_per_sample = freq_time_total / N_proc
        logging.info(
            f"Frequency k_{ti}/2pi={kbar_str} finished in {freq_time_total:.3f}s "
            f"(on average, {freq_time_per_sample:.3f}s/sample)"
        )
        if N_proc > 1:
            freq_time_per_sample_excl_first = np.sum(sample_times[1:]) / (N_proc - 1)
            logging.info(
                f"When excluding the first sample, the runtime averages to "
                f"{freq_time_per_sample_excl_first:.3f}s/sample"
            )
        vram_msg = get_memory_info_jax(jax_device, print_msg=False)
        logging.info(f"{vram_msg}")

        logging.info(f"Clearing jax caches...")
        jax.clear_caches()


    logging.info(f"Updating q_cart, sample_completion, and file_completion as needed")
    # Re-load the sample completion, sample_progress, and q_cart_tmp components from disk
    # in case somehow multiple scripts are working on the file at once
    sample_progress = load_field_in_hdf5(SAMPLE_PROGRESS, output_scobj_fp)
    q_cart_tmp      = load_field_in_hdf5("q_cart_tmp", output_scobj_fp)
    q_cart_out      = q_cart_tmp[:, -1]

    print(f"sample_progress={sample_progress}")
    print(f"q_cart_tmp shape={q_cart_tmp.shape}")
    print(f"q_cart_out shape={q_cart_out.shape}")

    # Update file if every sample is complete
    sample_completion = np.all(sample_progress, axis=1)
    file_completion   = np.all(sample_completion)
    if file_completion:
        # Update q_cart
        update_field_in_hdf5(
            Q_CART,
            q_cart_out,
            output_scobj_fp,
        )
    update_field_in_hdf5(
        SAMPLE_COMPLETION,
        sample_completion,
        output_scobj_fp,
    )
    update_field_in_hdf5(
        FILE_COMPLETION,
        file_completion,
        output_scobj_fp,
    )
    if not file_completion:
        logging.info(f"Caution: exiting but marking the file as incomplete!")
        return
    else:
        logging.info(f"Marking the file as complete!")

    _ = evaluate_q_cart(
        ref_scobj_dir,
        q_cart_eval=q_cart_out,
        args=args,
    )


if __name__ == "__main__":
    a = setup_args()
    # a.seed = eval(a.seed)

    root = logging.getLogger()

    handler = logging.StreamHandler(sys.stderr)
    # --debug is kept as a shorthand for full DEBUG output (including jaxhps's
    # own logging.debug(...) calls, which are otherwise never shown since
    # jaxhps never logs above DEBUG); --log_level gives finer control.
    log_level = logging.DEBUG if a.debug else getattr(logging, a.log_level)
    handler.level = log_level
    root.setLevel(log_level)

    formatter = logging.Formatter(FMT, datefmt=TIMEFMT)
    handler.setFormatter(formatter)
    root.addHandler(handler)

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

    # logging.info(f"disabling async dispatch for debugging")
    # print(f"disabling async dispatch for debugging")
    # jax.config.update('jax_cpu_enable_async_dispatch', False)

    # Well unfortunately I need the VRAM calculations
    # before loading jax
    from src.utils.vram_info_jax import (
        get_memory_info_jax,
        # get_vram_total_mb_jax,
        # vram_mb_to_frac_jax,
    )
    import jax.numpy as jnp
    from jaxhps import (
        DiscretizationNode2D,
        Domain,
    )
    from solvers.hps.wave_scattering import (
        gen_S_exterior,
        gen_D_exterior,
        get_SD_matrices_fp,
        load_SD_matrices,
        HPSScatteringSolver,
        SharedSolver,
        # GaussNewtonOperator,
        gauss_newton_loop_single_sample,
    )

    logging.info(f"Start: run_recursive_linearization.py")

    main(a)
