"""
This script does the following:
1. Grabs the scattering object file
2. Generates back-projected differences DF[q]^*(dk-F[q])
3. Saves the data to disk.
"""

import argparse
import logging
import os
import sys
from typing import Dict
import numpy as np
import wandb

from src.data.data_io import (
    save_dict_to_hdf5,
    load_hdf5_to_dict,
    load_field_in_hdf5,
    update_field_in_hdf5,
    load_single_dir_slice,
)
from src.data.layout import get_file_start_index
from src.data.add_noise import add_noise_to_d

from src.data.data_naming_constants import (
    NU_SF,
    OMEGA_SF,
    X_VALS,
    FILE_COMPLETION,
    SAMPLE_COMPLETION,
    Q_CART,
    Q_CART_LPF,
    Q_POLAR,
    Q_POLAR_LPF,
    D_RS,
    GAMMA_CART,
    KEYS_FOR_BPMD_SAMPLES,
)
from src.utils.logging_utils import hash_dict
from src.utils.vram_info import get_memory_info, get_vram_total_mb, vram_mb_to_frac
import psutil

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
    import jax
    device = jax.devices("gpu")[0]
    jax.config.update("jax_default_device", device)
    jax.config.update("jax_enable_x64", True)
else:
    import jax
import jax.numpy as jnp
from solvers.hps.wave_scattering import (
    gen_S_exterior,
    gen_D_exterior,
    get_SD_matrices_fp,
    load_SD_matrices,
    HPSScatteringSolver,
    SharedSolver,
)

FMT = "%(asctime)s:generate-data: %(levelname)s - %(message)s"
TIMEFMT = "%Y-%m-%d %H:%M:%S"


def wandb_entity_arg_type(value: str):
    """argparse type for --wandb_entity: treat "none"/"null" (any case) as
    no entity, so wandb.init() falls back to the caller's own default entity
    """
    if value is None or value.strip().lower() in ("none", "null"):
        return None
    return value

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_scobj_fp", type=str)
    parser.add_argument("--input_meas_dir", type=str)
    parser.add_argument("--output_fp", type=str)
    parser.add_argument("--kbar_source_freq", type=str)  # non-angular

    # HPS solver arguments
    parser.add_argument("--hps_l", type=int, help="Number of HPS quadtree levels")
    parser.add_argument(
        "--hps_p", type=int,
        help="Polynomial order of leaf-level Chebyshev grid"
    )
    parser.add_argument(
        "--hps_comp_domain_factor", type=float, default=1.1,
        help="How much larger of a computational domain to use for the HPS quadtree"
    )
    parser.add_argument(
        "--hps_sd_mat_dir", type=str,
        default="rlc_data/HPS_SD_matrices",
        help="Directory where to find the interior S and D scattering matrix files",
    )

    # General arguments
    parser.add_argument("--receiver_radius", type=float, default=100.0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument(
        "--create_n_samples", type=int, default=-1
    )  # by default do everything
    parser.add_argument("--write_every_n", type=int, default=1)

    parser.add_argument("--noise_to_signal_ratio", type=float, default=0.0)
    parser.add_argument("--use_noise_seed", choices=["true", "false"], default="false")
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--noise_norm_mode", choices=["l2", "inf"], default="inf")

    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--wandb_entity", type=wandb_entity_arg_type, default=None, help="The W&B entity")
    parser.add_argument("--wandb_project")
    parser.add_argument(
        "--wandb_mode", choices=["offline", "online"], default="offline"
    )
    parser.add_argument("--dont_use_wandb", default=False, action="store_true")

    a = parser.parse_args()
    a.kbar_str = a.kbar_source_freq
    a.kbar = float(a.kbar_str)
    a.use_noise_seed = (a.use_noise_seed == "true")

    return a


def create_new_bpmd_data_file(sdf_fp: str, bdf_fp: str, args: dict, create_dirs: bool=True) -> None:
    """Creates a blank file with the desired settings and sets up empty fields
    (in case the target output does not yet exist)
    Parameters:
        sdf_fp (str): scattering object datafile filepath
        bdf_fp (str): backprojected difference datafile filepath
        args (dict): the rest of the arguments passed to this script
        create_dir (bool): whether to create new directories in case

    Output:
        None (simply creates the new file at sdf_fp and will raise an error if it exists already)
    """
    scobj_dd = load_hdf5_to_dict(sdf_fp)
    num_samples = scobj_dd[SAMPLE_COMPLETION].shape[0]

    x_vals = scobj_dd[X_VALS]
    nu_sf = np.array([args.kbar])
    omega_sf = 2 * np.pi * nu_sf
    sample_completion = np.zeros(num_samples, dtype=bool)
    file_completion = np.array([False])

    q_cart = scobj_dd[Q_CART]
    # gamma_cart = np.full(q_cart.shape, np.nan, dtype=np.complex64)
    gamma_cart = np.full(q_cart.shape, np.nan, dtype=np.float32) # OOP this is actually real

    # Convenience settings
    sample_completion = np.zeros(num_samples, dtype=bool)
    file_completion = np.array([False])

    new_dd = {
        # Grid points
        X_VALS: x_vals,
        # Scattering objects
        # Q_CART: q_cart,
        GAMMA_CART: gamma_cart,

        # Frequency info
        NU_SF: nu_sf,
        OMEGA_SF: omega_sf,
        # Convenience settings
        SAMPLE_COMPLETION: sample_completion,
        FILE_COMPLETION: file_completion,
    }
    new_dd = {
        key: val if isinstance(val, np.ndarray) else np.array(val)
        for (key, val) in new_dd.items()
    }

    # Combine and overwrite dd as necessary
    bdf_dd = {**scobj_dd, **new_dd}
    drop_keys = [Q_POLAR] # Can add others if needed
    for dk in drop_keys:
        if dk in bdf_dd.keys():
            del bdf_dd[dk]

    if create_dirs:
        os.makedirs(os.path.split(bdf_fp)[0], exist_ok=True)
    save_dict_to_hdf5(bdf_dd, bdf_fp)
    return


def main(args: argparse.Namespace) -> None:
    ###########################################################################
    # Setup
    logging.info(f"Received arguments {vars(args)}")

    d = os.path.split(args.output_fp)[0]
    if not os.path.isdir(d):
        try:
            os.makedirs(d, exist_ok=True)
            logging.info(f"created directory: {d}")
        except:
            logging.warning(f"problem creating directory {d}; proceeding anyway...")
    else:
        logging.info(f"not creating dir {d} because it already exists")

    ### 1. Check whether the target backproj diff file is complete already
    # if so, we can avoid reading everything from disk
    try:
        bdf_already_complete = load_field_in_hdf5(
            FILE_COMPLETION, args.output_fp
        ).item()
    except:
        # Mark incomplete if the bdf doesn't exist
        # or has an issue with its "file_completion" field
        bdf_already_complete = False
    # Check for file completion here
    if bdf_already_complete == True:
        logging.warning(
            f"Backproj diff file marked complete; exiting early (file name: {args.output_fp})"
        )
        return

    # May raise a FileNotFoundError if the scattering file does not exist
    invalid_sdf = False
    if not os.path.exists(args.input_scobj_fp):
        invalid_sdf = True
        logging.error(f"Input scattering object file {args.input_scobj_fp} could not be found")
    else:
        # Check the "file_completion" flag
        sdf_already_complete = load_field_in_hdf5(
            FILE_COMPLETION, args.input_scobj_fp
        ).item()
        invalid_sdf = not sdf_already_complete
    if invalid_sdf:
        error_msg = (
            f"The input scattering object file at "
            f"{args.input_scobj_fp} appears to be invalid"
        )
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    else:
        num_samples_all = load_field_in_hdf5(
            SAMPLE_COMPLETION, args.input_scobj_fp
        ).shape[0]

    ### 2. Attempt to load the output backproj diff file if it exists
    bpmd_fp = args.output_fp
    # If no backproj diff file exists, create a new one
    try:
        logging.warning(f"Attempting to load settings from target file")
        if not os.path.exists(bpmd_fp):
            raise FileNotFoundError
        bpmd_settings = load_hdf5_to_dict(bpmd_fp)
        logging.debug("bpmd_settings: %s", bpmd_settings.keys())
    except Exception as e:
        # In case of an error while loading
        if os.path.exists(bpmd_fp):
            logging.warning(
                f"Deleting backproj diff output file {bpmd_fp} after"
                " encountering an error {e} while attempting to load output file"
            )
            os.remove(bpmd_fp)
        logging.warning(f"Creating new backproj diff file from scratch")
        create_new_bpmd_data_file(args.input_scobj_fp, bpmd_fp, args)
        bpmd_settings = load_hdf5_to_dict(bpmd_fp)


    # Try to load the measurement file corresponding to the scattering object
    try:
        # import pdb; pdb.set_trace()
        bdf_start = get_file_start_index(bpmd_fp)
        bdf_end   = bdf_start + bpmd_settings[SAMPLE_COMPLETION].shape[0]
        meas_dd   = load_single_dir_slice(
            args.input_meas_dir,
            global_idx_start=bdf_start,
            global_idx_end=bdf_end,
            ignore_keys=[Q_POLAR, Q_POLAR_LPF, Q_CART_LPF, Q_CART],
        )
        d_rs = meas_dd[D_RS]
        # Add noise as needed
        if args.noise_to_signal_ratio != 0:
            logging.info(
                f"Note: adding noise at a relative level of {args.noise_to_signal_ratio}"
            )
            eff_noise_seed = args.noise_seed if args.use_noise_seed else None
            d_rs = add_noise_to_d(
                d_rs,
                args.noise_to_signal_ratio,
                noise_seed=eff_noise_seed,
                seed_mode="sequential",
                norm_mode=args.noise_norm_mode,
            )
        else:
            logging.info(f"Note: not adding noise")
    except Exception as e:
        logging.info(f"Unable to load d_rs  from {args.input_meas_dir}")
        logging.info(f"Exception info: {str(e)}")
        exit(1)

    # Unload the settings into local variables
    # Grid variables
    kbar = args.kbar  # non-angular frequency of the source wave
    k = 2 * np.pi * kbar  # angular frequency of the source wave
    x_vals = bpmd_settings[X_VALS]
    N_x = x_vals.shape[0]
    sample_completion = bpmd_settings[SAMPLE_COMPLETION]
    # Just set N_r and N_s to match
    N_r = N_x
    N_s = N_x

    ### 3. set up the PDE solver and convolution operators
    logging.warning("Setting up solvers and convolution objects")
    spatial_domain_max = np.max(np.abs(x_vals))

    # Use Lippmann-Schwinger solver if the jaxhps code could not be imported
    logging.warning(f"Setting up HPS solver...")
    device = jax.devices("gpu")[0]
    # Convert non-angular wavenumber nu to a string
    # No decimals if it is an integer and just one otherwise
    # is_nu_integer = np.isclose(kbar, np.round(kbar))
    # nu_str = f"{int(kbar)}" if is_nu_integer else f"{kbar:.1f}"

    hps_cdf = args.hps_comp_domain_factor
    unif_domain_hlen = np.max(np.abs(x_vals))
    hpst_domain_hlen = hps_cdf * unif_domain_hlen
    sd_mat_fp = get_SD_matrices_fp(
        args.kbar_str,
        args.hps_l,
        args.hps_p,
        domain_half_length=hpst_domain_hlen,
        SD_matrices_dir=args.hps_sd_mat_dir
    )
    S_int, D_int = load_SD_matrices(sd_mat_fp)
    # xm = spatial_domain_max
    unif_bounds = spatial_domain_max * np.array([-1., 1., -1., 1.])
    hpst_bounds = hps_cdf * unif_bounds
    hps_solver = HPSScatteringSolver(
        L=args.hps_l,
        p=args.hps_p,
        N_x=N_x,
        k=k,
        S_int=S_int,
        D_int=D_int,
        N_r=N_r,
        N_s=N_s,
        unif_domain_bounds=unif_bounds,
        quad_domain_bounds=hpst_bounds,
    )
    logging.warning(f"Finished setting up the solver object")

    ### 4. Prepare the index range
    # num_samples_all = args.total_n_samples
    create_n_samples = (
        args.create_n_samples if args.create_n_samples != -1 else num_samples_all
    )
    args.end_idx = min(args.start_idx + create_n_samples, num_samples_all)
    full_slice = slice(args.start_idx, args.end_idx)
    num_samples_eff = args.end_idx - args.start_idx

    # Buffer variables
    # read in q from the measurement file
    q_cart_eff = bpmd_settings["q_cart"][full_slice]
    gamma_cart_eff = bpmd_settings["gamma_cart"][full_slice]

    ### 5. Filter and scatter the inputs
    logging.warning(
        f"Beginning to process {num_samples_eff} scattering objects"
    )

    # chunk_counter = 0 # absolute index from the beginning
    for chunk_start_idx in range(args.start_idx, args.end_idx, args.write_every_n):
        chunk_end_idx = min(chunk_start_idx + args.write_every_n, args.end_idx)
        made_changes_in_chunk = False # track whether we can skip the save step

        # Loop over the indices in the chunk
        for idx_abs in range(chunk_start_idx, chunk_end_idx):
            idx_eff = idx_abs - args.start_idx
            logging.warning("Working on sample %i of %i", idx_eff + 1, num_samples_eff)
            computed_soln_bool = None  # Leave blank for now...

            # It's possible the wave fields for this sample has already been computed, so we
            # want to skip computing it if possible
            is_any_scobj_nans = np.any(np.isnan(q_cart_eff[idx_eff])) or np.any(
                np.isnan(gamma_cart_eff[idx_eff])
            )

            # (Re-)do the back-projection PDE solve if necessary
            # First determine whether that is necessary
            is_any_data_nans = np.any(np.isnan(gamma_cart_eff[idx_eff]))
            is_all_data_nans = np.all(np.isnan(gamma_cart_eff[idx_eff]))
            logging.warning(f"Current entry has NaNs?    {is_any_data_nans}")
            # logging.warning(f"Current entry is all NaNs? {is_all_data_nans}")
            if not is_any_data_nans:
                if not sample_completion[idx_abs]:
                    made_changes_in_chunk = True # mark that we made a change in the sample_completion status
                sample_completion[idx_abs] = True
                logging.warning(
                    f"Identifying an existing solution at index {idx_eff} from lack of NaNs"
                )
            elif is_any_data_nans and sample_completion[idx_abs]:
                # sample should not have been marked complete yet
                sample_completion[idx_abs] = False

            already_present_bool = sample_completion[idx_abs]

            if already_present_bool:
                logging.warning("Solution at index %i is already present", idx_eff)
            else:
                made_changes_in_chunk = True
                # Now run the PDE solver in batches
                try:
                    # (OOT 2025-08-18) Should I give the option to save Fqi to disk??
                    qi_unif = q_cart_eff[idx_eff]
                    qi_hpst = hps_solver.UtQ.apply(qi_unif)
                    qi_solver = SharedSolver(hps_solver, qi_hpst, device=device)
                    qi_solver.forward_interior() # Currently we need a setup step from this
                    Fqi  = qi_solver.forward_exterior()
                    diff = d_rs[idx_eff].T - Fqi # convention is transposed wrt the dataset
                    DFh_qi_diff = qi_solver.vjp_exterior(diff)
                    # DFh_qi_diff = qi_solver.backproject_diff_exterior(
                    #     d_rs[idx_eff],
                    #     transpose_dk=True,
                    # )
                    gamma_cart_eff[idx_eff] = jax.device_put(hps_solver.QtU.apply(DFh_qi_diff))
                    computed_soln_bool = True
                except:
                    gamma_cart_eff[idx_eff] = np.nan
                    computed_soln_bool = False

            # Log updates from this sample sample
            sample_completion[idx_abs] = computed_soln_bool or already_present_bool
            sample_dd = {
                "i": idx_abs,
                "already_present_bool": already_present_bool,
                "computed_soln_bool": computed_soln_bool,
            }
            if not a.dont_use_wandb:
                wandbrun.log(sample_dd)
            # chunk_counter += 1

        # Write results to disk
        logging.warning(f"Saving data to disk (changes made? {made_changes_in_chunk})")

        # Write the d_rs and d_mh values to disk then mark sample as complete
        chunk_slice_abs = slice(chunk_start_idx, chunk_end_idx)
        chunk_slice_eff = slice(
            chunk_start_idx - args.start_idx, chunk_end_idx - args.start_idx
        )

        gamma_cart_nans = np.any(np.isnan(gamma_cart_eff[chunk_slice_eff]), axis=(1,2))
        logging.info(f"gamma_cart_nans after processing the chunk: {gamma_cart_nans}")

        if made_changes_in_chunk:
            update_field_in_hdf5(
                GAMMA_CART,
                gamma_cart_eff[chunk_slice_eff],
                bpmd_fp,
                chunk_slice_abs,
            )
            update_field_in_hdf5(
                SAMPLE_COMPLETION,
                sample_completion[chunk_slice_eff],
                bpmd_fp,
                chunk_slice_abs,
            )
        msg = get_memory_info()
        logging.info(f"{msg}")

    # Fetch from disk again in case someone else was working on the same file at the same time
    sample_completion_newest = load_field_in_hdf5(SAMPLE_COMPLETION, bpmd_fp)
    if np.all(sample_completion_newest):
        # If every sample has been completed then we can mark the file as complete
        # bdf_completion[0] = True
        bdf_completion = np.array([True])
        update_field_in_hdf5(
            FILE_COMPLETION,
            bdf_completion,
            bpmd_fp,
        )
        logging.warning(
            f"Marked the backproj diff file as complete! ( {bpmd_fp} )"
        )
    logging.warning("Finished")
    return


if __name__ == "__main__":
    a = setup_args()
    # a.seed = eval(a.seed)

    root = logging.getLogger()

    handler = logging.StreamHandler(sys.stderr)
    if a.debug:
        handler.level = logging.DEBUG
        root.setLevel(logging.DEBUG)
    else:
        handler.level = logging.WARNING
        root.setLevel(logging.WARNING)

    formatter = logging.Formatter(FMT, datefmt=TIMEFMT)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    hash_id = hash_dict(vars(a))

    # Calculate the vram fraction required to have jax_mem_alloc_mb allocated to jax
    jax_mem_alloc_frac = 0.6
    jax_mem_alloc_mb = int(jax_mem_alloc_frac * get_vram_total_mb())
    logging.info(f"JAX VRAM allocation: {jax_mem_alloc_frac:.3f} (in MB: {jax_mem_alloc_mb})")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{jax_mem_alloc_frac:.3f}"
    import jax
    jax_device = jax.devices("gpu")[0]
    jax.config.update("jax_default_device", jax_device)
    jax.config.update("jax_enable_x64", True)
    jax_device = jax.devices("gpu")[0]
    logging.info(f"Note: Jax has been loaded :)")


    logging.info(f"Start: generate_backproj_diff_files.py")

    if a.dont_use_wandb:
        main(a)
    else:
        with wandb.init(
            id=hash_id,
            project=a.wandb_project,
            entity=a.wandb_entity,
            config=vars(a),
            mode=a.wandb_mode,
            reinit=True,
            resume=None,
            settings=wandb.Settings(start_method="fork"),
        ) as wandbrun:
            try:
                main(a)
            except Exception as e:
                logging.error(f"Fatal Error encountered: {e}")
                logging.error(f"generate_backproj_diffs_files.py terminating early")
