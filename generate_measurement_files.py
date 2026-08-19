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

from solvers.integral_equation.helmholtz_solver_bicgstab import (
    DEFAULT_RTOL,
    setup_bicgstab_solver,
)
from src.data.data_transformations import (
    prep_rs_to_mh_interp,
    apply_interp_2d,
    get_scale_factor,
    CONST_RHO_PRIME,
    CONST_THETA_PRIME,
    polar_to_euclidean,
    prep_conv_interp_2d,
)
from src.data.lowpass_filter import (
    prep_lpf_from_wavenum,
    apply_filter_fourier_2d,
)

from src.data.data_io import (
    save_dict_to_hdf5,
    load_hdf5_to_dict,
    load_field_in_hdf5,
    update_field_in_hdf5,
    load_single_dir_slice,
)
from src.utils.logging_utils import hash_dict
from torch._C import _LinAlgError
import torch.cuda
import psutil

try:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
    import jax
    import jax.numpy as jnp
    # import jaxhps
    from solvers.hps.wave_scattering import (
        get_SD_matrices_fp,
        gen_S_exterior,
        gen_D_exterior,
        load_SD_matrices,
        HPSScatteringSolver,
    )
    device = jax.devices("gpu")[0]
    jax.config.update("jax_default_device", device)
    jax.config.update("jax_enable_x64", True)
    JAXHPS_IMPORTED = True
except:
    JAXHPS_IMPORTED = False


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

    parser.add_argument("--input_fp", type=str)
    parser.add_argument("--output_fp", type=str)
    # 2026-08-05: Alternate interface when a one-to-one file mapping does not work
    # (i.e., due to different shard sizes)
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory of scattering-object shard files to read a "
        "[--global_idx_start, --global_idx_end) slice from. Alternative to "
        "--input_fp; if given, --global_idx_start/--global_idx_end are required."
    )
    parser.add_argument("--global_idx_start", type=int, default=None)
    parser.add_argument("--global_idx_end", type=int, default=None)
    parser.add_argument(
        "--global_n_samples", type=int, default=None,
        help="Alternative to --global_idx_end: a sample count, so global_idx_end "
        "is computed as global_idx_start + global_n_samples. Convenient for "
        "badger configs, since badger's format_rule can't do arithmetic -- "
        "this lets --global_n_samples be a plain constant."
    )
    parser.add_argument("--nu_source_freq", type=float)  # non-angular
    parser.add_argument(
        "--solver_type", choices=["ls", "hps"], default="ls",
        help="Solver type: Lippmann-Schwinger ('ls') or Hierarchical Poincare-Steklov ('hps'). "
        "Defaults to the Lippmann-Schwinger solver."
    )
    # Lippmann-Schwinger solver arguments
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    # parser.add_argument("--linsys_solver", choices=["bicgstab", "gmres"], default="gmres")
    # 2026-08-05: DROPPED GMRES SUPPORT; argument remains for command-line interface backward-compatibility
    parser.add_argument("--linsys_solver", choices=["bicgstab"], default="bicgstab")
    parser.add_argument("--use_half_grid", choices=["true", "false"], default="false")
    parser.add_argument("--half_grid_tol_ratio", type=float, default=0.5)
    parser.add_argument("--convergence_by_dir", choices=["true", "false"], default="true")
    parser.add_argument("--max_iter", type=int, default=500)
    # 2026-08-05: DROPPED GMRES SUPPORT; argument remains for command-line interface backward-compatibility
    parser.add_argument("--restart", type=int, default=10, help="Deprecated argument; use bicgstab_restart instead")
    parser.add_argument("--bicgstab_restart", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)

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
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--no_filtering", default=False, action="store_true")
    parser.add_argument("--wandb_entity", type=wandb_entity_arg_type, default=None, help="The W&B entity")
    parser.add_argument("--wandb_project")
    parser.add_argument(
        "--wandb_mode", choices=["offline", "online"], default="offline"
    )
    parser.add_argument("--dont_use_wandb", default=False, action="store_true")

    a = parser.parse_args()
    a.use_half_grid = (a.use_half_grid == "true")
    a.convergence_by_dir = (a.convergence_by_dir == "true")
    return a


def create_new_meas_data_file(scobj_settings: dict, mdf_fp: str, args: dict) -> None:
    """Creates a blank file with the desired settings and sets up empty fields
    (in case the target output does not yet exist)
    Parameters:
        scobj_settings (dict): already-loaded scattering object data, either
            a whole file's contents (load_hdf5_to_dict) or a global-index
            slice across one or more shard files (load_single_dir_slice)
        mdf_fp (str): measurement datafile filepath
        args (dict): the rest of the arguments passed to this script

    Output:
        None (simply creates the new file at mdf_fp and will raise an error if it exists already)
    """
    ### Set up new fields ###
    # Grid points
    rho_vals = scobj_settings["rho_vals"]
    theta_vals = scobj_settings["theta_vals"]
    # x_vals = scobj_settings["x_vals"]

    num_rho = rho_vals.shape[0]
    num_theta = theta_vals.shape[0]
    num_r = num_theta
    num_s = num_theta
    num_m = num_theta
    num_h = num_rho

    m_vals = theta_vals
    h_vals = np.linspace(-np.pi / 2, np.pi / 2, num_rho, endpoint=False)

    # Scattering object
    q_cart = scobj_settings["q_cart"]
    q_polar = scobj_settings["q_polar"]
    num_samples = q_cart.shape[0]
    q_cart_lpf = np.full(q_cart.shape, np.nan, dtype=np.float32)
    q_polar_lpf = np.full(q_polar.shape, np.nan, dtype=np.float32)

    # Measured wavefields
    nu_sf = np.array([args.nu_source_freq])
    omega_sf = 2 * np.pi * nu_sf
    d_rs = np.full((num_samples, num_r, num_s), np.nan, dtype=np.complex64)
    d_mh = np.full((num_samples, num_m, num_h), np.nan, dtype=np.complex64)

    # Convenience settings
    sample_completion = np.zeros(num_samples, dtype=bool)
    file_completion = np.array([False])

    new_settings = {
        # Grid points
        "m_vals": m_vals,
        "h_vals": h_vals,
        # Scattering objects
        "q_cart_lpf": q_cart_lpf,
        "q_polar_lpf": q_polar_lpf,
        # Measured wavefields
        "nu_sf": nu_sf,
        "omega_sf": omega_sf,
        "d_rs": d_rs,
        "d_mh": d_mh,
        # Convenience settings
        "sample_completion": sample_completion,
        "file_completion": file_completion,
    }

    # Combine and overwrite settings as necessary
    mdf_settings = {**scobj_settings, **new_settings}
    save_dict_to_hdf5(mdf_settings, mdf_fp)
    return


def main(args: argparse.Namespace) -> None:
    ###########################################################################
    # Setup
    logging.info(f"Received arguments {vars(args)}")

    # 2026-08-05: gmres was dropped, so linsys_solver is always "bicgstab" now;
    # --restart is kept as a CLI argument for backward compatibility but is unused.
    restart_interval = args.bicgstab_restart

    # print(f"start index: {args.start_idx}")

    d = os.path.split(args.output_fp)[0]
    if not os.path.isdir(d):
        try:
            os.mkdir(d)
            logging.info(f"created directory: {d}")
        except:
            logging.warning(f"problem creating directory {d}; proceeding anyway...")
    else:
        logging.info(f"not creating dir {d} because it already exists")

    ### 1. Check whether the target measurement file is complete already
    # if so, we can avoid reading everything from disk
    try:
        mdf_already_complete = load_field_in_hdf5(
            "file_completion", args.output_fp
        ).item()
    except:
        # Mark incomplete if the mdf doesn't exist
        # or has an issue with its "file_completion" field
        mdf_already_complete = False
    # logging.warning(f"Measurement data file complete? {mdf_already_complete}")
    # Check for file completion here
    if mdf_already_complete == True:
        logging.warning(
            f"Measurement file marked complete; exiting early (file name: {args.output_fp})"
        )
        return

    # May raise a FileNotFoundError if the scattering object data is not found/incomplete
    scobj_settings = None  # populated here in --input_dir mode; loaded lazily otherwise
    if args.input_dir is not None:
        # New mode: read a global sample-index slice across one or more
        # scattering-object shard files, decoupled from the shard size.
        if args.global_idx_start is None:
            raise ValueError("--global_idx_start is required when --input_dir is given")
        if args.global_idx_end is not None:
            global_idx_end = args.global_idx_end
        elif args.global_n_samples is not None:
            global_idx_end = args.global_idx_start + args.global_n_samples
        else:
            raise ValueError(
                "One of --global_idx_end or --global_n_samples is required when --input_dir is given"
            )
        invalid_sdf = False
        try:
            scobj_settings = load_single_dir_slice(
                args.input_dir, args.global_idx_start, global_idx_end,
            )
            invalid_sdf = not bool(np.all(scobj_settings["sample_completion"]))
        except Exception as e:
            invalid_sdf = True
        if invalid_sdf:
            logging.error(
                f"The input scattering objects at {args.input_dir} "
                f"[{args.global_idx_start}:{global_idx_end}) appear to be invalid or incomplete"
            )
            raise FileNotFoundError(
                f"The input scattering objects at {args.input_dir} "
                f"[{args.global_idx_start}:{global_idx_end}) appear to be invalid or incomplete"
            )
        num_samples_all = global_idx_end - args.global_idx_start
    else:
        # Original mode: a single whole scattering-object file
        invalid_sdf = False
        if not os.path.exists(args.input_fp):
            invalid_sdf = True
            logging.error(f"Input scattering object file {args.input_fp} could not be found")
        else:
            # Check the "file_completion" flag
            sdf_already_complete = load_field_in_hdf5(
                "file_completion", args.input_fp
            ).item()
            invalid_sdf = not sdf_already_complete
        if invalid_sdf:
            logging.error(
                f"The input scattering object file at"
                f" {args.input_fp} appears to be invalid"
            )
            raise FileNotFoundError(
                f"The input scattering object file at"
                f" {args.input_fp} appears to be invalid"
            )
        else:
            num_samples_all = load_field_in_hdf5("sample_completion", args.input_fp).shape[0]

    ### 2. Attempt to load the output measurement file if it exists
    # If no measurement file exists, create a new one
    try:
        logging.warning(f"Attempting to load settings from target file")
        # print(f"meas file exists: {os.path.exists(args.output_fp)}")
        if not os.path.exists(args.output_fp):
            raise FileNotFoundError
        meas_settings = load_hdf5_to_dict(args.output_fp)
        logging.debug("meas_settings: %s", meas_settings.keys())
    except Exception as e:
        # In case of an error while loading
        if os.path.exists(args.output_fp):
            logging.warning(
                f"Deleting measurement output file {args.output_fp} after"
                " encountering an error {e} while attempting to load output file"
            )
            os.remove(args.output_fp)
        logging.warning(f"Creating new measurement file from scratch")
        if scobj_settings is None:
            scobj_settings = load_hdf5_to_dict(args.input_fp)
        create_new_meas_data_file(scobj_settings, args.output_fp, args)
        meas_settings = load_hdf5_to_dict(args.output_fp)

    # Unload the settings into local variables
    # Grid variables
    # omega_wf = args.omega_val # wave field measurement omega
    nu_sf = args.nu_source_freq  # non-angular frequency of the source wave
    omega_sf = 2 * np.pi * nu_sf  # angular frequency of the source wave
    x_vals = meas_settings["x_vals"]
    rho_vals = meas_settings["rho_vals"]
    theta_vals = meas_settings["theta_vals"]
    m_vals = meas_settings["m_vals"]
    h_vals = meas_settings["h_vals"]
    num_rho = rho_vals.shape[0]
    num_theta = theta_vals.shape[0]
    num_x = x_vals.shape[0]
    num_pixels = x_vals.shape[0]
    num_r = num_theta
    num_s = num_theta
    num_m = num_theta
    num_h = num_rho

    # Convenience objects
    sample_completion = meas_settings["sample_completion"]

    ### 3. set up the PDE solver and convolution operators
    logging.warning("Setting up solvers and convolution objects")
    spatial_domain_max = np.max(np.abs(x_vals))
    num_theta = theta_vals.shape[0]
    num_h = h_vals.shape[0]

    # Use Lippmann-Schwinger solver if the jaxhps code could not be imported
    use_ls = (args.solver_type == "ls") or not JAXHPS_IMPORTED
    if use_ls:
        if args.solver_type == "hps":
            logging.warning("Failed to import jaxhps...")
        logging.warning(f"Using Lippmann-Schwinger solver")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ls_solver = setup_bicgstab_solver(
            num_pixels, spatial_domain_max, nu_sf, args.receiver_radius,
            device=device,
            prepare_half_grid=args.use_half_grid,
            # Set the default settings
            max_iter=args.max_iter,
        )
    else:
        logging.warning(f"Using HPS solver")
        device = jax.devices("gpu")[0]
        hps_cdf = args.hps_comp_domain_factor
        unif_domain_hlen = np.max(np.abs(x_vals))
        hpst_domain_hlen = hps_cdf * unif_domain_hlen

        # 2026-08-05: get_SD_matrices_fp now accepts numeric nu_sf (not just strings)
        sd_mat_fp = get_SD_matrices_fp(
            nu_sf,
            args.hps_l,
            args.hps_p,
            domain_half_length=hpst_domain_hlen,
            SD_matrices_dir=args.hps_sd_mat_dir
        )
        S_int, D_int = load_SD_matrices(sd_mat_fp)
        unif_bounds = spatial_domain_max * np.array([-1., 1., -1., 1.])
        hpst_bounds = hps_cdf * unif_bounds
        hps_solver = HPSScatteringSolver(
            L=args.hps_l,
            p=args.hps_p,
            N_x=num_pixels,
            k=omega_sf,
            S_int=S_int,
            D_int=D_int,
            N_r=num_pixels,
            N_s=num_pixels,
            unif_domain_bounds=unif_bounds,
            quad_domain_bounds=hpst_bounds,
        )
    logging.warning(f"Finished setting up the solver object; time for interp operators...")


    # Measurement change-of-coordinates interp objects
    conv_rs_to_m, conv_rs_to_h = prep_rs_to_mh_interp(
        theta_vals,  # r grid points
        theta_vals,  # s grid points
        num_theta,
        num_h,
        a_neg_half=True,
    )
    # Scattering object polar-to-euclidean interp objects
    polar_grid = polar_to_euclidean(theta_vals, rho_vals)  # (n_theta*n_rho, 2)
    conv_cart_to_polar_x, conv_cart_to_polar_y = prep_conv_interp_2d(
        x_vals,
        x_vals,  # Use x points for y dim here
        polar_grid,
        bc_modes=("extend", "extend"),
        a_neg_half=True,  # set a=-1/2 or -3/4 as a parameter for the conv filter
    )

    # LPF object to make q_cart_lpf and q_polar_lpf
    dx = x_vals[1] - x_vals[0]
    nu_lpf = 2 * nu_sf
    lpf_x, _, _ = prep_lpf_from_wavenum(nu_lpf, num_x, pad_mode="power-of-two")
    lpf_y = np.copy(lpf_x)  # just reuse since x_vals=y_vals

    logging.warning(f"Finished setting up solver objects and conv operators")

    ### 4. Prepare the index range
    create_n_samples = (
        args.create_n_samples if args.create_n_samples != -1 else num_samples_all
    )
    args.end_idx = min(args.start_idx + create_n_samples, num_samples_all)
    full_slice = slice(args.start_idx, args.end_idx)
    num_samples_eff = args.end_idx - args.start_idx

    # Buffer variables
    # read in q/d from the measurement file
    q_cart_eff = meas_settings["q_cart"][full_slice]
    # q_polar_eff = meas_settings["q_polar"][full_slice]
    q_cart_lpf_eff = meas_settings["q_cart_lpf"][full_slice]
    q_polar_lpf_eff = meas_settings["q_polar_lpf"][full_slice]
    d_rs_eff = meas_settings["d_rs"][full_slice]
    d_mh_eff = meas_settings["d_mh"][full_slice]

    ### 5. Filter and scatter the inputs
    logging.warning(
        f"Beginning to process (filter+scatter) {num_samples_eff} scattering objects"
    )

    # chunk_counter = 0 # absolute index from the beginning
    for chunk_start_idx in range(args.start_idx, args.end_idx, args.write_every_n):
        chunk_end_idx = min(chunk_start_idx + args.write_every_n, args.end_idx)
        made_changes_in_chunk = False # track whether we can skip the save step

        # Loop over the indices in the chunk
        for idx_abs in range(chunk_start_idx, chunk_end_idx):
            # idx_abs = i # rename for clarity...
            idx_eff = idx_abs - args.start_idx
            logging.warning("Working on sample %i of %i", idx_eff + 1, num_samples_eff)
            computed_soln_bool = None  # Leave blank for now...

            # It's possible the wave fields for this sample has already been computed, so we
            # want to skip computing it if possible
            is_any_scobj_nans = (
                np.any(np.isnan(q_cart_lpf_eff[idx_eff]))
                or np.any(np.isnan(q_polar_lpf_eff[idx_eff]))
            )

            # (Re-)do the filtering if needed
            if is_any_scobj_nans:
                # Redo the filtering
                q_cart_lpf_i = apply_filter_fourier_2d(
                    q_cart_eff[idx_eff],
                    lpf_x,
                    lpf_y,
                )
                q_cart_lpf_eff[idx_eff] = q_cart_lpf_i
                q_polar_lpf_i = apply_interp_2d(
                    conv_cart_to_polar_x,
                    conv_cart_to_polar_y,
                    q_cart_lpf_i,
                ).reshape(num_theta, num_rho)
                q_polar_lpf_eff[idx_eff] = q_polar_lpf_i

            # (Re-)do the PDE solve if necessary
            # First determine whether that is necessary
            is_any_data_nans = np.any(np.isnan(d_rs_eff[idx_eff])) or np.any(
                np.isnan(d_mh_eff[idx_eff])
            )
            is_all_data_nans = np.all(np.isnan(d_rs_eff[idx_eff])) and np.all(
                np.isnan(d_mh_eff[idx_eff])
            )
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
                scattering_obj_i = q_cart_eff[idx_eff]
                try:
                    if use_ls:
                        u_scat_ext = ls_solver.Helmholtz_solve_exterior_batched(
                            scattering_obj_i,
                            rtol=args.rtol,
                            batch_size=args.batch_size,
                            linsys_solver=args.linsys_solver,
                            use_half_grid=args.use_half_grid,
                            half_grid_tol_ratio=args.half_grid_tol_ratio,
                            convergence_by_dir=args.convergence_by_dir,
                            max_iter=args.max_iter,
                            restart=restart_interval,
                            report_status=False,
                        )
                        d_rs_eff[idx_eff, :] = u_scat_ext
                    else:
                        # Use HPS if requested
                        _, u_scat_ext = hps_solver.solve_exterior(
                            scattering_obj_i
                        )
                        d_rs_eff[idx_eff, :] = u_scat_ext
                    if np.any(np.isnan(u_scat_ext)):
                        raise RuntimeError

                    computed_soln_bool = True

                    # This transforms the wave field from (r, s) coordinates to (m, h) coords
                    # as specified by the FY19 paper
                    mh_soln_pre = apply_interp_2d(
                        conv_rs_to_m, conv_rs_to_h, d_rs_eff[idx_eff]
                    ).reshape(num_m, num_h)

                    # Correct for geometric spreading as suggested by FY19
                    d_mh_eff[idx_eff] = mh_soln_pre * get_scale_factor(
                        CONST_RHO_PRIME, CONST_THETA_PRIME
                    )
                except _LinAlgError:
                    d_mh_eff[idx_eff] = np.full_like(d_mh_eff[idx_eff], np.nan)
                    logging.warning("Singular matrix for sample %i", idx_abs)
                    computed_soln_bool = False
                    continue
                except RuntimeError:
                    d_mh_eff[idx_eff] = np.full_like(d_mh_eff[idx_eff], np.nan)
                    logging.warning("NaN encountered sample %i", idx_abs)
                    computed_soln_bool = False
                    continue

            # Log updates from this sample sample
            sample_completion[idx_abs] = True  # mark as complete :D
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

        d_rs_nans = np.any(np.isnan(d_rs_eff[chunk_slice_eff]), axis=(1,2))
        d_mh_nans = np.any(np.isnan(d_mh_eff[chunk_slice_eff]), axis=(1,2))
        logging.info(f"d_rs_nans: {d_rs_nans}")
        logging.info(f"d_mh_nans: {d_mh_nans}")

        if made_changes_in_chunk:
            update_field_in_hdf5(
                "q_cart_lpf",
                q_cart_lpf_eff[chunk_slice_eff],
                args.output_fp,
                chunk_slice_abs,
            )
            update_field_in_hdf5(
                "q_polar_lpf",
                q_polar_lpf_eff[chunk_slice_eff],
                args.output_fp,
                chunk_slice_abs,
            )
            update_field_in_hdf5(
                "d_rs", d_rs_eff[chunk_slice_eff], args.output_fp, chunk_slice_abs
            )
            update_field_in_hdf5(
                "d_mh", d_mh_eff[chunk_slice_eff], args.output_fp, chunk_slice_abs
            )
            # Make sure this is the last thing to get written to disk
            update_field_in_hdf5(
                "sample_completion",
                sample_completion[chunk_slice_abs],
                args.output_fp,
                chunk_slice_abs,
            )

        try:
            # Log RAM+VRAM usage
            process = psutil.Process()
            logging.warning(f"Memory usage: {process.memory_info().rss >> 20} MB")
            # this is not where the memory usage peaks
            vram_free_bytes, vram_available_bytes = torch.cuda.mem_get_info()
            vram_used_mb = (vram_available_bytes - vram_free_bytes) >> 20
            logging.warning(
                f"Current VRAM usage: {vram_used_mb} MB / {vram_available_bytes>>20} MB"
            )
        except:
            logging.warning(
                f"Skipping memory or VRAM usage calculation due to an error"
            )

    # Fetch from disk again in case someone else was working on the same file at the same time
    sample_completion_newest = load_field_in_hdf5("sample_completion", args.output_fp)
    if np.all(sample_completion_newest):
        # If every sample has been completed then we can mark the file as complete
        mdf_completion = np.array([True])
        update_field_in_hdf5(
            "file_completion",
            mdf_completion,
            args.output_fp,
        )
        logging.warning(
            f"Marked the measurement file as complete! ( {args.output_fp} )"
        )
    logging.warning("Finished")
    return

if __name__ == "__main__":
    a = setup_args()

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

    logging.info(f"Start: generate_measurement_file.py")

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
                logging.error(f"generate_measurements_files.py terminating early")
