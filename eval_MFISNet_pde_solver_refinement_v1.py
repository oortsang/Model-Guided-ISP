# Load and evaluate the MFISNet_pde_solver_refinement_v1 on the requested dataset
# Also saves the predictions to disk

import logging
import argparse
import yaml
import os

import numpy as np
import torch

from src.data.data_io import (
    save_dict_to_hdf5,
)

from src.models.MFISNet_pde_solver_refinement_v1 import (
    MFISNet_pde_solver_refinement_v1,
    load_MFISNet_pde_solver_refinement_v1_from_state_dict,
    TupleLinearData,
)
from train_MFISNet_pde_solver_refinement_v1 import setup_pde_solver_refinement_dataset

from src.data.data_transformations import (
    prep_conv_interp_2d,
    prep_polar_padder,
    polar_pad_and_apply,
)
from src.data.data_io import (
    load_hdf5_to_dict,
    load_field_in_hdf5,
    load_multifreq_dataset,
    load_scobj_dir,
)
from src.data.layout import _file_start_idx as _get_number_from_filename
from src.utils.logging_utils import FMT, TIMEFMT, find_best_epoch
from src.data.data_naming_constants import (
    KEYS_FOR_EXPERIMENT_INFO_OUT,
    Q_POLAR,
    Q_CART,
    D_MH,
    D_RS,
    Q_POLAR_LPF,
    Q_CART_LPF,
    NU_SF,
    OMEGA_SF,
    X_VALS,
    RHO_VALS,
    THETA_VALS,
    H_VALS
)


from src.training_utils.make_predictions import make_preds_on_dataset
from src.training_utils.loss_functions import (
    psnr,
    relative_l2_error,
    _mse_along_batch,
)



SCOBJ_DIR_TEST = "test_scattering_objs"

import psutil
def get_ram_usage():
    process = psutil.Process()
    logging.info(
        f"Memory usage: {process.memory_info().rss>>20} MB"
    )
    if torch.cuda.is_available():
        vram_free_bytes, vram_available_bytes = torch.cuda.mem_get_info()
        vram_used_mb = (vram_available_bytes - vram_free_bytes) >> 20
        logging.info(
            f"Current VRAM usage: {vram_used_mb} MB / {vram_available_bytes>>20} MB"
        )

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_data_dir_base",
        type=str,
        help="For the reference dataset, indicate the directory containing all the "
        "measurement folders corresponding to the relevant frequencies and data subsets",
    )
    parser.add_argument(
        "--pred_data_dir_base",
        type=str,
        help="For the prediction dataset, indicate the directory containing all the "
        "measurement folders corresponding to the relevant frequencies and data subsets",
    )
    parser.add_argument("--data_input_nus", type=str, nargs="+")

    # New option to use smoothed targets or not
    parser.add_argument(
        "--use_targets", choices=["original", "smoothed", "legacy"], default="legacy",
        help=(
            "Set target as original or smoothed; alternatively, set to legacy to use "
            "either --used_smoothed_targets or --use_original_targets."
        )
    )
    parser.add_argument("--use_smoothed_targets", default=False, action="store_true")
    parser.add_argument("--use_original_targets", action="store_false", dest="use_smoothed_targets")

    # Old flags
    parser.add_argument("--eval_on_test_set", default=True, action="store_true")
    parser.add_argument(
        "--no_eval_on_test_set", action="store_false", dest="eval_on_test_set"
    )
    # Updated flag; use the old flag if set to old_flag
    parser.add_argument(
        "--eval_on_set",
        choices=["train", "val", "test", "old_flag"],
        default="old_flag",
        help="Updated flag: uses the old flag "
        "(--eval_on_test_set or --no_eval_on_test_set) "
        "if set to old_flag; otherwise, choose one of the train/val/test sets directly",
    )
    parser.add_argument("--truncate_num", type=int, default=None)

    parser.add_argument(
        "--manual_model_fp",
        type=str,
        help="Manually select the (full) model file path; if set, this will override the other selection rules.",
        default=None,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Point to the directory containing the desired model parameters",
    )

    # Would require re-writing too much in the logging utils right now...
    # epoch_choice="latest"
    #     --use_epoch ${epoch_choice} \
    parser.add_argument(
        "--use_epoch",
        choices=["best", "latest"],
        default="best",
        help="Choose the best or latest epoch",
    )

    parser.add_argument(
        "--training_results_fp",
        type=str,
        help="tab-separated file containing training results.",
    )
    parser.add_argument(
        "--training_results_key",
        type=str,
        help="key used to select the epoch with the minimal validation loss.",
    )
    parser.add_argument("--model_fp_format", type=str, default="epoch_{}.pickle")
    # parser.add_argument(
    #     "--hyperparam_summary_fp",
    #     type=str,
    #     help="Point to the hyperparameter search summary file (yaml format)",
    # )
    parser.add_argument(
        "--test_output_summary_fp",
        type=str,
        help="Point to the desired output summary file",
    )
    parser.add_argument(
        "--test_output_predictions_dir",
        type=str,
        help="Point to the desired output predictions file",
    )

    parser.add_argument("--seed", default=None, type=int)  # seed bc we're using noise
    parser.add_argument(
        "--noise_to_signal_ratio", default=None, type=float
    )  # test with noise
    parser.add_argument("--use_noise_seed", choices=["true", "false"], default="false")
    parser.add_argument("--noise_seed",  type=int, default=None)
    parser.add_argument("--noise_norm_mode", choices=["l2", "inf"], default="inf")


    parser.add_argument(
        "--samples_per_chunk",
        type=int,
        default=500,
        help="This is the 'shard_size' for make_predictions_on_dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="For evaluation this just needs to be small enough to fit into GPU memory",
    )
    parser.add_argument(
        "--use_pred_d_mh", default="true", choices=["true", "false"],
        help="Whether to load predicted d_mh values from the predictions dataset"
    )
    parser.add_argument(
        "--timing_run", default="false", choices=["true", "false"],
        help="Evaluates a second time without saving outputs for a better timing estimate"
    )

    parser.add_argument("--debug", default=False, action="store_true")
    a = parser.parse_args()

    if a.eval_on_set == "compat_flag":
        a.eval_on_set = "test" if a.eval_on_test_set else "val"

    # Parsing boolean argument values
    a.use_pred_d_mh = (a.use_pred_d_mh == "true")
    a.use_noise_seed = (a.use_noise_seed == "true")
    a.timing_run = (a.timing_run == "true")

    # Override unless use_targets="legacy"
    if a.use_targets == "smoothed":
        a.use_smoothed_targets = True
    elif a.use_targets == "original":
        a.use_smoothed_targets = False
    return a

n
def main(args: argparse.Namespace) -> None:
    """
    1. Select the model from the best epoch of the given run
    2. Load data
    3. Set up NN with hyperparameters
    4. Run the NN on the requested dataset and save the predictions to disk
    5. Compute the error/performance statistics and save the results to disk
    """
    if not os.path.isdir(args.test_output_predictions_dir):
        logging.info(f"Attempting to mkdir {args.test_output_predictions_dir}")
        os.mkdir(args.test_output_predictions_dir)
    # Set seeds for reproducible noise
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Set up noise seeds
    if args.use_noise_seed:
        eff_noise_seed = args.noise_seed
    else:
        eff_noise_seed = None

    if args.noise_to_signal_ratio != 0:
        logging.info(f"Using seed as {eff_noise_seed} for the dataset")
    else:
        logging.info(f"Not adding noise!")


    # Find the best epoch
    if args.use_epoch == "best":
        best_epoch_dd = find_best_epoch(
            args.training_results_fp, args.training_results_key, selection_mode="min"
        )
    else:
        best_epoch_dd = find_best_epoch(
            args.training_results_fp, "epoch", selection_mode="max"
        )
    best_epoch = best_epoch_dd["epoch"]
    hps_polar_padding = best_epoch_dd["polar_padding"]
    logging.info(f"Best epoch: {best_epoch}")

    if args.manual_model_fp is None:
        logging.info(f"Using the automatically-selected model")
        model_fp = os.path.join(args.model_dir, args.model_fp_format.format(best_epoch))
    else:
        logging.info(f"Using the manually-selected model")
        model_fp = args.manual_model_fp
    logging.info(f"Selected model filepath: {model_fp}")

    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Evaluating on device: %s", device)

    #########################################################
    # Load data
    ref_data_dir_base = args.ref_data_dir_base
    pred_data_dir_base = args.pred_data_dir_base
    str_nu_list = (
        args.data_input_nus
    )  # nu in string form (to preserve decimals properly)
    nu_list = [float(str_nu) for str_nu in str_nu_list]
    N_freqs = len(nu_list)
    logging.info(f"ref_data_dir_base:  {ref_data_dir_base}")
    logging.info(f"pred_data_dir_base: {pred_data_dir_base}")
    logging.info(f"nu values received: {str_nu_list}")

    eval_set_name = args.eval_on_set
    ref_eval_files = [
        os.path.join(ref_data_dir_base, f"{eval_set_name}_measurements_nu_{nu}")
        for nu in str_nu_list
    ]

    pred_dir_basename = lambda nu: f"measurements_nu_{nu}" if args.use_pred_d_mh else f"scattering_objs"
    pred_eval_files = [
        os.path.join(pred_data_dir_base, f"{eval_set_name}_{pred_dir_basename(nu)}") for nu in str_nu_list
    ]

    logging.info(
        f"Attempting to load the {eval_set_name} sets: {ref_eval_files} "
        f"and {pred_eval_files}"
    )

    ### Load Evaluation dataset to a dictionary and local variables ###
    logging.info(f"Loading evaluation dataset")
    ref_eval_dd, _ = load_multifreq_dataset(
        ref_eval_files,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_mh",
        nan_mode="skip",
        load_cart=True,
        truncate_num=args.truncate_num,
        noise_seed=eff_noise_seed,
    )
    pred_eval_dd, pred_eval_metadata_dd = load_multifreq_dataset(
        pred_eval_files,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_mh",
        nan_mode="skip",
        load_cart=True,
        truncate_num=args.truncate_num,
        scobj_only_mode=not args.use_pred_d_mh,
        # Don't add noise to predictions
    )
    if args.use_smoothed_targets:
        ref_eval_q_polar = ref_eval_dd[Q_POLAR_LPF][:, -1, ...]
        ref_eval_q_cart  = ref_eval_dd[Q_CART_LPF][:, -1, ...]
        logging.info(f"Using smoothed targets")
    else:
        ref_eval_q_polar = ref_eval_dd[Q_POLAR]
        ref_eval_q_cart = ref_eval_dd[Q_CART]
        logging.info(f"Using original targets")
    pred_eval_q_polar  = pred_eval_dd[Q_POLAR].astype(np.float32) # cast just in case

    ref_eval_d_mh = ref_eval_dd[D_MH]
    if args.use_pred_d_mh:
        pred_eval_d_mh = pred_eval_dd[D_MH]
    else:
        pred_eval_d_mh = None
    logging.info(f"loaded q_polar(_lpf) has shape: {ref_eval_q_polar.shape}")
    logging.info(f"loaded q_cart(_lpf) has shape: {ref_eval_q_cart.shape}")
    logging.info(f"loaded d_mh has shape: {ref_eval_d_mh.shape}")

    rho_vals = ref_eval_dd[RHO_VALS]
    theta_vals = ref_eval_dd[THETA_VALS]
    h_vals = ref_eval_dd[H_VALS]
    omega_vals = ref_eval_dd[OMEGA_SF]
    x_vals = (
        ref_eval_dd[X_VALS]
        if X_VALS in ref_eval_dd.keys()
        else np.linspace(-0.5, 0.5, ref_eval_q_cart.shape[-1])
    )  # default value..
    N_x = x_vals.shape[0]
    N_rho = rho_vals.shape[0]
    N_h = h_vals.shape[0]
    N_theta = theta_vals.shape[0]
    N_eval = ref_eval_q_polar.shape[0]

    # Prepare the LinearData object
    eval_valid_idcs = np.logical_and(
        pred_eval_dd["orig_idcs"],
        ref_eval_dd["orig_idcs"],
    )
    eval_dset = setup_pde_solver_refinement_dataset(
        pred_eval_q_polar,
        pred_eval_d_mh,
        ref_eval_d_mh[eval_valid_idcs],
        ref_eval_q_polar[eval_valid_idcs],
        use_pred_d_mh=args.use_pred_d_mh,
    )

    # Prepare the DataLoader
    eval_dloader = torch.utils.data.DataLoader(
        eval_dset, batch_size=args.batch_size, shuffle=False
    )

    ##### Load model from disk #####
    model_state_dict = torch.load(model_fp, map_location=device)
    model = load_MFISNet_pde_solver_refinement_v1_from_state_dict(
        model_state_dict,
        N_freqs,
        # metadata_dd=pred_eval_metadata_dd,
        epoch_results_dd=best_epoch_dd,
        # polar_padding=hps_polar_padding,
        N_h=N_h,
        use_pred_d_mh=args.use_pred_d_mh,
    )
    model = model.to(device)
    model.eval()

    logging.info(f"Loaded model: {model}")

    #########################################################
    # Make the predictions and save them to disk.

    experiment_info = {}
    for key, value in ref_eval_dd.items():
        if key in KEYS_FOR_EXPERIMENT_INFO_OUT:
            experiment_info[key] = value

    make_preds_on_dataset(
        model=model,
        dloader=eval_dloader,
        experiment_info=experiment_info,
        output_dir=args.test_output_predictions_dir,
        device=device,
        shard_size=args.samples_per_chunk,
        use_orig_idcs=True,
    )
    if args.timing_run:
        t0 = time.perf_counter()
        make_preds_on_dataset(
            model=model,
            dloader=eval_dloader,
            experiment_info=experiment_info,
            output_dir=None,
            device=device,
            shard_size=args.samples_per_chunk,
            use_orig_idcs=True,
        )
        t1 = time.perf_counter()
        logging.info(f"Timing run took {t1-t0:.3f}s for {arg.truncate_num} samples")
    logging.info(f"Finished generating predictions...")

    #########################################################
    # Load the predictions to evaluate
    preds_out_dir = args.test_output_predictions_dir
    preds_out_dd, _ = load_scobj_dir(
        preds_out_dir,
        args.truncate_num,
        load_cart = True,
        nan_mode = "keep",
    )
    logging.info(f"Re-loaded predictions have {preds_out_dd['num_nan_samples']} nan samples (ignored for evaluation purposes)")
    preds_out_q_polar_unskipped = preds_out_dd[Q_POLAR]
    preds_out_q_cart_unskipped  = preds_out_dd[Q_CART]
    num_tot = ref_eval_dd["num_loaded_samples"]

    ref_eval_valid_idcs  = ref_eval_dd["orig_idcs"]
    preds_out_valid_idcs = preds_out_dd["orig_idcs"]
    both_valid_idcs = np.logical_and(
        preds_out_valid_idcs,
        ref_eval_valid_idcs,
    )

    # Blow up ref_eval_q_{polar,cart} so we can share the valid idcs
    ref_eval_q_polar_unskipped = np.full(
        [num_tot, *ref_eval_q_polar.shape[1:]],
        fill_value=np.nan
    )
    ref_eval_q_polar_unskipped[eval_valid_idcs] = ref_eval_q_polar
    ref_eval_q_cart_unskipped  = np.full(
        [num_tot, *ref_eval_q_cart.shape[1:]],
        fill_value=np.nan
    )
    ref_eval_q_cart_unskipped[eval_valid_idcs] = ref_eval_q_cart

    ### Polar section sanity check ###
    preds_polar_valid   = torch.from_numpy(preds_out_q_polar_unskipped[both_valid_idcs])
    targets_polar_valid = torch.from_numpy(ref_eval_q_polar_unskipped[both_valid_idcs])

    assert not torch.any(torch.isnan(preds_polar_valid))
    assert not torch.any(torch.isnan(targets_polar_valid))

    rel_l2_polar_errors = relative_l2_error(
        preds=preds_polar_valid,
        targets=targets_polar_valid,
    ).numpy()
    logging.info(f"Relative l2 error (polar): {np.mean(rel_l2_polar_errors)}")

    #########################################################
    # Evaluate the predictions
    test_preds_arr   = torch.from_numpy(preds_out_q_cart_unskipped[both_valid_idcs])
    test_targets_arr = torch.from_numpy(ref_eval_q_cart_unskipped[both_valid_idcs])

    assert not torch.any(torch.isnan(test_preds_arr))
    assert not torch.any(torch.isnan(test_targets_arr))
    # test_preds_arr = torch.from_numpy(test_preds_arr)
    # test_targets_arr = torch.from_numpy(test_targets_arr)

    #########################################################
    # Remove nans if necessary
    # is_nan_preds = torch.isnan(test_preds_arr[:, 0, 0])
    # is_not_nan = torch.logical_not(is_nan_preds)
    # logging.info(
    #     "Removing %i samples due to truncations or NaNs", n_samples - torch.sum(is_not_nan)
    # )

    # test_preds_arr = test_preds_arr[is_not_nan]
    # test_targets_arr = test_targets_arr[is_not_nan]

    rel_l2_errors = relative_l2_error(
        preds=test_preds_arr,
        targets=test_targets_arr,
    ).numpy()


    mse_errors = _mse_along_batch(
        preds=test_preds_arr, targets=test_targets_arr
    ).numpy()
    psnrs = psnr(preds=test_preds_arr, targets=test_targets_arr).numpy()

    cart_rel_l2_mean = np.mean(rel_l2_errors)
    cart_rel_l2_std = np.std(rel_l2_errors)
    cart_mse_mean = np.mean(mse_errors)
    cart_mse_std = np.std(mse_errors)
    cart_psnr_mean = np.mean(psnrs)
    cart_psnr_std = np.std(psnrs)


    # logging.info(f"Main loop successful")
    logging.info(f"~~~Summary~~~")
    logging.info(f"MSE error: {cart_mse_mean:.3e}±{cart_mse_std:.3e}")
    logging.info(f"Rel l2 error: {cart_rel_l2_mean:.5f}±{cart_rel_l2_std:.5f}")
    logging.info(f"PSNR: {cart_psnr_mean:.5f}±{cart_psnr_std:.5f}")

    common_settings_dict = {
        # Grid info
        "N_rho": N_rho,
        "N_theta": N_theta,
        "N_m": N_theta,
        "N_h": N_h,
        "N_x": N_x,
        "N_freqs": N_freqs,
        # Hyperparam info
        "N_cnn_1d": model.N_cnn_1d,
        "N_fynet_cnn_2d": model.N_fynet_cnn_2d,
        "N_update_cnn_2d": model.N_update_cnn_2d,
        "N_channels_cnn_1d": model.c_1d,
        "N_channels_cnn_2d": model.c_2d,
        "kernel_size_1d": model.w_1d,
        "kernel_size_2d": model.w_2d,
    }
    summary_errors_dict = {
        "cart_mse_mean": cart_mse_mean,
        "cart_mse_std": cart_mse_std,
        "cart_rel_l2_mean": cart_rel_l2_mean,
        "cart_rel_l2_std": cart_rel_l2_std,
        "cart_psnr_mean": cart_psnr_mean,
        "cart_psnr_std": cart_psnr_std,
    }
    # common_settings_dict = {key: val for (key,val) in common_settings_dict.items()}
    summary_errors_dict = {
        key: val.item() for (key, val) in summary_errors_dict.items()
    }

    summary_dict = {
        # Summary values
        **summary_errors_dict,
        # Metadata
        "model_file_name": model_fp,
        "predictions_fp": args.test_output_predictions_dir,
        **common_settings_dict,
    }

    # Save to disk
    with open(args.test_output_summary_fp, "w") as sfile:
        yaml.dump(summary_dict, sfile, default_flow_style=False)
    logging.info(f"Saved summary file to {args.test_output_summary_fp}")
    logging.info(f"Saved predictions to {args.test_output_predictions_dir}")
    logging.info(f"Finished!")


if __name__ == "__main__":
    a = setup_args()

    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    if a.debug:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)

    logging.info(f"Received the following arguments: {a}")
    main(a)
