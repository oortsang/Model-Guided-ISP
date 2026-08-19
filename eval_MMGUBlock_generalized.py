# Evaluate a single MMGUBlock
# (Measurement misfit gradient update...)
# Expects to receive just one frequency
# This version of the file gives more control
# over which datasets should be evaluated
# e.g., could be just "test", or it could be something
# else entirely, like "ood"

import logging
from typing import List
import argparse
import os, time
import numpy as np
import torch
import wandb
import copy
from src.utils.vram_info import get_memory_info, free_vram

from src.data.add_noise import add_noise_to_d
from src.data.data_io import (
    load_dir,
    load_multifreq_dataset,
    load_scobj_dir,
    load_predictions_dataset,
)
from src.data.datasets import (
    FullData,
    setup_preprocessed_predictions_dataset,
)
from src.models.MMGUBlock import (
    MMGUBlock,
    load_MMGUBlock_from_state_dict,
)

from src.training_utils.train_loop import (
    train,
    evaluate_losses_on_dataloader_only_cartesian,
)
from src.training_utils.make_predictions import (
    FMT_STR as SCOBJ_FMT_STR,
    make_preds_on_dataset_only_cartesian,
)
from src.training_utils.loss_functions import MSEModule
from src.utils.logging_utils import (
    write_result_to_file,
    save_dict_to_yaml,
    load_yaml_to_dict,
    load_field_in_yaml_file,
    FMT,
    TIMEFMT,
    hash_dict
)

from src.data.data_naming_constants import (
    Q_CART,
    Q_CART_LPF,
    GAMMA_CART,
    D_RS,
    NU_SF,
    OMEGA_SF,
    X_VALS,
)

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    bool_choices = ["true", "false"]

    # Expect a single frequency but let the argument accept a list to avoid
    # needing to re-write the data loading code
    parser.add_argument("--data_input_nus", type=str, nargs="+")
    parser.add_argument("--dset_names", type=str, nargs="+")

    # File/directory-related arguments
    parser.add_argument(
        "--ref_data_dir_base",
        type=str,
        help="For the reference dataset, indicate the directory containing all the "
        "measurement folders corresponding to the relevant frequencies and data subsets",
    )
    parser.add_argument(
        "--input_pred_scobj_dir_base",
        type=str,
        help="For the prediction dataset, indicate the directory containing all the "
        "predicted scattering objects",
    )
    parser.add_argument(
        "--input_pred_mmg_dir_base",
        type=str,
        help="For the prediction dataset, indipcate the directory containing all the "
        "measurement-misfit-gradient objects",
    )
    parser.add_argument(
        "--in_central_summary_fp",
        type=str,
        help="Generate a centralized summary yaml file with hyperparameters and errors",
    )
    parser.add_argument(
        "--out_central_summary_fp",
        type=str,
        help="Generate a centralized summary yaml file with hyperparameters and errors",
    )
    # parser.add_argument(
    #     "--central_model_dir",
    #     type=str,
    #     help="Save the best weights to a centralized location",
    # )
    parser.add_argument(
        "--central_model_fp",
        type=str,
        help="Weight name",
        default=None,
    )

    ### Training/validation-related arguments ###
    parser.add_argument(
        "--use_targets", choices=["original", "smoothed"], default="original",
        help=(
            "Set target as original or smoothed"
        )
    )
    parser.add_argument("--truncate_nums", type=int, nargs="+")
    # parser.add_argument("--truncate_num_train", type=int)
    # parser.add_argument("--truncate_num_val", type=int)
    # parser.add_argument("--truncate_num_test", type=int)

    parser.add_argument("--use_noise_seed", choices=bool_choices, default="false")
    parser.add_argument("--noise_seeds", type=int, nargs="+")
    parser.add_argument("--noise_norm_mode", choices=["l2", "inf"], default="inf")
    # parser.add_argument("--noise_seed_train",  type=int, default=10128329)
    # parser.add_argument("--noise_seed_val",    type=int, default=20293834)
    # parser.add_argument("--noise_seed_test",   type=int, default=30943792)

    parser.add_argument("--seed", type=int, default=35675)
    parser.add_argument("--log_batch_size", type=int, default=16)

    parser.add_argument(
        "--noise_to_signal_ratio", default=None, type=float
    )  # train and test with noise

    parser.add_argument("--freq_idx", default=0, type=int, help="The index of the frequency block at use in the naming schemes; defaults to match freq_lvl")

    ### Logging options ###
    parser.add_argument("--debug", default=False, action="store_true")

    parser.add_argument("--output_pred_save", choices=bool_choices)
    parser.add_argument(
        "--output_pred_dir",
        type=str,
        help="target location to save the outputs if output_pred_save is set to true",
    )
    parser.add_argument(
        "--output_pred_shard_size",
        type=int,
        default=1000,
        help="specify the shard size of the outputted predictions"
    )

    # Weights and Biases setup
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, help="The W&B entity")
    parser.add_argument(
        "--wandb_mode", choices=["offline", "online", "disabled"], default="offline"
    )

    # Misc. options
    a = parser.parse_args()

    # Processing for ease of use
    if a.use_targets == "smoothed":
        a.use_smoothed_targets = True
    elif a.use_targets == "original":
        a.use_smoothed_targets = False

    bool_args = [
        "output_pred_save",
        "use_noise_seed",
    ]
    # Process the boolean arguments from strings
    for bool_arg in bool_args:
        str_val = getattr(a, bool_arg)
        setattr(a, bool_arg, str_val == "true")
    # a.save_all_model_weights = (a.save_all_model_weights == "true")
    # a.output_pred_save = (a.output_pred_save=="true")

    return a


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

def main(
    args: argparse.Namespace,
    # Extra arguments for testing purposes
    return_model: bool = False,
) -> None:
    """
    1. Load datasets
    2. Prepare the datasets
    3. Prepare the logging function
    4. Evaluation run; optionally write to disk
    """
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # # Set up noise seeds
    # if args.use_noise_seed:
    #     eff_noise_seed_train = args.noise_seed_train
    #     eff_noise_seed_val   = args.noise_seed_val
    #     eff_noise_seed_test  = args.noise_seed_test
    # else:
    #     eff_noise_seed_train = None
    #     eff_noise_seed_val   = None
    #     eff_noise_seed_test  = None

    # if args.noise_to_signal_ratio != 0:
    #     logging.info(f"Using seed as {eff_noise_seed_train} for the training set")
    #     logging.info(f"Using seed as {eff_noise_seed_val} for the val set")
    #     logging.info(f"Using seed as {eff_noise_seed_test} for the test set (load later)")
    # else:
    #     logging.info(f"Not adding noise!")


    #########################################################
    # 1. Basic setup...
    ref_data_dir_base  = args.ref_data_dir_base
    pred_mmg_dir_base = args.input_pred_mmg_dir_base
    str_nu_list = (
        args.data_input_nus
    )  # nu in string form (to preserve decimals properly)
    nu_list = [float(str_nu) for str_nu in str_nu_list]
    N_freqs = len(nu_list)
    logging.info(f"ref_data_dir_base: {ref_data_dir_base}")
    logging.info(f"nu values received: {str_nu_list}")

    # Prepare to set up the model...
    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training on device: %s", device)

    #########################################################
    # 2. Set up the logging function
    # N_epochs = args.n_epochs
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


    # 3. Load the model from disk
    model_fp   = args.central_model_fp
    logging.info(f"Attempting to load model from path {model_fp}...")
    model_state_dict = torch.load(model_fp, map_location=device)

    logging.info(f"Attempting to load summary file from  {args.in_central_summary_fp}")
    summary_results_dd = load_field_in_yaml_file(f"freq_idx_{args.freq_idx}", args.in_central_summary_fp)
    model_loaded = load_MMGUBlock_from_state_dict(
        model_state_dict,
        summary_results_dd,
        hss=None,
    )
    logging.info(f"Loaded the model successfully!")

    #########################################################
    # 4. Evaluate on all the datasets, then optionally write the outputs to disk
    # 4a. load the test set and predictions
    # Common setup
    base_output_dir = args.output_pred_dir if args.output_pred_save else None
    pred_scobj_dir_base = args.input_pred_scobj_dir_base

    # dset_list = ["train", "val", "test"]
    dset_list = args.dset_names
    # expt_info_list = [pred_train_meta_dd, pred_val_meta_dd, pred_test_meta_dd]
    last_eval_dict = {}
    key_max_num_chars = max(len(key) for key in cart_loss_fn_dd.keys())
    cart_dd_list = []

    for i, dset in enumerate(dset_list):
        #########################################################
        # 4b. Load the relevant dataset
        logging.info(f"Loading {dset}...")
        truncate_num = args.truncate_nums[i]
        eff_noise_seed = args.noise_seeds[i] if args.use_noise_seed else None

        # Prepare the file directory names
        ref_dset_dirs = [
            os.path.join(ref_data_dir_base, f"{dset}_measurements_nu_{nu}")
            for nu in str_nu_list
        ]
        pred_dset_scobj_dir = os.path.join(pred_scobj_dir_base, f"{dset}_scattering_objs")
        pred_dset_mmg_rel_dirs = [
            f"{dset}_gammas_nu_{nu}" for nu in str_nu_list
        ]

        # Load the references
        logging.info(f"Loading the reference {dset} set")
        ref_dset_dd, ref_dset_meta_dd = load_multifreq_dataset(
            ref_dset_dirs,
            truncate_num=truncate_num,
            # key_replacement=key_replacement,
            noise_to_sig_ratio=args.noise_to_signal_ratio,
            add_noise_to="d_rs",
            nan_mode="skip",
            load_cart=True,
            noise_seed=eff_noise_seed,
            noise_norm_mode=args.noise_norm_mode,
        )

        # Load the predictions
        logging.info(f"Loading predictions for the {dset} set")
        pred_dset_dd, pred_dset_meta_dd = load_predictions_dataset(
            pred_dset_scobj_dir,
            pred_mmg_dir_base,
            pred_dset_mmg_rel_dirs,
            truncate_num=truncate_num,
            nan_mode="skip",
        )
        ref_dset_dd_short  = dict(kv_shrinker(k, v) for (k, v) in ref_dset_dd.items())
        pred_dset_dd_short = dict(kv_shrinker(k, v) for (k, v) in pred_dset_dd.items())
        logging.info(f"ref_{dset}_dd has entries with shapes: {ref_dset_dd_short}")
        logging.info(f"pred_{dset}_dd has entries with shapes: {pred_dset_dd_short}")

        ## 4c. Extract the relevant fields
        ref_dset_q_cart_orig = ref_dset_dd[Q_CART]
        if args.use_smoothed_targets:
            logging.info(f"Using smoothed targets for dataset {dset}")
            ref_dset_q_cart = ref_dset_dd[Q_CART_LPF][:, -1, ...]
        else:
            logging.info(f"Using original targets for dataset {dset}")
            ref_dset_q_cart = ref_dset_q_cart_orig
        pred_dset_q_cart = pred_dset_dd[Q_CART]
        pred_dset_gamma_cart = pred_dset_dd[GAMMA_CART]
        ref_dset_d_rs = ref_dset_dd[D_RS]

        ## 4d. Set up the dataset object and data loader
        dset_obj = setup_preprocessed_predictions_dataset(
            pred_dset_q_cart,
            ref_dset_d_rs,
            ref_dset_q_cart,
            ref_dset_q_cart_orig,
            pred_dset_gamma_cart,
        )
        dset_dloader = torch.utils.data.DataLoader(
            dset_obj,
            batch_size=args.log_batch_size,
            num_workers=1,
            prefetch_factor=2,
        )

        ## 4e. Evaluating the dataset
        logging.info(f"Evaluating dset {dset}...")
        expt_info = pred_dset_meta_dd

        # Prepare the output directory if relevant
        if args.output_pred_save:
            dset_output_dir = os.path.join(
                base_output_dir,
                f"{dset}_scattering_objs",
            )
            os.makedirs(dset_output_dir, exist_ok=True)
        else:
            dset_output_dir = None
        logging.info(f"Making predictions on {dset} set; saving to {dset_output_dir}")

        # Main predictions code
        t0 = time.perf_counter()
        cart_preds, cart_dd = make_preds_on_dataset_only_cartesian(
            model=model_loaded,
            dloader=dset_dloader,
            device=device,
            output_dir=dset_output_dir,
            shard_size=args.output_pred_shard_size,
            experiment_info=expt_info,
            format_str=SCOBJ_FMT_STR,
            use_orig_idcs=False,
            evaluate_outputs=True,
            cart_loss_fn_dict=cart_loss_fn_dd,
            model_takes_x_mh=True,
            model_outputs_tuple=False,
        )
        t1 = time.perf_counter()

        mean_dict   = {k: torch.mean(v).item() for (k,v) in cart_dd.items()}
        stdev_dict  = {k: torch.std(v).item() for (k,v) in cart_dd.items()}
        median_dict = {k: torch.median(v).item() for (k,v) in cart_dd.items()}
        cart_dd_list.append(cart_dd)


        for key in cart_dd.keys():
            last_eval_dict[f"{dset}_{key}"]        = mean_dict[key]
            last_eval_dict[f"{dset}_{key}_stdev"]  = stdev_dict[key]
            last_eval_dict[f"{dset}_{key}_median"] = median_dict[key]

            key_ljust = (key + ":").ljust(key_max_num_chars+1)
            logging.info(f"{dset} {key_ljust} {mean_dict[key]:.5e}±{stdev_dict[key]:.3e}")

        logging.info(f"Finished evaluating dataset {dset}... ({t1-t0:.3f}s)")

    # import pdb; pdb.set_trace()
    logging.info(f"Compilation of the metrics for different datasets...")
    for i, dset in enumerate(dset_list):
        cart_dd = cart_dd_list[i]
        for key in cart_dd.keys():
            key_ljust = (key + ":").ljust(key_max_num_chars+1)
            mean_val  = last_eval_dict[f"{dset}_{key}"]
            stdev_val = last_eval_dict[f"{dset}_{key}_stdev"]
            logging.info(
                f"{dset} {key_ljust} {mean_val:.5e}±{stdev_val:.3e}"
            )

    logging.info(f"Compressed representation:")
    print(f"Evaluation metrics on datasets {dset_list}:")
    for key in cart_loss_fn_dd.keys():
        key_ljust = (key + ":").ljust(key_max_num_chars+1)
        msg = f""
        for i in range(len(dset_list)):
            dset = dset_list[i]
            mean_val  = last_eval_dict[f"{dset}_{key}"]
            stdev_val = last_eval_dict[f"{dset}_{key}_stdev"]
            msg += f"{mean_val:.5e}±{stdev_val:.3e} "
        msg = msg[:-1]
        print_msg = f"Selected model {key_ljust} {msg}"
        logging.info(print_msg)
        print(print_msg)
        last_eval_dict["rel_l2_for_table"] = msg

    try:
        init_central_summary_dd = load_yaml_to_dict(args.out_central_summary_fp)
    except FileNotFoundError:
        # just make a new one if none exists
        os.makedirs(os.path.split(args.out_central_summary_fp)[0], exist_ok=True)
        init_central_summary_dd = dict()

    summary_key = f"freq_idx_{args.freq_idx}"
    out_central_summary_dd = {
        **init_central_summary_dd,
        summary_key: {
            **summary_results_dd,
            **last_eval_dict,
            "central_model_fp": model_fp,
        },
    }

    logging.info(f"Updating eval info in summary file {args.out_central_summary_fp}.")
    save_dict_to_yaml(
        out_central_summary_dd,
        args.out_central_summary_fp,
    )

    logging.info("Evaluation finished!")
    if return_model:
        return model_0
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
    main(a, return_model=False)
