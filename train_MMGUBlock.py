# Train a single MMGUBlock
# (Measurement misfit gradient update...)
# Expects to receive just one frequency

import logging
from typing import List
import argparse
# from timeit import default_timer
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

def wandb_entity_arg_type(value: str):
    """argparse type for --wandb_entity: treat "none"/"null" (any case) as
    no entity, so wandb.init() falls back to the caller's own default entity
    """
    if value is None or value.strip().lower() in ("none", "null"):
        return None
    return value

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    bool_choices = ["true", "false"]

    # Expect a single frequency but let the argument accept a list to avoid
    # needing to re-write the data loading code
    parser.add_argument("--data_input_nus", type=str, nargs="+")
    # parser.add_argument("--data_input_nus", type=str) # actually single frequency here

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
        "--central_summary_fp",
        type=str,
        help="Generate a centralized summary yaml file with hyperparameters and errors",
    )
    parser.add_argument(
        "--central_model_dir",
        type=str,
        help="Save the best weights to a centralized location",
    )
    parser.add_argument(
        "--central_model_fp",
        type=str,
        help="Weight name",
        default=None,
    )
    parser.add_argument("--train_results_fp", type=str)
    parser.add_argument("--model_weights_dir", type=str)

    ### Training/validation-related arguments ###
    parser.add_argument(
        "--use_targets", choices=["original", "smoothed"], default="original",
        help=(
            "Set target as original or smoothed"
        )
    )
    parser.add_argument("--truncate_num_train", type=int)
    parser.add_argument("--truncate_num_val", type=int)
    parser.add_argument("--truncate_num_test", type=int)
    parser.add_argument("--log_train_subset_frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=35675)
    parser.add_argument("--use_noise_seed", choices=bool_choices, default="false")
    parser.add_argument("--noise_seed_train",  type=int, default=10128329)
    parser.add_argument("--noise_seed_val",    type=int, default=20293834)
    parser.add_argument("--noise_seed_test",   type=int, default=30943792)
    parser.add_argument("--noise_norm_mode", choices=["l2", "inf"], default="inf")
    parser.add_argument("--n_cnn_layers_2d",   type=int, default=4)
    parser.add_argument("--n_cnn_channels_2d", type=int, default=48)
    parser.add_argument("--kernel_size_2d", type=int, default=7)
    parser.add_argument("--init_cnn_scale", type=float, default=1)
    parser.add_argument("--learn_cnn_scale", choices=bool_choices, default="false")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--log_batch_size", type=int, default=100, help="batch size while logging")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr_init_base", type=float, default=3e-4)
    parser.add_argument("--eta_min_base", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay_base", type=float, default=0.0)
    parser.add_argument("--n_epochs_per_log", type=int, default=5)
    parser.add_argument(
        "--noise_to_signal_ratio", default=None, type=float
    )  # train and test with noise

    parser.add_argument("--lr_decrease_factor", default=1.0, type=float)
    parser.add_argument("--freq_lvl", default=1, type=int, help="Which frequency level to use for lr decrease (start from 1, 2,...); defaults to 1")
    parser.add_argument("--freq_idx", default=None, type=int, help="The index of the frequency block at use in the naming schemes; defaults to match freq_lvl")

    ### Logging options ###
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--save_all_model_weights", choices=bool_choices, default="true")
    parser.add_argument("--selection_field", default="val_rel_l2")
    parser.add_argument("--selection_mode", default="min", choices=["min", "max"])

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
    parser.add_argument("--wandb_entity", type=wandb_entity_arg_type, default=None, help="The W&B entity")
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
    a.freq_idx = a.freq_idx if a.freq_idx is not None else a.freq_lvl

    bool_args = [
        "save_all_model_weights",
        "output_pred_save",
        "use_noise_seed",
        "learn_cnn_scale",
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
    4. Train NN
    5. Save weights/results
    6. Evaluation run; optionally write to disk
    """
    lr_lvl_adjust = args.lr_decrease_factor ** (args.freq_lvl - 1)
    lr_init_eff = args.lr_init_base * lr_lvl_adjust
    eta_min_eff = args.eta_min_base * lr_lvl_adjust
    weight_decay_eff = args.weight_decay_base * lr_lvl_adjust
    logging.info(
        f"Adjusting the learning rate by a factor of {lr_lvl_adjust} "
        f"since freq_lvl={args.freq_lvl} "
        f"and lr_decreas_factor={args.lr_decrease_factor}. "
        f"Using lr_init_eff={lr_init_eff} "
        f"(c.f.: lr_init_base={args.lr_init_base})."
    )

    if not os.path.isdir(args.model_weights_dir):
        # os.mkdir(args.model_weights_dir)
        os.makedirs(args.model_weights_dir, exist_ok=True)
    if not os.path.isdir(args.central_model_dir):
        os.makedirs(args.central_model_dir, exist_ok=True)

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up noise seeds
    if args.use_noise_seed:
        eff_noise_seed_train = args.noise_seed_train
        eff_noise_seed_val   = args.noise_seed_val
        eff_noise_seed_test  = args.noise_seed_test
    else:
        eff_noise_seed_train = None
        eff_noise_seed_val   = None
        eff_noise_seed_test  = None

    if args.noise_to_signal_ratio != 0:
        logging.info(f"Using seed as {eff_noise_seed_train} for the training set")
        logging.info(f"Using seed as {eff_noise_seed_val} for the val set")
        logging.info(f"Using seed as {eff_noise_seed_test} for the test set (load later)")
    else:
        logging.info(f"Not adding noise!")

    #########################################################
    # 1. Load data
    ref_data_dir_base  = args.ref_data_dir_base
    pred_mmg_dir_base = args.input_pred_mmg_dir_base
    str_nu_list = (
        args.data_input_nus
    )  # nu in string form (to preserve decimals properly)
    nu_list = [float(str_nu) for str_nu in str_nu_list]
    N_freqs = len(nu_list)
    logging.info(f"ref_data_dir_base: {ref_data_dir_base}")
    logging.info(f"nu values received: {str_nu_list}")

    ### Set up directory names ###
    # Reference
    ref_train_dirs = [
        os.path.join(ref_data_dir_base, f"train_measurements_nu_{nu}")
        for nu in str_nu_list
    ]
    ref_val_dirs = [
        os.path.join(ref_data_dir_base, f"val_measurements_nu_{nu}")
        for nu in str_nu_list
    ]
    # Predictions
    pred_scobj_dir_base  = args.input_pred_scobj_dir_base
    pred_train_scobj_dir = os.path.join(pred_scobj_dir_base, f"train_scattering_objs")
    pred_val_scobj_dir   = os.path.join(pred_scobj_dir_base, f"val_scattering_objs")
    pred_train_mmg_rel_dirs = [
        f"train_gammas_nu_{nu}" for nu in str_nu_list
    ]
    pred_val_mmg_rel_dirs = [
        f"val_gammas_nu_{nu}" for nu in str_nu_list
    ]

    ### Load the reference training dataset ###
    logging.info(f"Loading the reference training set")
    logging.info(f"ref_train_dirs={ref_train_dirs}")
    ref_train_dd, ref_train_meta_dd = load_multifreq_dataset(
        ref_train_dirs,
        truncate_num=args.truncate_num_train,
        # key_replacement=key_replacement,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_rs",
        nan_mode="skip",
        load_cart=True,
        noise_seed=eff_noise_seed_train,
        noise_norm_mode=args.noise_norm_mode,
    )

    ### Load the predictions for the training dataset ###
    logging.info(f"Loading predictions for the training set")
    pred_train_dd, pred_train_meta_dd = load_predictions_dataset(
        pred_train_scobj_dir,
        pred_mmg_dir_base,
        pred_train_mmg_rel_dirs,
        truncate_num=args.truncate_num_train,
        nan_mode="skip",
    )
    ref_train_dd_short  = dict(kv_shrinker(k, v) for (k, v) in ref_train_dd.items())
    pred_train_dd_short = dict(kv_shrinker(k, v) for (k, v) in pred_train_dd.items())
    logging.info(f"ref_train_dd has entries with shapes: {ref_train_dd_short}")
    logging.info(f"pred_train_dd has entries with shapes: {pred_train_dd_short}")

    ### Evaluation dataset -- use validation set ###
    logging.info(f"Loading the reference validation set")
    logging.info(f"ref_train_dirs={ref_val_dirs}")
    ref_val_dd, ref_val_meta_dd = load_multifreq_dataset(
        ref_val_dirs,
        truncate_num=args.truncate_num_val,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_rs",
        nan_mode="skip",
        load_cart=True,
        noise_seed=eff_noise_seed_val,
        noise_norm_mode=args.noise_norm_mode,
    )

    logging.info(f"Loading predictions for the validation set")
    pred_val_dd, pred_val_meta_dd = load_predictions_dataset(
        pred_val_scobj_dir,
        pred_mmg_dir_base,
        pred_val_mmg_rel_dirs,
        truncate_num=args.truncate_num_val,
        nan_mode="skip",
    )

    ref_val_dd_short  = dict(kv_shrinker(k, v) for (k, v) in ref_val_dd.items())
    pred_val_dd_short = dict(kv_shrinker(k, v) for (k, v) in pred_val_dd.items())
    logging.info(f"ref_val_dd has entries with shapes: {ref_val_dd_short}")
    logging.info(f"pred_val_dd has entries with shapes: {pred_val_dd_short}")

    #########################################################
    # 2. Extract the relevant fields
    ref_train_q_cart_orig = ref_train_dd[Q_CART]
    ref_val_q_cart_orig  = ref_val_dd[Q_CART]
    if args.use_smoothed_targets:
        logging.info(f"Using smoothed targets for training and validation")
        ref_train_q_cart = ref_train_dd[Q_CART_LPF][:, -1, ...]
        ref_val_q_cart   = ref_val_dd[Q_CART_LPF][:, -1, ...]
    else:
        logging.info(f"Using original targets for training and validation")
        ref_train_q_cart = ref_train_q_cart_orig
        ref_val_q_cart   = ref_val_q_cart_orig
    pred_train_q_cart = pred_train_dd[Q_CART]
    pred_val_q_cart  = pred_val_dd[Q_CART]

    # For simplicity assume we have access to gamma_cart
    pred_train_gamma_cart = pred_train_dd[GAMMA_CART]
    pred_val_gamma_cart  = pred_val_dd[GAMMA_CART]

    ref_train_d_rs = ref_train_dd[D_RS]
    ref_val_d_rs  = ref_val_dd[D_RS]

    k_vals = ref_train_dd[OMEGA_SF]
    x_vals = ref_train_dd[X_VALS]
    N_x = x_vals.shape[0]

    N_train = ref_train_q_cart.shape[0]
    N_eval  = ref_val_q_cart.shape[0]

    train_dset = setup_preprocessed_predictions_dataset(
        pred_train_q_cart,
        ref_train_d_rs,
        ref_train_q_cart,
        ref_train_q_cart_orig,
        pred_train_gamma_cart,
    )
    val_dset = setup_preprocessed_predictions_dataset(
        pred_val_q_cart,
        ref_val_d_rs,
        ref_val_q_cart,
        ref_val_q_cart_orig,
        pred_val_gamma_cart,
    )

    # Send to the data loader
    train_dloader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size,
        num_workers=1,
        prefetch_factor=2,
    )
    val_dloader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.log_batch_size,
        num_workers=1,
        prefetch_factor=2,
    )

    # Create a subset of the training set for faster logging
    # if log_train_subset_frac=1 we just do the whole set
    rng = np.random.default_rng(args.seed+17934)
    subset_size = int(args.log_train_subset_frac * args.truncate_num_train)
    train_subset_idcs = rng.choice(args.truncate_num_train, size=subset_size, replace=False)
    train_subset = torch.utils.data.Subset(
        dataset=train_dset,
        indices=train_subset_idcs,
    )
    train_subset_dloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.log_batch_size,
        num_workers=1,
        prefetch_factor=2,
    )

    # Prepare to set up the model...
    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training on device: %s", device)

    #########################################################
    # 3. Set up the logging function
    N_epochs = args.n_epochs
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
    current_best_info = {
        "epoch": None,
        "batch_idx": None,
        "model": None,
        "field_val": None,
        "train_log_dd": None,
    }

    misc_info =  {
        "n_train":  N_train,
        "n_eval":   N_eval,
        "n_freqs":  N_freqs,
        "n_x_vals": N_x,
        "source_nu_list": nu_list,
        "seed": args.seed,
        "log_train_subset_frac": args.log_train_subset_frac,
        "log_batch_size": args.log_batch_size,
    }
    hyperparam_info = {
        # Arch-related
        "n_cnn_layers_2d": args.n_cnn_layers_2d,
        "n_cnn_channels_2d": args.n_cnn_channels_2d,
        "kernel_size_2d": args.kernel_size_2d,
        "learn_cnn_scale":  args.learn_cnn_scale,
        "init_cnn_scale":  args.init_cnn_scale,

        # Opt-related
        "lr_decrease_factor": args.lr_decrease_factor,
        "freq_lvl": args.freq_lvl,
        "freq_idx": args.freq_idx,
        "lr_init_eff": lr_init_eff,
        "eta_min_eff": eta_min_eff,
        "weight_decay_eff": weight_decay_eff,
        "lr_init_base": args.lr_init_base,
        "eta_min_base": args.eta_min_base,
        "weight_decay_base": args.weight_decay_base,
        "batch_size": args.batch_size,
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
        # reinit=True,
        resume=None,
        settings=wandb.Settings(start_method="fork"),
    ) as wandbrun:
        # First, set up the logging function
        def log_function(model_0, epoch_local):
            nonlocal args
            nonlocal train_dloader, val_dloader, train_subset_dloader
            nonlocal current_best_info
            nonlocal cart_loss_fn_dd
            # 1. Perform the evaluation (forward pass only)
            with torch.no_grad():
                # 1a. training (sub)set
                outs = evaluate_losses_on_dataloader_only_cartesian(
                    model_0, train_subset_dloader,
                    cart_loss_fn_dd,
                    device,
                    model_outputs_tuple=False,
                    model_takes_x_mh=True,
                )
                train_loss_dd = outs
                # train_polar_rel_err = torch.mean(train_loss_dd["polar_rel_l2"]).item()
                train_cart_rel_err = torch.mean(train_loss_dd["cart_rel_l2"]).item()

                logging.info(f"Epoch: {epoch_local + epoch_stagger}")
                logging.info(
                    "(cart)  Train MSE: {:.5e}, Train Rel L2: {:.5f}, Train PSNR: {:.3f}".format(
                        torch.mean(train_loss_dd["cart_mse"]).item(),
                        torch.mean(train_loss_dd["cart_rel_l2"]).item(),
                        torch.mean(train_loss_dd["cart_psnr"]).item(),
                    )
                )
                logging.info(
                    "(cart)  Train stdevs: {:.5e}; {:.5e}; {:.5e}".format(
                        torch.std(train_loss_dd["cart_mse"]).item(),
                        torch.std(train_loss_dd["cart_rel_l2"]).item(),
                        torch.std(train_loss_dd["cart_psnr"]).item(),
                    )
                )
                # 1b. validation set
                outs = evaluate_losses_on_dataloader_only_cartesian(
                    model_0, val_dloader,
                    cart_loss_fn_dd, # polar_to_cart_fn,
                    device,
                    model_takes_x_mh=True,
                    model_outputs_tuple=False,
                )
                val_loss_dd = outs
                val_cart_rel_err = torch.mean(val_loss_dd["cart_rel_l2"]).item()
                logging.info(
                    "(cart)    Val MSE: {:.5e}, Val Rel L2: {:.5f}, Val PSNR: {:.3f}".format(
                        torch.mean(val_loss_dd["cart_mse"]).item(),
                        torch.mean(val_loss_dd["cart_rel_l2"]).item(),
                        torch.mean(val_loss_dd["cart_psnr"]).item(),
                    )
                )
                logging.info(
                    "(cart)    Val stdevs: {:.5e}; {:.5e}; {:.5e}".format(
                        torch.std(val_loss_dd["cart_mse"]).item(),
                        torch.std(val_loss_dd["cart_rel_l2"]).item(),
                        torch.std(val_loss_dd["cart_psnr"]).item(),
                    )
                )
                # 1c. Weight norm
                weight_norm = torch.norm(
                    torch.cat([x.view(-1) for x in model_0.parameters()]), 2
                )
                logging.info("  Weight L2 norm: {:.3e}".format(weight_norm.item()))

            # Memory info in case of a memory leak
            mem_info = get_memory_info(device=device, print_msg=False)
            logging.info(f"{mem_info}")

            # 2. Collect the relevant information into the logs
            epoch_overall = epoch_local + epoch_stagger
            train_loss_entries = {
                f"train_{k}": torch.mean(v).item() for (k, v) in train_loss_dd.items()
            }
            val_loss_entries = {
                f"val_{k}": torch.mean(v).item() for (k, v) in val_loss_dd.items()
            }
            # logging.info(f"train_loss_entries={train_loss_entries}")
            # logging.info(f"val_loss_entries={val_loss_entries}")
            train_log_dd = {
                ##### These entries change each round... #####
                # Epoch/batch information
                "epoch": epoch_overall,

                # Evaluation metrics
                **train_loss_entries,
                **val_loss_entries,
                "weight_norm": weight_norm.item(),

                # Same info every time
                **misc_info,
                **hyperparam_info,
            }
            # 3. Write the log entry to disk and wandb; also save weights if relevant
            write_result_to_file(args.train_results_fp, **train_log_dd)

            if args.save_all_model_weights:
                # TODO: figure out where to save this???
                model_fp = os.path.join(
                    args.model_weights_dir,
                    f"model_params_f{args.freq_idx}_epoch_{epoch_overall}.pickle"
                )
                logging.info(f"Saving weights to {model_fp}")
                torch.save(
                    model_0.state_dict(),
                    model_fp,
                )

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
                current_best_info["train_log_dd"] = {**train_log_dd}
            errs = (
                train_cart_rel_err,
                val_cart_rel_err,
            )
            return errs

    #########################################################
    # 4. Set up the model and train
    t0 = time.perf_counter()
    model_0 = MMGUBlock(
        N_x,
        N_x,
        N_cnn_layers=args.n_cnn_layers_2d,
        c2d_hidden=args.n_cnn_channels_2d,
        kernel_size=args.kernel_size_2d,
        c2d_input=2,
        skip_connection=True,
        learn_cnn_scale=args.learn_cnn_scale,
        init_cnn_scale=args.init_cnn_scale,
    )

    model_0 = train(
        model_0,
        train_loader=train_dloader,
        n_epochs=N_epochs,
        lr_init=lr_init_eff,
        weight_decay=weight_decay_eff,
        eta_min=eta_min_eff,
        momentum=args.momentum,
        device=device,
        n_epochs_per_log=args.n_epochs_per_log,
        log_function=log_function,
        loss_function=loss_module_0,
        use_cart_output=True,
    )

    t1 = time.perf_counter()
    logging.info(f"Optimization complete!") # easier to grep for
    logging.info(f"Fine-tuning finished after {t1-t0:.3f}s")

    #########################################################
    # 5. Save the model and results to disk
    # 5a. Select the relevant information
    cbi_model = current_best_info["model"]
    cbi_epoch = current_best_info["epoch"]
    cbi_field_val = current_best_info["field_val"]
    cbi_train_log_dd = current_best_info["train_log_dd"]
    logging.info(
        f"Selected the model from epoch {cbi_epoch}, with "
        f"field '{args.selection_field}' value {cbi_field_val} "
        f"(selection mode: {args.selection_mode})."
    )
    logging.info(f"Selected train log... {cbi_train_log_dd}")

    # 5b. Save model to disk
    # TODO: identify the centralized location to save weights to
    if not args.save_all_model_weights:
        model_fp = os.path.join(
            args.model_weights_dir,
            f"epoch_{cbi_epoch}.pickle"
        )
        logging.info(f"Saving model weights to {model_fp}")
        torch.save(
            cbi_model.state_dict(),
            model_fp
        )

    # Centralized location
    central_model_fp = os.path.join(
        args.central_model_dir,
        (
            args.central_model_fp
            if args.central_model_fp is not None
            else f"model_params_{args.freq_idx}.pickle"
            # else f"model_weights_f{args.freq_idx}.pickle"
        ),
    )
    logging.info(f"Saving model weights to {central_model_fp} (in the centralized location)")
    torch.save(
        cbi_model.state_dict(),
        central_model_fp,
    )

    # 5c. Save results to disk if not there yet...
    # Save to the central summary fp
    try:
        init_central_summary_dd = load_yaml_to_dict(args.central_summary_fp)
    except FileNotFoundError:
        # just make a new one if none exists
        os.makedirs(os.path.split(args.central_summary_fp)[0], exist_ok=True)
        init_central_summary_dd = dict()

    summary_key = f"freq_idx_{args.freq_idx}"
    out_central_summary_dd = {
        **init_central_summary_dd,
        summary_key: cbi_train_log_dd
    } # make a copy
    # out_central_summary_dd[summary_key] = cbi_train_log_dd
    logging.info(f"Saving summary info to {args.central_summary_fp}")
    save_dict_to_yaml(
        out_central_summary_dd,
        args.central_summary_fp,
    )

    # 6. Evaluate on all the datasets, then optionally write the outputs to disk
    # 6a. load the test set and predictions
    base_output_dir = args.output_pred_dir if args.output_pred_save else None
    # Try to load the test set as well
    try:
        ref_test_dirs = [
            os.path.join(ref_data_dir_base, f"test_measurements_nu_{nu}")
            for nu in str_nu_list
        ]
        pred_test_scobj_dir = os.path.join(pred_scobj_dir_base, f"test_scattering_objs")
        pred_test_mmg_rel_dirs = [
            f"test_gammas_nu_{nu}" for nu in str_nu_list
        ]

        logging.info(f"Loading the reference test set")
        logging.info(f"ref_test_dirs={ref_test_dirs}")
        ref_test_dd, ref_test_meta_dd = load_multifreq_dataset(
            ref_test_dirs,
            truncate_num=args.truncate_num_test,
            noise_to_sig_ratio=args.noise_to_signal_ratio,
            add_noise_to="d_rs",
            nan_mode="skip",
            load_cart=True,
            noise_seed=eff_noise_seed_test,
            noise_norm_mode=args.noise_norm_mode,
        )
        logging.info(f"Successfully loaded the reference test set")
        ref_test_d_rs = ref_test_dd[D_RS]
        ref_test_q_cart = (
            ref_test_dd[Q_CART]
            if args.use_targets == "original"
            else ref_test_dd[Q_CART_LPF][:, -1, ...]
        )
        ref_test_q_cart_orig = ref_test_dd[Q_CART]

        logging.info(f"Loading predictions for the test set")
        pred_test_dd, pred_test_meta_dd = load_predictions_dataset(
            pred_test_scobj_dir,
            pred_mmg_dir_base,
            pred_test_mmg_rel_dirs,
            truncate_num=args.truncate_num_test,
            nan_mode="skip",
        )
        logging.info(f"Successfully loaded the predictions for the test set")
        pred_test_q_cart = pred_test_dd[Q_CART]
        pred_test_gamma_cart = pred_test_dd[GAMMA_CART]

        # Extract the relevant fields and set up the dataset
        test_dset = setup_preprocessed_predictions_dataset(
            pred_test_q_cart,
            ref_test_d_rs,
            ref_test_q_cart,
            ref_test_q_cart_orig,
            pred_test_gamma_cart,
        )
        # Send to the data loader
        test_dloader = torch.utils.data.DataLoader(
            test_dset,
            batch_size=args.log_batch_size,
            num_workers=1,
            prefetch_factor=2,
        )

        loaded_test_set = True
    except Exception as e:
        logging.ERROR(f"Unable to load the test set :(")
        logging.ERROR(f"More info: {str(e)}")
        loaded_test_set = False

    log_train_dloader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.log_batch_size,
        num_workers=1,
        prefetch_factor=2,
    )

    dset_list = ["train", "val"]
    dloader_list = [log_train_dloader, val_dloader]
    expt_info_list = [pred_train_meta_dd, pred_val_meta_dd]
    last_eval_dict = {}
    key_max_num_chars = max(len(key) for key in cart_loss_fn_dd.keys())

    if loaded_test_set:
        dset_list = [*dset_list, "test"]
        dloader_list = [*dloader_list, test_dloader]
        expt_info_list = [*expt_info_list, pred_test_meta_dd]

    for i, dset in enumerate(dset_list):
        logging.info(f"Evaluating dset {dset}...")
        dloader = dloader_list[i]
        expt_info = expt_info_list[i]

        if args.output_pred_save:
            dset_output_dir = os.path.join(
                base_output_dir,
                f"{dset}_scattering_objs",
            )
            os.makedirs(dset_output_dir, exist_ok=True)
        else:
            dset_output_dir = None
        logging.info(f"Making predictions on {dset} set; saving to {dset_output_dir}")

        cart_preds, cart_dd = make_preds_on_dataset_only_cartesian(
            model=cbi_model,
            dloader=dloader,
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
        mean_dict  = {k: v.mean().item() for (k,v) in cart_dd.items()}
        stdev_dict = {k: v.std().item() for (k,v) in cart_dd.items()}
        for key in cart_dd.keys():
            last_eval_dict[f"{dset}_{key}"]       = mean_dict[key]
            last_eval_dict[f"{dset}_{key}_stdev"] = stdev_dict[key]

            key_ljust = (key + ":").ljust(key_max_num_chars+1)
            logging.info(f"{dset} {key_ljust} {mean_dict[key]:.5e}±{stdev_dict[key]:.3e}")

    logging.info(f"Compressed representation:")
    print(f"Evaluation metrics on datasets {dset_list}:")
    for key in cart_loss_fn_dd.keys():
        key_ljust = (key + ":").ljust(key_max_num_chars+1)
        msg = f"Selected model {key_ljust}"
        for i in range(len(dset_list)):
            dset = dset_list[i]
            mean_val  = last_eval_dict[f"{dset}_{key}"]
            stdev_val = last_eval_dict[f"{dset}_{key}_stdev"]
            msg += f"{mean_val:.5e}±{stdev_val:.3e} "
        msg = msg[:-1]
        logging.info(msg)
        print(msg)


    # TODO: maybe should update the central summary file now
    out_central_summary_dd = {
        **out_central_summary_dd,
        summary_key: {
            **cbi_train_log_dd,
            **last_eval_dict,
        },
    }
    logging.info(f"Updating eval info in summary file {args.central_summary_fp}.")
    save_dict_to_yaml(
        out_central_summary_dd,
        args.central_summary_fp,
    )

    # Exit when finished, optionally returning the model
    logging.info("Optimization+Evaluation finished!")
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
