# Train the MFISNet-Fused variant from the previous paper
# NOTE: when given a single frequency, this is equivalent to FYNet.
# This version of the code is better-maintained, so I dropped
# train_FYNet.py and direct invocation of FYNet.
# This new-interface variant of the code

import logging
from typing import List
import argparse
from timeit import default_timer
import os
import numpy as np
import torch
import wandb
import os, psutil  # to fetch memory usage
import random

from src.data.datasets import (
    LinearData,
    setup_dataset_linear as setup_single_dataset,
)
from src.data.add_noise import add_noise_to_d
from src.data.data_io import (
    load_dir,
    load_multifreq_dataset,
    load_single_dir_slice,
)
from src.models.MFISNet_Fused import MFISNet_Fused
from src.training_utils.train_loop import train, evaluate_losses_on_dataloader
from src.training_utils.loss_functions import MSEModule

from src.utils.logging_utils import FMT, TIMEFMT, write_result_to_file, hash_dict

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

def wandb_entity_arg_type(value: str):
    """argparse type for --wandb_entity: treat "none"/"null" (any case) as
    no entity, so wandb.init() falls back to the caller's own default entity
    """
    if value is None or value.strip().lower() in ("none", "null"):
        return None
    return value

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir_base",
        type=str,
        help="Indicate the directory containing all the measurement folders"
        " corresponding to the relevant frequencies and data subsets",
    )
    parser.add_argument("--data_input_nus", type=str, nargs="+")

    # New option to use smoothed targets or not
    # Option to use smoothed targets or not
    # alternate option
    parser.add_argument(
        "--train_targets", choices=["original", "smoothed"], default="original",
        help=(
            "Set target as original or smoothed"
        )
    )
    parser.add_argument(
        "--eval_targets", choices=["original", "smoothed"], default="original",
        help=(
            "Set target as original or smoothed"
        )
    )
    parser.add_argument("--use_smoothed_targets", default=False, action="store_true")
    parser.add_argument("--use_original_targets", action="store_false", dest="use_smoothed_targets")

    parser.add_argument("--train_results_fp")
    parser.add_argument("--model_weights_dir")
    parser.add_argument("--truncate_num", type=int)
    parser.add_argument("--truncate_num_val", type=int)
    parser.add_argument("--seed", type=int, default=35675)
    parser.add_argument("--n_cnn_1d", type=int, default=3)
    parser.add_argument("--n_cnn_2d", type=int, default=3)
    parser.add_argument("--n_cnn_channels_1d", type=int, default=10)
    parser.add_argument("--n_cnn_channels_2d", type=int, default=10)
    parser.add_argument("--kernel_size_1d", type=int, default=13)
    parser.add_argument("--kernel_size_2d", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr_init", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eta_min", type=float, default=1e-04)
    parser.add_argument("--n_epochs_per_log", type=int, default=5)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument(
        "--noise_to_signal_ratio", default=None, type=float
    )  # train and test with noise
    parser.add_argument("--use_noise_seed", choices=["true", "false"], default="false")
    parser.add_argument("--noise_seed_train",  type=int, default=None)
    parser.add_argument("--noise_seed_val",    type=int, default=None)

    parser.add_argument(
        "--init_mode",
        default="original",
        choices = [
            "original",
            "uniform-with-old-scale",
            "normal-with-old-scale",
            "he-normal",
        ],
    )

    # parser.add_argument("--save_all_model_weights", choices=bool_choices, default="true")

    # Weights and Biases setup
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--wandb_entity", type=wandb_entity_arg_type, default=None, help="The W&B entity")
    parser.add_argument(
        "--wandb_mode", choices=["offline", "online", "disabled"], default="offline"
    )

    # Misc. options
    parser.add_argument("--big_init", default=False, action="store_true")
    parser.add_argument("--small_init", action="store_false", dest="big_init")
    a = parser.parse_args()

    # Override unless use_targets="legacy"
    if a.train_targets == "smoothed":
        a.use_smoothed_targets = True
    elif a.train_targets == "original":
        a.use_smoothed_targets = False
    a.use_noise_seed = True if a.use_noise_seed=="true" else False

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
    # skip_wandb: bool = False,
    return_model: bool = False,
) -> None:
    """
    1. Load data
    2. Do necessary transformations
    3. Set up NN
    4. Train NN
    """
    if not os.path.isdir(args.model_weights_dir):
        os.mkdir(args.model_weights_dir)

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #########################################################
    # Load data
    data_dir_base = args.data_dir_base
    str_nu_list = (
        args.data_input_nus
    )  # nu in string form (to preserve decimals properly)
    nu_list = [float(str_nu) for str_nu in str_nu_list]
    N_freqs = len(nu_list)
    logging.info(f"data_dir_base: {data_dir_base}")
    logging.info(f"nu values received: {str_nu_list}")

    # os.path.join()
    train_files = [
        os.path.join(data_dir_base, f"train_measurements_nu_{nu}") for nu in str_nu_list
    ]
    val_files = [
        os.path.join(data_dir_base, f"val_measurements_nu_{nu}") for nu in str_nu_list
    ]
    test_files = [
        os.path.join(data_dir_base, f"test_measurements_nu_{nu}")
        for nu in str_nu_list
    ]

    logging.info(
        f"Attempting to load the following folders: {train_files} and {val_files}"
    )
    # Handle seeding for the noise
    if args.use_noise_seed:
        eff_noise_seed_train = args.noise_seed_train
        eff_noise_seed_val   = args.noise_seed_val
    else:
        eff_noise_seed_train = None
        eff_noise_seed_val   = None

    if args.noise_to_signal_ratio != 0:
        logging.info(f"Using seed as {eff_noise_seed_train} for the training set")
        logging.info(f"Using seed as {eff_noise_seed_val} for the val set")
    else:
        logging.info(f"Not adding noise!")


    # Training data dictionary
    logging.info(f"Loading training dataset")
    train_dd, train_meta_dd = load_multifreq_dataset(
        train_files,
        truncate_num=args.truncate_num,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_mh",
        nan_mode="skip",
        load_cart=False,
        noise_seed=eff_noise_seed_train,
    )
    train_dd_short = dict(kv_shrinker(k, v) for (k, v) in train_dd.items())
    logging.info(f"train_dd has entries with shapes: {train_dd_short}")

    # Evaluation data dictionary
    logging.info(f"Loading evaluation dataset")
    val_dd, val_meta_dd = load_multifreq_dataset(
        val_files,
        truncate_num=args.truncate_num_val,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_mh",
        nan_mode="skip",
        load_cart=False,
        noise_seed=eff_noise_seed_val,
    )
    val_dd_short = dict(kv_shrinker(k, v) for (k, v) in val_dd.items())
    logging.info(f"val_dd has entries with shapes: {val_dd_short}")

    if args.use_smoothed_targets:
        logging.info(f"Using smoothed targets for training and validation")
        train_q_polar = train_dd[Q_POLAR_LPF][:, -1, ...]
        val_q_polar  = val_dd[Q_POLAR_LPF][:, -1, ...]
    else:
        logging.info(f"Using original targets for training and validation")
        train_q_polar = train_dd[Q_POLAR]
        val_q_polar  = val_dd[Q_POLAR]

    # Also provide an alias for the original target regardless of training setting
    train_q_polar_orig = train_dd[Q_POLAR]
    val_q_polar_orig  = val_dd[Q_POLAR]

    train_d_mh = train_dd[D_MH]
    val_d_mh  = val_dd[D_MH]

    rho_vals = train_dd["rho_vals"]
    theta_vals = train_dd["theta_vals"]
    h_vals = train_dd["h_vals"]
    omega_vals = train_dd["omega_sf"]
    x_vals = train_dd["x_vals"]

    N_rho = rho_vals.shape[0]
    N_h = h_vals.shape[0]
    N_theta = theta_vals.shape[0]
    N_train = train_q_polar.shape[0]
    N_eval = val_q_polar.shape[0]

    # Next, run the "setup_single_dataset" function
    train_dset = setup_single_dataset(train_q_polar, train_d_mh, train_q_polar_orig)
    val_dset  = setup_single_dataset(val_q_polar, val_d_mh, val_q_polar_orig)
    logging.info(f"Finished loading data. N_train={N_train}, N_eval={N_eval}")

    ### Prepare for NN training ###
    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training on device: %s", device)
    # Send to the data loader
    train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size)
    val_dloader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size)

    extra_params = {}

    # Initialize the model
    model = MFISNet_Fused(
        N_h=N_h,
        N_rho=N_rho,
        N_freqs=N_freqs,
        c_1d=args.n_cnn_channels_1d,
        c_2d=args.n_cnn_channels_2d,
        w_1d=args.kernel_size_1d,
        w_2d=args.kernel_size_2d,
        N_cnn_1d=args.n_cnn_1d,
        N_cnn_2d=args.n_cnn_2d,
        big_init=args.big_init,
        init_mode=args.init_mode,
        **extra_params,
    )

    # Just for debugging regarding the initialization...
    simplehash = lambda x: hash(tuple(x.reshape(-1).tolist()))
    logging.info(f"Initialized parameter hashes: {[simplehash(p) for p in model.parameters()]}")

    ########################### Training procedure ###########################
    N_epochs = args.n_epochs

    # loss_module_0 = MSEModule(loss_idx=slice(None), final_output_idx=slice(None))
    loss_module_0 = MSEModule()
    loss_fn_dd = {
        "mse": loss_module_0.mse,
        "psnr": loss_module_0.psnr,
        "rel_l2": loss_module_0.relative_l2_error,
        "final_mse": loss_module_0.mse_against_final,
        "final_psnr": loss_module_0.psnr_against_final,
        "final_rel_l2": loss_module_0.relative_l2_error_against_final,
    }

    id_hash = hash_dict(vars(args))
    epoch_stagger = 0  # Just a single training step

    # Spin up the Weights and Biases environment
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
        def log_function(model_0, epoch_local):
            """
            Need to set:
            - loss_fn_dd
            """
            nonlocal train_dloader, val_dloader
            with torch.no_grad():
                epoch_eff = epoch_stagger + epoch_local

                # 1. Evaluate on train set
                train_loss_dd = evaluate_losses_on_dataloader(
                    model_0, train_dloader, loss_fn_dd, device
                )
                val_loss_dd = evaluate_losses_on_dataloader(
                    model_0, val_dloader, loss_fn_dd, device
                )

                weight_norm = torch.norm(
                    torch.cat([x.view(-1) for x in model_0.parameters()]), 2
                )

                # 3. Log to console and log file
                logging.info(
                    "Epoch %i/%i. Train MSE: %f, Train Rel L2: %f, Train PSNR: %f",
                    epoch_local,
                    N_epochs,
                    torch.mean(train_loss_dd["mse"]).item(),
                    torch.mean(train_loss_dd["rel_l2"]).item(),
                    torch.mean(train_loss_dd["psnr"]).item(),
                )
                logging.info(
                    "\t Val MSE: %f, Val Rel L2: %f, Val PSNR: %f",
                    torch.mean(val_loss_dd["mse"]).item(),
                    torch.mean(val_loss_dd["rel_l2"]).item(),
                    torch.mean(val_loss_dd["psnr"]).item(),
                )
                logging.info("\t Weight L2 norm: %f", weight_norm.item())

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

                train_dd = {
                    # Optimization info
                    "epoch": epoch_local + epoch_stagger,
                    "weight_norm": weight_norm.item(),
                    # Experiment info
                    "n_train": N_train,
                    "n_eval": N_eval,
                    "n_freqs": N_freqs,
                    "n_cnn_1d": args.n_cnn_1d,
                    "n_cnn_2d": args.n_cnn_2d,
                    "n_cnn_channels_1d": args.n_cnn_channels_1d,
                    "n_cnn_channels_2d": args.n_cnn_channels_2d,
                    "kernel_size_1d": args.kernel_size_1d,
                    "kernel_size_2d": args.kernel_size_2d,
                    "lr_init": args.lr_init,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "big_init": args.big_init,
                    "eta_min": args.eta_min,
                    "n_rho_vals": N_rho,
                    "n_theta_vals": N_theta,
                    "init_mode": args.init_mode,
                    "hash": id_hash,
                    # Extra data
                    "source_nu_list": nu_list,
                }
                for k, v in train_loss_dd.items():
                    train_dd["train_" + k] = torch.mean(v).item()
                for k, v in val_loss_dd.items():
                    train_dd["val_" + k] = torch.mean(v).item()
                write_result_to_file(args.train_results_fp, **train_dd)

                # Try to log results to W&B
                try:
                    wandbrun.log(train_dd)
                except ValueError:
                    logging.error("Error: wandb logging failed for %s" % wandbrun.id)

            fp_weights = os.path.join(
                args.model_weights_dir, f"epoch_{epoch_eff}.pickle"
            )
            torch.save(model_0.state_dict(), fp_weights)
            model_0 = model_0.to(device)

        for p in model.parameters():
            logging.info(
                f"Parameter with shape {p.shape} requires grad: {p.requires_grad}"
            )

        # Now train it!
        model = train(
            model=model,
            n_epochs=N_epochs,
            lr_init=args.lr_init,
            weight_decay=args.weight_decay,
            momentum=0.0,
            eta_min=args.eta_min,
            train_loader=train_dloader,
            device=device,
            n_epochs_per_log=args.n_epochs_per_log,
            log_function=log_function,
            loss_function=loss_module_0,
        )

    logging.info("Finished!")
    if return_model:
        return model
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
    main(a)
