# Train a single block of PDE Solver Refinement
# Expects to receive just one frequency

import logging
from typing import List
import argparse
from timeit import default_timer
import os
import numpy as np
import torch
import wandb
import os, psutil  # to fetch memory usage

from src.data.add_noise import add_noise_to_d

from src.data.data_io import (
    load_dir, load_multifreq_dataset, load_scobj_dir
)
# from src.models.MFISNet_Fused import MFISNet_Fused
from src.models.MFISNet_pde_solver_refinement_v1 import (
    MFISNet_pde_solver_refinement_v1,
    # TupleWrapper,
    # TupleLinearData,
)
from src.data.datasets import (
    TupleLinearData,
    setup_dataset_tuplelinear as setup_pde_solver_refinement_dataset
)

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
    parser.add_argument("--merge_middle_freq_channels", choices=["true", "false"])
    parser.add_argument("--polar_padding", choices=["true", "false"], default="true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr_init", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eta_min", type=float, default=1e-04)
    parser.add_argument("--n_epochs_per_log", type=int, default=5)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--forward_model_adjustment", type=float, default=1.0)
    parser.add_argument(
        "--noise_to_signal_ratio", default=None, type=float
    )  # train and test with noise
    parser.add_argument("--use_noise_seed", choices=["true", "false"], default="false")
    parser.add_argument("--noise_seed_train",  type=int, default=None)
    parser.add_argument("--noise_seed_val",    type=int, default=None)
    parser.add_argument("--noise_norm_mode", choices=["l2", "inf"], default="inf")

    parser.add_argument(
        "--use_cnns_2d",
        default = "both",
        choices = ["fynet", "update", "both"],
        help=(
            "Choose whether to use the 2d CNNs in the FYNet block, Update block, or both. "
            "The 'Update' block refers to the section where FYNet[d_kt] is used to update qhat_kt"
        ),
    )
    parser.add_argument(
        "--init_mode",
        default = "original",
        choices = [
            "original",
            "uniform-with-old-scale",
            "normal-with-old-scale",
            "he-normal",
        ],
    )
    parser.add_argument(
        "--embedding_mode",
        default = "none",
        choices = ["none", "fynet-forward"],
        help="Choose whether to embed q-hat as part of the input to the MFISNet-Fused block"
    )
    parser.add_argument("--n_emb_channels_out", type=int, default=0)
    parser.add_argument("--set_c1d_per_freq", default="true", choices=["true", "false"])
    parser.add_argument("--lr_decrease_factor", default=1.0, type=float)
    parser.add_argument("--freq_lvl", default=1, type=int, help="Which frequency level to use for lr decrease (start from 1, 2,...)")
    parser.add_argument(
        "--use_pred_d_mh", default="true", choices=["true", "false"],
        help="Whether to load predicted d_mh values from the predictions dataset"
    )

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

    # parsing boolean argument values
    a.set_c1d_per_freq = (a.set_c1d_per_freq == "true")
    a.use_pred_d_mh = (a.use_pred_d_mh == "true")
    a.use_noise_seed = (a.use_noise_seed == "true")

    # Override unless use_targets="legacy"
    if a.use_targets == "smoothed":
        a.use_smoothed_targets = True
    elif a.use_targets == "original":
        a.use_smoothed_targets = False

    return a

def main(
    args: argparse.Namespace,
    # Extra arguments for testing purposes
    return_model: bool = False,
) -> None:
    """
    1. Load data
    2. Set up NN
    3. Prepare the logging function
    4. Train NN
    """
    # Start by processing the arguments
    mmfc_bool = False if args.merge_middle_freq_channels.lower() == "false" else True
    polar_padding_bool = False if args.polar_padding.lower() == "false" else True
    logging.info(
        f"Received: merge_middle_freq_channels={mmfc_bool} and polar_pad={polar_padding_bool}"
    )
    args.merge_middle_freq_channels_bool = mmfc_bool
    args.polar_padding_bool = polar_padding_bool
    # Adjust the learning rate according to which frequency-level we are at
    lr_lvl_adjust = args.lr_decrease_factor ** (args.freq_lvl - 1)
    lr_init_eff = args.lr_init * lr_lvl_adjust
    eta_min_eff = args.eta_min * lr_lvl_adjust
    logging.info(
        f"Adjusting the learning rate by a factor of {lr_lvl_adjust} since freq_lvl={args.freq_lvl} "
        f"and lr_decreas_factor={args.lr_decrease_factor}. Using lr_init_eff={lr_init_eff} "
        f"(c.f.: lr_init={args.lr_init})."
    )

    if not os.path.isdir(args.model_weights_dir):
        os.mkdir(args.model_weights_dir)

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up noise seeds
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


    #########################################################
    # Load data
    ref_data_dir_base  = args.ref_data_dir_base
    pred_data_dir_base = args.pred_data_dir_base
    str_nu_list = (
        args.data_input_nus
    )  # nu in string form (to preserve decimals properly)
    nu_list = [float(str_nu) for str_nu in str_nu_list]
    N_freqs = len(nu_list)
    logging.info(f"ref_data_dir_base: {ref_data_dir_base}")
    logging.info(f"nu values received: {str_nu_list}")

    # os.path.join()
    ref_train_files = [
        os.path.join(ref_data_dir_base, f"train_measurements_nu_{nu}") for nu in str_nu_list
    ]
    ref_val_files = [
        os.path.join(ref_data_dir_base, f"val_measurements_nu_{nu}") for nu in str_nu_list
    ]

    pred_dir_basename = lambda nu: f"measurements_nu_{nu}" if args.use_pred_d_mh else f"scattering_objs"
    pred_train_files = [
        os.path.join(pred_data_dir_base, f"train_{pred_dir_basename(nu)}") for nu in str_nu_list
    ]
    pred_val_files = [
        os.path.join(pred_data_dir_base, f"val_{pred_dir_basename(nu)}") for nu in str_nu_list
    ]

    logging.info(f"Loading training dataset")
    logging.info(
        f"Attempting to load the following folders: {ref_train_files} and {pred_train_files}"
    )
    ref_train_dd, ref_train_meta_dd = load_multifreq_dataset(
        ref_train_files,
        truncate_num=args.truncate_num,
        # key_replacement=key_replacement,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_mh",
        nan_mode="skip",
        load_cart=False,
        noise_seed=eff_noise_seed_train,
        noise_norm_mode=args.noise_norm_mode,
    )
    pred_train_dd, pred_train_meta_dd = load_multifreq_dataset(
        pred_train_files,
        truncate_num=args.truncate_num,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        nan_mode="skip",
        load_cart=False,
        scobj_only_mode=not args.use_pred_d_mh,
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

    ref_train_dd_short = dict(kv_shrinker(k, v) for (k, v) in ref_train_dd.items())
    pred_train_dd_short = dict(kv_shrinker(k, v) for (k, v) in pred_train_dd.items())
    logging.info(f"ref_train_dd has entries with shapes: {ref_train_dd_short}")
    logging.info(f"pred_train_dd has entries with shapes: {pred_train_dd_short}")

    # Evaluation data dictionary
    logging.info(f"Loading evaluation dataset")
    logging.info(
        f"Attempting to load the following folders: {ref_val_files} and {pred_val_files}"
    )
    ref_eval_dd, ref_eval_meta_dd = load_multifreq_dataset(
        ref_val_files,
        truncate_num=args.truncate_num_val,
        # key_replacement=key_replacement,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_mh",
        nan_mode="skip",
        load_cart=False,
        noise_seed=eff_noise_seed_val,
        noise_norm_mode=args.noise_norm_mode,
    )
    pred_eval_dd, pred_eval_meta_dd = load_multifreq_dataset(
        pred_val_files,
        truncate_num=args.truncate_num_val,
        nan_mode="skip",
        load_cart=False,
        scobj_only_mode=not args.use_pred_d_mh,
    )

    ref_eval_dd_short = dict(kv_shrinker(k, v) for (k, v) in ref_eval_dd.items())
    pred_eval_dd_short = dict(kv_shrinker(k, v) for (k, v) in pred_eval_dd.items())
    logging.info(f"ref_eval_dd has entries with shapes: {ref_eval_dd_short}")
    logging.info(f"pred_eval_dd has entries with shapes: {pred_eval_dd_short}")

    # logging.info(f"Received a dictionary with keys: {list(train_dd.keys())}")
    ref_train_q_polar_orig = ref_train_dd[Q_POLAR]
    ref_eval_q_polar_orig  = ref_eval_dd[Q_POLAR]
    if args.use_smoothed_targets:
        logging.info(f"Using smoothed targets for training and validation")
        ref_train_q_polar = ref_train_dd[Q_POLAR_LPF][:, -1, ...]
        ref_eval_q_polar  = ref_eval_dd[Q_POLAR_LPF][:, -1, ...]
    else:
        logging.info(f"Using original targets for training and validation")
        ref_train_q_polar = ref_train_q_polar_orig
        ref_eval_q_polar  = ref_eval_q_polar_orig
    pred_train_q_polar = pred_train_dd[Q_POLAR]
    pred_eval_q_polar  = pred_eval_dd[Q_POLAR]

    ref_train_d_mh  = ref_train_dd[D_MH]
    ref_eval_d_mh   = ref_eval_dd[D_MH]
    if args.use_pred_d_mh:
        pred_train_d_mh = pred_train_dd[D_MH]
        pred_eval_d_mh  = pred_eval_dd[D_MH]
    else:
        pred_train_d_mh = None
        pred_eval_d_mh  = None

    rho_vals = ref_train_dd["rho_vals"]
    theta_vals = ref_train_dd["theta_vals"]
    h_vals = ref_train_dd["h_vals"]
    omega_vals = ref_train_dd["omega_sf"]
    x_vals = ref_train_dd["x_vals"]

    N_rho = rho_vals.shape[0]
    N_h = h_vals.shape[0]
    N_theta = theta_vals.shape[0]
    N_train = ref_train_q_polar.shape[0]
    N_eval = ref_eval_q_polar.shape[0]

    # Next... run the "setup_dataset" function
    train_valid_idcs = pred_train_dd["orig_idcs"]
    train_dset = setup_pde_solver_refinement_dataset(
        pred_train_q_polar,
        pred_train_d_mh,
        ref_train_d_mh[train_valid_idcs],
        ref_train_q_polar[train_valid_idcs],
        ref_q_polar_orig=ref_train_q_polar_orig[train_valid_idcs],
    )
    eval_valid_idcs  = pred_eval_dd["orig_idcs"]
    eval_dset = setup_pde_solver_refinement_dataset(
        pred_eval_q_polar,
        pred_eval_d_mh,
        ref_eval_d_mh[eval_valid_idcs],
        ref_eval_q_polar[eval_valid_idcs],
        ref_q_polar_orig=ref_eval_q_polar_orig[eval_valid_idcs],
    )
    logging.info(f"Finished loading data. N_train={N_train}, N_eval={N_eval}")

    ### Prepare for NN training ###
    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training on device: %s", device)
    # Send to the data loader
    train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size)
    eval_dloader = torch.utils.data.DataLoader(eval_dset, batch_size=args.batch_size)

    # Initialize the model
    model = MFISNet_pde_solver_refinement_v1(
        N_h=N_h,
        N_rho=N_rho,
        N_freqs=N_freqs, # compensate for the doubled inputs
        c_1d=args.n_cnn_channels_1d,
        c_2d=args.n_cnn_channels_2d,
        w_1d=args.kernel_size_1d,
        w_2d=args.kernel_size_2d,
        N_cnn_1d=args.n_cnn_1d,
        N_cnn_2d=args.n_cnn_2d,
        merge_middle_freq_channels=args.merge_middle_freq_channels_bool,
        big_init=args.big_init,
        polar_padding=args.polar_padding_bool,
        init_mode=args.init_mode,
        use_cnns_2d=args.use_cnns_2d,
        embedding_mode=args.embedding_mode,
        N_emb_channels_out=args.n_emb_channels_out,
        set_c1d_per_freq=args.set_c1d_per_freq,
        use_pred_d_mh=args.use_pred_d_mh,
    )

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
            nonlocal train_dloader, eval_dloader
            with torch.no_grad():
                epoch_eff = epoch_stagger + epoch_local

                # 1. Evaluate on train set
                train_loss_dd = evaluate_losses_on_dataloader(
                    model_0, train_dloader, loss_fn_dd, device
                )
                eval_loss_dd = evaluate_losses_on_dataloader(
                    model_0, eval_dloader, loss_fn_dd, device
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
                    torch.mean(eval_loss_dd["mse"]).item(),
                    torch.mean(eval_loss_dd["rel_l2"]).item(),
                    torch.mean(eval_loss_dd["psnr"]).item(),
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

                train_log_dd = {
                    # Optimization info
                    "epoch": epoch_local + epoch_stagger,
                    "weight_norm": weight_norm.item(),
                    # Experiment info
                    # "eval_on_val_set": not args.eval_on_test_set,
                    # "eval_on_test_set": args.eval_on_test_set,
                    "n_train": N_train,
                    "n_eval": N_eval,
                    "n_freqs": N_freqs,
                    "n_cnn_1d": args.n_cnn_1d,
                    "n_fynet_cnn_2d":  model_0.N_fynet_cnn_2d,
                    "n_update_cnn_2d": model_0.N_update_cnn_2d,
                    "n_cnn_channels_1d": args.n_cnn_channels_1d,
                    "n_cnn_channels_2d": args.n_cnn_channels_2d,
                    "merge_middle_freq_channels": args.merge_middle_freq_channels_bool,
                    "polar_padding": args.polar_padding_bool,
                    "kernel_size_1d": args.kernel_size_1d,
                    "kernel_size_2d": args.kernel_size_2d,
                    "lr_init": args.lr_init,
                    "eta_min": args.eta_min,
                    "lr_decrease_factor": args.lr_decrease_factor,
                    "freq_lvl": args.freq_lvl,
                    "lr_lvl_adjust": lr_lvl_adjust, # adjustment for which frequency level we are at
                    "lr_init_eff": lr_init_eff,
                    "eta_min_eff": eta_min_eff,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "big_init": args.big_init,
                    "eta_min": args.eta_min,
                    "n_rho_vals": N_rho,
                    "n_theta_vals": N_theta,
                    "n_h_vals": N_h,
                    "init_mode": args.init_mode,
                    "use_cnns_2d": args.use_cnns_2d,
                    "embedding_mode": args.embedding_mode,
                    "n_emb_channels_out": args.n_emb_channels_out,
                    "set_c1d_per_freq": args.set_c1d_per_freq,
                    "use_pred_d_mh": args.use_pred_d_mh,

                    "hash": id_hash,
                    # Extra data
                    "source_nu_list": nu_list,
                }
                for k, v in train_loss_dd.items():
                    train_log_dd["train_" + k] = torch.mean(v).item()
                for k, v in eval_loss_dd.items():
                    train_log_dd["eval_" + k] = torch.mean(v).item()
                write_result_to_file(args.train_results_fp, **train_log_dd)

                # Try to log results to W&B
                try:
                    wandbrun.log(train_log_dd)
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
