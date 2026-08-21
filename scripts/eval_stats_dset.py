# Simple helper script to calculate the stats for
# an existing dataset
# Intended for RL-Modified or RL-Plain


import logging
from typing import List
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_io import (
    load_single_dir_slice,
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
    NU_SF,
    OMEGA_SF,
    X_VALS,
    SAMPLE_COMPLETION,
)
Q_CART_TMP = "q_cart_tmp"

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    bool_choices = ["true", "false"]

    parser.add_argument("--dset_names", type=str, nargs="+", default=["test"])
    parser.add_argument(
        "--ref_data_dir_base",
        type=str,
        help="For the reference dataset, indicate the directory containing all the "
        "measurement folders corresponding to the relevant frequencies and data subsets",
        default="dataset",
    )
    parser.add_argument(
        "--pred_data_dir_base",
        type=str,
        help="For the prediction dataset to be evaluated, indicate the base directory",
    )
    parser.add_argument(
        "--global_idx_start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--global_idx_end",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=100,
    )
    a = parser.parse_args()
    bool_args = []
    # Process the boolean arguments from strings
    for bool_arg in bool_args:
        str_val = getattr(a, bool_arg)
        setattr(a, bool_arg, str_val == "true")
    return a

def main(args):
    """Evaluates error metrics on the desired dataset(s)"""

    # Set up the loss functions to calculate
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

    # Book-keeping for final output
    last_eval_dict = {}
    key_max_num_chars = max(len(key) for key in cart_loss_fn_dd.keys())
    cart_dd_list = []

    ref_data_dir_base  = args.ref_data_dir_base
    pred_data_dir_base = args.pred_data_dir_base
    for dset in args.dset_names:
        # Load the datasets
        ref_scobj_dir  = os.path.join(ref_data_dir_base,  f"{dset}_scattering_objs")
        pred_scobj_dir = os.path.join(pred_data_dir_base, f"{dset}_scattering_objs")

        ref_scobj_dd = load_single_dir_slice(
            ref_scobj_dir,
            global_idx_start=args.global_idx_start,
            global_idx_end=args.global_idx_end,
            load_keys=[Q_CART, SAMPLE_COMPLETION],
        )
        ref_q_cart = ref_scobj_dd[Q_CART]
        pred_scobj_dd = load_single_dir_slice(
            pred_scobj_dir,
            global_idx_start=args.global_idx_start,
            global_idx_end=args.global_idx_end,
            load_keys=[Q_CART, SAMPLE_COMPLETION],
        )
        pred_q_cart = pred_scobj_dd[Q_CART]
        N_loaded = min(ref_scobj_dd[SAMPLE_COMPLETION].shape[0], pred_scobj_dd[SAMPLE_COMPLETION].shape[0])
        ebs = args.eval_batch_size

        print(f"ref_q_cart shape:  {ref_q_cart.shape}")
        print(f"pred_q_cart shape: {pred_q_cart.shape}")

        # rel_err_fn = lambda x, ref, **kwargs: np.linalg.norm(x-ref, **kwargs) / np.linalg.norm(ref, **kwargs)
        # manual_rel_l2 = rel_err_fn(pred_q_cart, ref_q_cart, axis=(-1,-2))
        # print(f"manual errors (mean: {manual_rel_l2.mean()})... {manual_rel_l2}")

        batched_eval = lambda f, *arg_list: np.concatenate(
            [
                f(*[torch.tensor(a[i: min(i+ebs, N_loaded)]) for a in arg_list]).cpu().numpy()
                # , print(f"i={i}, slice={i*ebs}:{min((i+1)*ebs, arg_list[0].shape[0])}")
                for i in range(0, N_loaded, ebs)
            ],
            axis=0,
        )
        cart_dd = {
            k: batched_eval(loss_f, pred_q_cart, ref_q_cart, ref_q_cart)
            for (k,loss_f) in cart_loss_fn_dd.items()
        }
        print(f"Rel_l2 vals: {cart_dd['cart_rel_l2']}")
        # print(f"cart_dd={cart_dd}")
        mean_dict   = {k: np.mean(v).item() for (k,v) in cart_dd.items()}
        stdev_dict  = {k: np.std(v).item() for (k,v) in cart_dd.items()}
        median_dict = {k: np.median(v).item() for (k,v) in cart_dd.items()}
        cart_dd_list.append(cart_dd)
        for key in cart_dd.keys():
            last_eval_dict[f"{dset}_{key}"]        = mean_dict[key]
            last_eval_dict[f"{dset}_{key}_stdev"]  = stdev_dict[key]
            last_eval_dict[f"{dset}_{key}_median"] = median_dict[key]

            key_ljust = (key + ":").ljust(key_max_num_chars+1)
            logging.info(f"{dset} {key_ljust} {mean_dict[key]:.5e}±{stdev_dict[key]:.3e}")

        logging.info(f"Finished evaluating dataset {dset}...")

    logging.info(f"Compilation of the metrics for different datasets...")
    for i, dset in enumerate(args.dset_names):
        cart_dd = cart_dd_list[i]
        for key in cart_dd.keys():
            key_ljust = (key + ":").ljust(key_max_num_chars+1)
            mean_val  = last_eval_dict[f"{dset}_{key}"]
            stdev_val = last_eval_dict[f"{dset}_{key}_stdev"]
            logging.info(
                f"{dset} {key_ljust} {mean_val:.5e}±{stdev_val:.3e}"
            )
    logging.info(f"Compressed representation:")
    print(f"Evaluation metrics on datasets {args.dset_names}:")
    for key in cart_loss_fn_dd.keys():
        key_ljust = (key + ":").ljust(key_max_num_chars+1)
        msg = f""
        for i in range(len(args.dset_names)):
            dset = args.dset_names[i]
            mean_val  = last_eval_dict[f"{dset}_{key}"]
            stdev_val = last_eval_dict[f"{dset}_{key}_stdev"]
            msg += f"{mean_val:.5e}±{stdev_val:.3e} "
        msg = msg[:-1]
        print_msg = f"Overall {key_ljust} {msg}"
        logging.info(print_msg)
        # print(print_msg)
        last_eval_dict["rel_l2_for_table"] = msg


if __name__ == "__main__":
    a = setup_args()
    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)

    logging.info(f"Received the following arguments: {a}")
    main(a)
