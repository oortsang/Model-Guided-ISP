from typing import Dict, Callable, List, Optional
import torch
import numpy as np
import os
import logging
from src.data.data_io import save_dict_to_hdf5, load_multifreq_dataset
from src.data.data_naming_constants import (
    Q_CART,
    Q_POLAR,
    THETA_VALS,
    RHO_VALS,
    X_VALS,
    SAMPLE_COMPLETION,
)
from src.data.data_transformations import (
    prep_polar_padder,
    polar_pad_and_apply,
    prep_conv_interp_2d,
    prepare_polar_to_cart,
)

FMT_STR = "scattering_objs_{}.h5"

from src.training_utils.train_loop import (
    move_helper,
    unpack_helper,
)

def make_preds_on_dataset(
    model: torch.nn.Module,
    dloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    shard_size: int,
    experiment_info: Dict[str, np.ndarray],
    format_str: str = FMT_STR,
    incl_x_rs: bool = False,
    # File-wise options
    use_orig_idcs: bool = False,
    to_cart_fn: Callable = None,
    evaluate_outputs: bool = False,
    polar_loss_fn_dict: bool = dict(),
    cart_loss_fn_dict: bool = dict(),
) -> Optional[tuple]:
    """
    This function makes predictions on the dataset in dloader, converts them
    to cartesian coordinates, and then saves the dataset of predictions to
    different files, using the output_dir, format_str and shard_size.
    The predictions are saved using save_dict_to_hdf5.

    Args:
        model (torch.nn.Module): Model with trained parameters
        dloader (torch.utils.data.DataLoader): Data loader for the dataset
        device (torch.device): Device where the model is located
        output_dir (str): Where to save the data
        shard_size (int): Number of predictions to save in each file
        conv_filter_to_x (np.ndarray): The convolution filter to convert from
            polar to cartesian coordinates in the x direction
        conv_filter_to_y (np.ndarray): The convolution filter to convert from
            polar to cartesian coordinates in the y direction
        experiment_info (Dict[str, np.ndarray]): A dictionary containing the
            metadata such as x_vals, rho_vals, theta_vals, etc.
        format_str (str, optional): The format string for the output files.
        incl_x_rs (bool): indicate whether to x as [x_mh, x_rs] when applicable
            by default, x_mh = x
            if incl_x_rs=True but there is no rs data, the behavior
            will effectively still be x_mh = x
        use_orig_idcs (bool): Decide whether to use the original underlying indices
            before NaNs were dropped
            Inserts NaNs if necessary.
        to_cart_fn (Callable): optionally provide a pre-prepared function that performs the
            coordinate transform from polar to cartesian coordinates
            Will be generated and run on the cuda device by default
        evaluate_outputs (bool): choose whether to evaluate the outputs while generating the predictions

    Returns:
        If output_dir is provided, None
        Otherwise, or a tuple of predictions:
            (
                preds_polar.numpy(),
                preds_cart.numpy(),
                polar_out_dd,
                cart_out_dd,
            )
    """
    N_x = experiment_info[X_VALS].shape[0]
    N_rho = experiment_info[RHO_VALS].shape[0]
    N_theta = experiment_info[THETA_VALS].shape[0]
    dloader_len = len(dloader.dataset)

    preds_cart  = torch.zeros((dloader_len, N_x, N_x), dtype=torch.float32)
    preds_polar = torch.zeros((dloader_len, N_theta, N_rho), dtype=torch.float32)
    to_cart_fn = to_cart_fn if to_cart_fn is not None else prepare_polar_to_cart(
        experiment_info[X_VALS], experiment_info[THETA_VALS], experiment_info[RHO_VALS],
        conv_op_device=device, return_conv_tensors=False,
    )

    model = model.to(device) # 2026-08-03: Adding this to ensure the model is in the correct spot
    model.eval()
    if evaluate_outputs:
        polar_out_dd = {
            k: torch.zeros(dloader_len, dtype=torch.float32)
            for k in polar_loss_fn_dict.keys()
        }
        cart_out_dd = {
            k: torch.zeros(dloader_len, dtype=torch.float32)
            for k in cart_loss_fn_dict.keys()
        }


    with torch.no_grad():
        # for i, (x, y, z) in enumerate(dloader):
        for i, data_batch in enumerate(dloader):
            data_batch = [move_helper(db_entry, device) for db_entry in data_batch]
            if not incl_x_rs:
                x_mh = data_batch[0]
            else:
                x_mh, x_rs = unpack_helper(data_batch[0])
            y_p,  y_c  = unpack_helper(data_batch[1])
            yf_p, yf_c = unpack_helper(data_batch[2])

            n_samples = y_p.shape[0]

            output_polar = model(x_mh)
            output_cart  = to_cart_fn(output_polar)
            # This happens in the case when we're making predictions with a
            # MFISNet-Refinement model, which makes a prediction for each input wave frequency
            if output_polar.dim() == 4:
                output_polar = output_polar[:, -1]
            nn = i * dloader.batch_size

            preds_polar[nn : nn + n_samples] = output_polar.to("cpu")
            preds_cart[nn : nn + n_samples]  = output_cart.to("cpu")

            if evaluate_outputs:
                for loss_key, loss_fn in polar_loss_fn_dict.items():
                    polar_out_dd[loss_key][nn : nn + n_samples] = loss_fn(
                        output_polar, y_p, yf_p
                    ).cpu()

                for loss_key, loss_fn in cart_loss_fn_dict.items():
                    cart_out_dd[loss_key][nn : nn + n_samples] = loss_fn(
                        output_cart, y_c, yf_c
                    ).cpu()


    # Now, convert the predictions to cartesian coordinates.
    tot_num_samples = dloader_len
    use_orig_idcs = use_orig_idcs \
        and "num_nan_samples" in experiment_info.keys() \
         and "orig_idcs" in experiment_info.keys()
    if use_orig_idcs:
        tot_num_samples = experiment_info["num_loaded_samples"]
        logging.info(f"Preparing to write out {tot_num_samples} samples ")
        logging.info(f"This includes restoring {experiment_info['num_nan_samples']} missing NaNs")
    else:
        logging.info(f"Caution: not restoring the missing samples to the outputs")

    # Prepare space for the predictions
    if output_dir is None:
        if evaluate_outputs:
            res = (
                preds_polar.numpy(),
                preds_cart.numpy(),
                polar_out_dd,
                cart_out_dd,
            )
        else:
            res = preds_polar.numpy(), preds_cart.numpy()
        return res

    sample_completion = experiment_info[SAMPLE_COMPLETION]
    preds_polar = preds_polar.numpy()
    preds_cart = preds_cart.numpy()

    # Restore NaNs using the orig_idcs flag
    if use_orig_idcs:
        orig_idcs = experiment_info["orig_idcs"]
        # Move the polar predictions first
        tmp_polar = np.full(
            (tot_num_samples, N_theta, N_rho),
            fill_value=torch.nan,
            dtype=preds_polar.dtype,
        )
        tmp_polar[orig_idcs, ...] = preds_polar
        del preds_polar
        preds_polar = tmp_polar

        # Move the cartesian predictions next
        tmp_cart = np.full(
            (tot_num_samples, N_x, N_x),
            fill_value=torch.nan,
            dtype=preds_cart.dtype,
        )
        tmp_cart[orig_idcs, ...] = preds_cart
        # dense_preds_cart = preds_cart
        del preds_cart
        preds_cart = tmp_cart

        # Move the sample completion markers
        # indicate that any NaN is associated with an incomplete sample
        tmp_completion = np.full(tot_num_samples, dtype=bool, fill_value=False)
        tmp_completion[orig_idcs] = True
        sample_completion = tmp_completion

    # Finally, save the predictions to different files
    # Write things out in the current order
    for i in range(0, tot_num_samples, shard_size):
        shard_cart = preds_cart[i : i + shard_size]
        shard_polar = preds_polar[i : i + shard_size]
        sample_completion_shard = sample_completion[i : i + shard_size]

        out_dd = {
            Q_CART: shard_cart,
            Q_POLAR: shard_polar,
            SAMPLE_COMPLETION: sample_completion_shard,
        }

        experiment_info.update(out_dd)
        out_fp = os.path.join(output_dir, format_str.format(i))
        save_dict_to_hdf5(
            experiment_info,
            out_fp,
        )

    if evaluate_outputs:
        res = (
            preds_polar,
            preds_cart,
            polar_out_dd,
            cart_out_dd,
        )
    else:
        res = preds_polar, preds_cart
    return res

def eval_losses_batched(
    preds: torch.Tensor,
    targets: torch.Tensor,
    targets_final: torch.Tensor,
    loss_fn_dd: dict,
    batch_size: int=0,
    return_mean_and_stdev: bool=True
):
    """Helper function to evaluate losses on torch tensor objects
    In case of excessive memory overhead, this function performs the calculations in batches
    if batch_size=0, everything will be done in a single batch
    """
    loss_val_dict = {
        k: torch.zeros(preds.shape[0], dtype=torch.float32)
        for k in loss_fn_dd.keys()
    }
    batch_size = batch_size if batch_size!=0 else preds.shape[0]
    for i in range(0, preds.shape[0], batch_size):
        idcs = slice(i, min(i+batch_size, preds.shape[0]))
        for k, loss_fn in loss_fn_dd.items():
            loss_val_dict[k][idcs] = loss_fn(
                preds[idcs], targets[idcs], targets_final[idcs],
            )
    mean_dict  = {k: loss_val.mean().item() for k, loss_val in loss_val_dict.items()}
    stdev_dict = {k: loss_val.std().item() for k, loss_val in loss_val_dict.items()}
    res = (
        (loss_val_dict, mean_dict, stdev_dict)
        if return_mean_and_stdev else
        loss_val_dict
    )
    return res

def make_preds_on_dataset_only_cartesian(
    model: torch.nn.Module,
    dloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    shard_size: int,
    experiment_info: Dict[str, np.ndarray],
    format_str: str = FMT_STR,
    # File-wise options
    use_orig_idcs: bool = False,
    evaluate_outputs: bool = False,
    cart_loss_fn_dict: bool = dict(),
    model_takes_x_mh: bool=False,
    model_outputs_tuple: bool=False,
) -> None:
    """Similar to make_preds_on_dataset except that for models that only output in cartesian
    (and when we don't necessarily care about polar performance)
    """
    N_x = experiment_info[X_VALS].shape[0]
    N_rho = experiment_info[RHO_VALS].shape[0]
    N_theta = experiment_info[THETA_VALS].shape[0]
    dloader_len = len(dloader.dataset)

    preds_cart  = torch.zeros((dloader_len, N_x, N_x), dtype=torch.float32)

    model = model.to(device)
    model.eval()
    if evaluate_outputs:
        cart_out_dd = {
            k: torch.zeros(dloader_len, dtype=torch.float32)
            for k in cart_loss_fn_dict.keys()
        }

    with torch.no_grad():
        # for i, (x, y, z) in enumerate(dloader):
        for i, data_batch in enumerate(dloader):
        # Modify to handle x as a list/tuple
            data_batch = [move_helper(db_entry, device) for db_entry in data_batch]
            x_rs, x_mh = unpack_helper(data_batch[0])
            y_c,  y_p  = unpack_helper(data_batch[1])
            yf_c, yf_p = unpack_helper(data_batch[2])
            n_samples = y_c.shape[0]

            if model_takes_x_mh:
                x = (x_rs, x_mh)
            else:
                x = x_rs
            model_output = model(x)
            if model_outputs_tuple:
                output_cart = model_output[0]
            else:
                output_cart = model_output

            nn = i * dloader.batch_size
            preds_cart[nn : nn + n_samples]  = output_cart.to("cpu")

            for loss_key, loss_fn in cart_loss_fn_dict.items():
                cart_out_dd[loss_key][nn : nn + n_samples] = loss_fn(
                    output_cart, y_c, yf_c
                ).cpu()

    # Next, handle saving to disk etc.
    tot_num_samples = dloader_len
    use_orig_idcs = use_orig_idcs \
        and "num_nan_samples" in experiment_info.keys() \
         and "orig_idcs" in experiment_info.keys()
    if use_orig_idcs:
        tot_num_samples = experiment_info["num_loaded_samples"]
        logging.info(f"Preparing to write out {tot_num_samples} samples ")
        logging.info(f"This includes restoring {experiment_info['num_nan_samples']} missing NaNs")
    else:
        logging.info(f"Caution: not restoring the missing samples to the outputs")


    if output_dir is None:
        res = (
            preds_cart.numpy(),
            cart_out_dd,
        )
        return res

    sample_completion = experiment_info[SAMPLE_COMPLETION]
    preds_cart = preds_cart.numpy()

    # Restore NaNs using the orig_idcs flag
    if use_orig_idcs:
        orig_idcs = experiment_info["orig_idcs"]
        # Move the cartesian predictions next
        tmp_cart = np.full(
            (tot_num_samples, N_x, N_x),
            fill_value=torch.nan,
            dtype=preds_cart.dtype,
        )
        tmp_cart[orig_idcs, ...] = preds_cart
        del preds_cart
        preds_cart = tmp_cart

        # Move the sample completion markers
        # indicate that any NaN is associated with an incomplete sample
        tmp_completion = np.full(tot_num_samples, dtype=bool, fill_value=False)
        tmp_completion[orig_idcs] = True
        sample_completion = tmp_completion

    # Finally, save the predictions to different files
    # Write things out in the current order
    for i in range(0, tot_num_samples, shard_size):
        shard_cart = preds_cart[i : i + shard_size]
        sample_completion_shard = sample_completion[i : i + shard_size]

        out_dd = {
            Q_CART: shard_cart,
            SAMPLE_COMPLETION: sample_completion_shard,
        }

        experiment_info.update(out_dd)
        out_fp = os.path.join(output_dir, format_str.format(i))
        save_dict_to_hdf5(
            experiment_info,
            out_fp,
        )
    res = (
        preds_cart,
        cart_out_dd,
    )
    return res
