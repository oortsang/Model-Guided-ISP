from typing import Callable, Dict, Optional
from timeit import default_timer
import logging
import torch
import numpy as np

def move_helper(obj, target):
    if hasattr(obj, "to"):
        out = obj.to(target)
    elif isinstance(obj, tuple):
        out = tuple(elem.to(target) for elem in obj)
    elif isinstance(obj, list):
        out = [elem.to(target) for elem in obj]
    else:
        raise ValueError(
            f"train_loop's move_helper passed a value that is neither iterable "
            f"nor has the `to` method"
        )
    return out
def unpack_helper(db):
    if isinstance(db, list):
        out1 = db[0]
        out2 = db[1]
    else:
        out1 = db
        out2 = None
    return out1, out2

def train(
    model: torch.nn.Module,
    n_epochs: int,
    lr_init: float,
    weight_decay: float,
    momentum: float,
    eta_min: float,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_epochs_per_log: int,
    log_function: Callable = None,
    loss_function: Callable = None,
    model_outputs_cart: bool = False,
    use_cart_output: bool = False,
    reweighted_polar: bool = False,
) -> torch.nn.Module:
    """A general-purpose training script using Adam as the optimizer and
    CosineAnnealingLR as the LR scheduler

    Args:
        model (torch.nn.Module): Model with trainable parameters
        n_epochs (int): Number of epochs to do training. No early stopping.
        lr_init (float): Initial learning rate
        weight_decay (float): Amount of L2 regularization applied
        momentum (float): Amount of momentum applied (NOT USED)
        eta_min (float): A parameter for the CoseineLR
        train_loader (torch.utils.data.DataLoader): _description_
        device (torch.device): _description_
        n_epochs_per_log (int): _description_
        log_function (Callable, optional): _description_. Defaults to None.

        use_cart_output (bool): whether to use the cartesian output of the neural network for the loss function

    Returns:
        torch.nn.Module: _description_
    """
    # Allow for simply doing the logging step
    # if n_epochs == 0:
    #     return model.to("cpu")

    model = model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr_init,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=eta_min
    )
    # logging.info("Beginning model training for %i epochs", n_epochs)

    loss_fn_name = (
        "cartesian"
        if use_cart_output
        else ("polar" if not reweighted_polar else "reweighted-polar")
    )
    logging.info("Using dataset: %s", train_loader.dataset)
    logging.info("Training using loss function/module: %s", loss_function)
    logging.info("Training model: %s", model)
    logging.info("Using LR: %f and min LR: %f", lr_init, eta_min)
    t1 = default_timer()

    model = model.to(device)
    for epoch in range(n_epochs):

        running_sum_squared_error_polar = 0
        running_sum_squared_error_cart = 0

        # klayer2dcnn_old_params = [p.clone() for p in model.filter_block.parameters()]

        for i, data_i in enumerate(train_loader):
            x = data_i[0]
            # logging.debug("train: x shape: %s", x.shape)
            y = data_i[1]
            y_final = data_i[2]
            # logging.debug("train: y shape: %s", y.shape)

            x = move_helper(x, device)
            y = move_helper(y, device)
            optimizer.zero_grad()

            model_output = model(x)
            if isinstance(model_output, tuple):
                # Choose between polar/cartesian
                pred_polar, pred_cart = model_output
                pred_for_loss = pred_cart if use_cart_output else pred_polar
            elif model_outputs_cart:
                pred_cart = model_output
                pred_for_loss = pred_cart
            else:
                pred_polar = model_output
                pred_for_loss = model_output

            loss_val = loss_function(pred_for_loss, y)
            loss_val.backward()
            optimizer.step()

            if log_function is None:
                with torch.no_grad():
                    running_sum_squared_error_polar += y.shape[0] * loss_function(
                        pred_polar, y
                    )

        scheduler.step()

        if epoch % n_epochs_per_log == 0:
            if log_function is not None:
                log_function(model, epoch)
            else:
                epoch_mse_polar = running_sum_squared_error_polar / len(
                    train_loader.dataset
                )

                logging.info(
                    f"Epoch {epoch} / {n_epochs}. Polar mse: {epoch_mse_polar:.3e}"
                )

    if n_epochs == 0:
        epoch = 0
        logging.info(f"n_epochs=0 so just perform the logging function")

    if log_function is not None:
        log_function(model, epoch)
    else:
        logging.info(f"No logging function provided")
    t2 = default_timer()
    logging.info("Optimization is complete in %f seconds", t2 - t1)
    return model.to("cpu")

def evaluate_losses_on_dataloader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn_dict: Dict,
    device: torch.device,
    incl_x_rs: bool = False,
    progress_bar: bool=False,
) -> Dict:
    model = model.to(device)
    model.eval()
    n_samples_tot = len(loader.dataset)
    n_batch = loader.batch_size

    # Set up an optional progress bar because the PDE-Solver version
    # is very slow and I am not sure how long to wait...
    if progress_bar:
        import tqdm
        wrapper = tqdm.tqdm
    else:
        wrapper = lambda x: x

    out_dd = {
        k: torch.zeros(n_samples_tot, dtype=torch.float32) for k in loss_fn_dict.keys()
    }

    for i, (x, y, y_final) in wrapper(enumerate(loader)):
        # Modify to handle x as a list/tuple
        if hasattr(x, "to"):
            x = x.to(device)
        elif isinstance(x, list):
            x = [x_elem.to(device) for x_elem in x]
        elif isinstance(x, tuple):
            x = tuple(x_elem.to(device) for x_elem in x)
        else:
            raise ValueError(f"train_loop passed a value x that is neither iterable nor has the `to` method")
        if incl_x_rs:
            x_mh, x_rs = x
        else:
            x_mh = x

        # x = x.to(device)
        y = y.to(device)
        y_final = y_final.to(device)
        n_samples = y.shape[0]

        preds = model(x_mh)

        for loss_key, loss_fn in loss_fn_dict.items():
            out_dd[loss_key][i * n_batch : (i * n_batch) + n_samples] = loss_fn(
                preds, y, y_final
            ).cpu()

    return out_dd


def evaluate_losses_on_dataloader_with_cartesian(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    polar_loss_fn_dict: Dict,
    cart_loss_fn_dict: Dict,
    polar_to_cart_fn: Callable,
    device: torch.device,
    model_outputs_cart: bool = False,
    cart_to_polar_fn: Callable = None,
    incl_x_rs: bool = False, # indicate whether x includes rs components
    model_takes_x_rs: bool = False, # whether to feed x_rs to the model...
    progress_bar: bool=False,
) -> Dict:
    """Helper function to evaluate loss functions with batched execution
    Modified from the original from Owen to also calculate statistics in cartesian coordinates

    Parameters:
        model (torch.nn.Module): pytorch (FYNet) model that returns polar and cartesian outputs
        loader (torch.utils.data.DataLoader): pytorch data loader for easy batched data access
        loss_fn_dict (Dict of functions of type (tensor, tensor, tensor, tensor) -> array):
            dictionary of functions to calculate the error on a batch of examples compared to the reference
            takes in (pred_polar, pred_cart, yp [y_polar], yc [y_cart]) and returns an array with floats for each batched example
        device (torch.device): cpu/gpu device to perform the evaluation on
        model_outputs_cart (bool): whether the model outputs a cartesian object
        cart_to_polar_fn (bool): if the model outputs in cartesian and we want to evaluate
            on polar loss functions, then it would be good to pass this function
        incl_x_rs (bool): whether the dataset includes x_rs, which usually corresponds to d_rs
            however, it can also be used to hold secondary data...
        model_takes_x_rs (bool): whether to feed in the x_rs component of the data
        progress_bar (bool): can get a tqdm progress bar for interactive environments...
    Returns:
        out_dd (dictionary of scalars)
    """

    model = model.to(device)
    model.eval()
    n_samples_tot = len(loader.dataset)
    n_batch = loader.batch_size

    # Set up an optional progress bar because the PDE-Solver version
    # is very slow and I am not sure how long to wait...
    if progress_bar:
        import tqdm
        wrapper = tqdm.tqdm
    else:
        wrapper = lambda x: x


    polar_out_dd = {
        k: torch.zeros(n_samples_tot, dtype=torch.float32) for k in polar_loss_fn_dict.keys()
    }
    cart_out_dd = {
        k: torch.zeros(n_samples_tot, dtype=torch.float32) for k in cart_loss_fn_dict.keys()
    }


    for i, data_batch in wrapper(enumerate(loader)):
        # Modify to handle x as a list/tuple
        data_batch = [move_helper(db_entry, device) for db_entry in data_batch]
        x_mh, x_rs = unpack_helper(data_batch[0])
        y_p,  y_c  = unpack_helper(data_batch[1])
        yf_p, yf_c = unpack_helper(data_batch[2])

        n_samples = y_p.shape[0]

        if model_takes_x_rs:
            x = (x_mh, x_rs)
        else:
            x = x_mh
        model_output = model(x)
        if model_outputs_cart:
            preds_cart  = model_output
            preds_polar = cart_to_polar_fn(preds_cart) if cart_to_polar_fn is not None else None
        else:
            preds_polar = model_output
            preds_cart  = polar_to_cart_fn(preds_polar)

        for loss_key, loss_fn in polar_loss_fn_dict.items():
            polar_out_dd[loss_key][i * n_batch : (i * n_batch) + n_samples] = loss_fn(
                preds_polar, y_p, yf_p
            ).cpu()

        for loss_key, loss_fn in cart_loss_fn_dict.items():
            cart_out_dd[loss_key][i * n_batch : (i * n_batch) + n_samples] = loss_fn(
                preds_cart, y_c, yf_c
            ).cpu()

    return polar_out_dd, cart_out_dd


def evaluate_losses_on_dataloader_only_cartesian(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    cart_loss_fn_dict: Dict,
    device: torch.device,
    model_takes_x_mh: bool = False, # whether to feed x_mh to the model...
    model_outputs_tuple: bool = False, # whether to extract the cart output from a tuple
    progress_bar: bool=False,
) -> Dict:
    """Helper function to evaluate loss functions with batched execution
    Specifically for cartesian-only models (I couldn't get the previous function to play nicely...)

    Parameters:
        model (torch.nn.Module): pytorch (FYNet) model that returns polar and cartesian outputs
        loader (torch.utils.data.DataLoader): pytorch data loader for easy batched data access
        loss_fn_dict (Dict of functions of type (tensor, tensor, tensor, tensor) -> array):
            dictionary of functions to calculate the error on a batch of examples compared to the reference
            takes in (pred_cart, yc [y_cart]) and returns an array with floats for each batched example
        device (torch.device): cpu/gpu device to perform the evaluation on
        model_outputs_cart (bool): whether the model outputs a cartesian object
        model_takes_x_mh (bool): whether to feed in the x_mh component of the data
        model_outputs_tuple (bool): whether the model gives multiple tuples (and therefore would
            need to be clipped to just the first entry of the tuple)
        progress_bar (bool): can get a tqdm progress bar for interactive environments...
    Returns:
        out_dd (dictionary of scalars)
    """

    model = model.to(device)
    model.eval()
    n_samples_tot = len(loader.dataset)
    n_batch = loader.batch_size

    # Set up an optional progress bar because the PDE-Solver version
    # is very slow and I am not sure how long to wait...
    if progress_bar:
        import tqdm
        wrapper = tqdm.tqdm
    else:
        wrapper = lambda x: x

    cart_out_dd = {
        k: torch.zeros(n_samples_tot, dtype=torch.float32) for k in cart_loss_fn_dict.keys()
    }

    for i, data_batch in wrapper(enumerate(loader)):
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
            preds_cart = model_output[0]
        else:
            preds_cart = model_output

        for loss_key, loss_fn in cart_loss_fn_dict.items():
            cart_out_dd[loss_key][i * n_batch : (i * n_batch) + n_samples] = loss_fn(
                preds_cart, y_c, yf_c
            ).cpu()

    return cart_out_dd
