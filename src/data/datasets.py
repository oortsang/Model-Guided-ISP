# datasets.py
# Hold the dataset types and loading functions so we can more easily re-use them...
# Contents:
# LinearData
# TupleLinearData
# FullData
from typing import Tuple, List
import logging

import torch
import numpy as np


class LinearData(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, y_orig: torch.Tensor=None) -> None:
        self.X = X
        self.y = y
        self.y_orig = y_orig if y_orig is not None else y
        self.X_is_tuple = isinstance(X, tuple)
        self.y_is_tuple = isinstance(y, tuple)
        self.X_shape = X.shape if not self.X_is_tuple else tuple(xe.shape for xe in X)
        self.y_shape = y.shape if not self.y_is_tuple else tuple(ye.shape for ye in y)
        logging.info(
            "Initialized a LinearData instance with X shape: %s and y shape: %s",
            self.X_shape,
            self.y_shape,
        )
        self.n_samples = y.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.X_is_tuple:
            return list(xe[idx] for xe in self.X), self.y[idx], self.y_orig[idx]
        else:
            return self.X[idx], self.y[idx], self.y_orig[idx]

def setup_dataset_linear(
    q_polar: np.ndarray,
    wave_field_mh: np.ndarray,
    q_polar_orig: np.ndarray = None,
) -> LinearData:
    """
    Set up a single (multi-frequency) dataset (such as training/eval)
    Parameters:
        # data_dd (dict): dictionary received while loading the dataset
        q_polar (np.ndarray): stack of scattering objects
        wave_field_mh (np.ndarray): stack of wavefield patterns
    Return values:
        dset (LinearData): torch-ready data
    """
    inputs_dset = torch.view_as_real(torch.from_numpy(wave_field_mh))
    targets_dset = torch.from_numpy(q_polar)
    targets_orig_dset = torch.from_numpy(q_polar_orig) if q_polar_orig is not None else None
    dset = LinearData(
        inputs_dset,
        targets_dset,
        targets_orig_dset,
    )
    return dset


class TupleLinearData(torch.utils.data.Dataset):
    def __init__(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        y: torch.Tensor,
        y_orig: torch.Tensor = None
    ) -> None:
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.y_orig = y_orig if y_orig is not None else y
        logging.info(
            "Initialized a TupleData instance with X shape: %s and y shape: %s",
            (self.X1.shape, self.X2.shape,),
            self.y.shape,
        )
        self.n_samples = X2.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # (OOT 6/4/2024) Gives two copies of the target because the training
        # loop seems to expect a filtered and final version of the sample
        # (OOT 9/8/2024) Wrap the two inputs to accommodate inputs of different shapes
        # return TupleWrapper(self.X1[idx], self.X2[idx]), self.y[idx], self.y[idx]
        return [self.X1[idx], self.X2[idx]], self.y[idx], self.y_orig[idx]

def setup_dataset_tuplelinear(
        pred_q_polar: np.ndarray,
        pred_d_mh: np.ndarray,
        ref_d_mh: np.ndarray,
        ref_q_polar: np.ndarray,
        ref_q_polar_orig: np.ndarray = None,
        use_pred_d_mh: bool = True,
) -> TupleLinearData:
    """
    Set up a single (multi-frequency) dataset (such as training/eval)
    Parameters:
        pred_q_polar (np.ndarray): stack of scattering potential estimates
        pred_d_mh (np.ndarray): stack of wavefield patterns predicted from the previous scattering potential estimates
        ref_d_mh (np.ndarray): stack of true reference wavefield patterns (from the true scatterer)
        ref_q_polar (np.ndarray): stack of true (possibly smoothed) scattering objects
            These are used for training off of
        ref_q_polar_orig (np.ndarray): stack of original scattering objects
            These are used in the "final" logging values
    Return values:
        dset (LinearData): torch-ready data
    """
    # Treat them like different frequency channels
    # For now try inputting the reference and predicted difference
    if use_pred_d_mh and pred_d_mh is not None:
        stacked_d_mh = np.concatenate([pred_d_mh, pred_d_mh-ref_d_mh, ref_d_mh], axis=1)
    else:
        stacked_d_mh = ref_d_mh
    # stacked_d_mh  = ref_d_mh
    inputs_q_dset = torch.from_numpy(pred_q_polar)
    inputs_d_mh_dset = torch.view_as_real(torch.from_numpy(stacked_d_mh))
    targets_dset = torch.from_numpy(ref_q_polar)
    targets_orig_dset = torch.from_numpy(ref_q_polar_orig) if ref_q_polar_orig is not None else None
    dset = TupleLinearData(inputs_q_dset, inputs_d_mh_dset, targets_dset, targets_orig_dset)
    return dset

class FullData(torch.utils.data.Dataset):
    """A data loader that allows for X to come in (d_mh, d_rs) variants
    and also for y to come in (q_polar, q_cart) variants
    and same for y_orig.
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, y_orig: torch.Tensor=None) -> None:
        self.X = X
        self.y = y
        self.y_orig = y_orig if y_orig is not None else y
        self.X_is_tuple = isinstance(X, tuple)
        self.y_is_tuple = isinstance(y, tuple)
        self.y_orig_is_tuple = isinstance(self.y_orig, tuple)
        self.X_shape = X.shape if not self.X_is_tuple else tuple(xe.shape for xe in X)
        self.y_shape = y.shape if not self.y_is_tuple else tuple(ye.shape for ye in y)
        self.y_orig_shape = y_orig.shape if not self.y_orig_is_tuple else \
            tuple(ye.shape for ye in y_orig)
        logging.info(
            "Initialized a FullData instance with X shape: %s and y shape: %s",
            self.X_shape,
            self.y_shape,
        )
        self.n_samples = self.y_shape[0][0] if self.y_is_tuple else self.y_shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X_entry = list(xe[idx] for xe in self.X) if self.X_is_tuple else self.X[idx]
        y_entry = list(ye[idx] for ye in self.y) if self.y_is_tuple else self.y[idx]
        y_orig_entry = list(ye[idx] for ye in self.y_orig) if self.y_orig_is_tuple else \
            self.y_orig[idx]
        return X_entry, y_entry, y_orig_entry

def setup_dataset_full(
    q_polar: np.ndarray,
    q_cart: np.ndarray,
    wave_field_mh: np.ndarray,
    q_polar_orig: np.ndarray = None,
    q_cart_orig: np.ndarray = None,
    wave_field_rs: np.ndarray = None,
) -> FullData:
    """
    Set up a single (multi-frequency) dataset (such as training/eval)
    Parameters:
        q_polar (np.ndarray): stack of scattering objects in polar coordinates
        q_cart (np.ndarray): stack of scattering objects in cartesian coordinates
        wave_field_mh (np.ndarray): stack of wavefield patterns
        q_polar_orig (np.ndarray): un-smoothed version of q_polar
        q_cart_orig (np.ndarray): un-smoothed version of q_polar
        wave_field_mh (np.ndarray): stack of wavefield patterns
        wave_field_mh (np.ndarray): stack of wavefield patterns
    Return values:
        dset (LinearData): torch-ready data
    """
    if wave_field_rs is None:
        inputs_dset = torch.view_as_real(torch.from_numpy(wave_field_mh))
    else:
        inputs_dset = (
            torch.view_as_real(torch.from_numpy(wave_field_mh)),
            torch.view_as_real(torch.from_numpy(wave_field_rs)),
        )
    targets_dset = (torch.from_numpy(q_polar), torch.from_numpy(q_cart))

    qpo_part = torch.from_numpy(q_polar_orig) if q_polar_orig is not None else q_polar
    qco_part = torch.from_numpy(q_cart_orig) if q_cart_orig is not None else q_cart
    targets_orig_dset = (qpo_part, qco_part)

    dset = FullData(
        inputs_dset,
        targets_dset,
        targets_orig_dset,
    )
    return dset

def setup_preprocessed_predictions_dataset(
    pred_q_cart,
    ref_d_rs,
    ref_q_cart_target,
    ref_q_cart_orig=None,
    pred_gamma_cart=None,
) -> FullData:
    """Set up the dataset for the preprocessed predictions
    Parameters:
        pred_q_cart (np.ndarray): Predicted q in cartesian coordinates
        ref_d_rs (np.ndarray): reference measurements d_rs
        ref_q_cart_target (np.ndarray): reference training target for q_cart
            might be smoothed or original
        ref_q_cart_orig (np.ndarray): reference training value q_cart
            expected to be unsmoothed/original
        pred_gamma_cart (np.ndarray): gamma values corresponding to the predicted q
            these are the measurement misfit gradients
            These can be ommitted, but the code is not as thoroughly tested yet
            Expected shape: (N_samples, (1?), N_x, N_x)
            The second axis might be the frequency axis if left in
    Returns:
        dset (FullData): FullData object
            X / inputs: corresponds to (ref_d_rs, pred_q_cart, pred_gamma_cart)
            y / outputs: corresponds to (ref_q_cart_target, ref_q_cart_orig)
    """
    # complex numpy array to torch tensor
    cnp_to_torch = lambda x: torch.view_as_real(torch.from_numpy(x))
    torch_pred_q_cart = torch.from_numpy(pred_q_cart)

    # import pdb; pdb.set_trace()
    if pred_gamma_cart is not None:
        if pred_gamma_cart.ndim == 4:
            # in case the frequency axis was left in
            pred_gamma_cart = pred_gamma_cart[:, -1, ...]
        # -> (N_samples, 2, N_x, N_x)
        # torch_pred_gamma_cart = cnp_to_torch(pred_gamma_cart).permute(0,3,1,2)
        # -> (N_samples, 1, N_x, N_x)
        torch_pred_gamma_cart = torch.real(torch.from_numpy(pred_gamma_cart))
        inputs_q_space_dset = torch.concatenate([
            torch_pred_gamma_cart.unsqueeze(1),
            torch_pred_q_cart.unsqueeze(1),
        ], axis=1) # -> (N_samples, 3, N_x, N_x)
    else:
        inputs_q_space_dset = torch_pred_q_cart
    # logging.info(f"inputs_q_space_dset: {inputs_q_space_dset.shape}")


    torch_ref_d_rs = cnp_to_torch(ref_d_rs) # -> (N_samples, N_freqs, N_s, N_r, 2)
    # logging.info(f"torch_ref_d_rs: {torch_ref_d_rs.shape}")

    # double up on these so they hopefully act as the polar parts as well...
    torch_targets = torch.from_numpy(ref_q_cart_target)
    torch_targets_orig = (
        torch.from_numpy(ref_q_cart_orig)
        if ref_q_cart_orig is not None
        else torch_targets
    )

    dset = FullData(
        X=(inputs_q_space_dset, torch_ref_d_rs),
        y=torch_targets,
        y_orig=torch_targets_orig,
    )
    return dset
