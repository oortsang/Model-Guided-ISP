"""
This file has helper functions for loading and saving data. See the README in
the root directory for a description about the data saved in hdf5 format, naming
conventions used, and the directory structure of our dataset.

The directory choices were made to facilitate faster IO operations.
"""

from __future__ import annotations
import logging
import h5py
import numpy as np
from typing import Dict, List, Iterable
import os
import time
from src.data.data_naming_constants import (
    KEYS_FOR_TRAINING_SAMPLES_MEAS,
    KEYS_FOR_TRAINING_SAMPLES_ALL,
    KEYS_FOR_TRAINING_METADATA,
    KEYS_FOR_SCOBJ_SAMPLES,
    Q_CART,
    Q_POLAR,
    Q_CART_LPF,
    Q_POLAR_LPF,
    SAMPLE_COMPLETION,
    D_MH, D_RS,
    NU_SF, OMEGA_SF,
    FREQ_DEPENDENT_KEYS,
    TRUNCATABLE_KEYS,
    KEYS_FOR_EXPERIMENT_INFO_OUT,
    # Back-projection measurement differences
    GAMMA_CART,
    KEYS_FOR_BPMD_SAMPLES
)
from src.data.add_noise import add_noise_to_d
from src.data.layout import list_h5_files, find_files_index_range

import psutil  # to fetch memory usage
import time

# global constant for the number of times the IO functions will re-try accessing a given file
# MAX_RETRIES = 10
# RETRY_SLEEP_DUR = 30

MAX_RETRIES = 30
RETRY_SLEEP_DUR = 10

### Single-file HDF5 access functions ###

def load_hdf5_to_dict(fp_in, key_replacement: dict = None, retries: int=0) -> Dict:
    """Loads all the fields in a hdf5 file"""
    if retries >= MAX_RETRIES:
        raise IOError(f"(lhtd) Couldn't open load {fp_in} after {MAX_RETRIES} tries")
    key_replacement = key_replacement if key_replacement is not None else {}
    # key replacement function
    krfn = lambda key: key_replacement[key] if key in key_replacement.keys() else key
    # destination dictionary
    data_dict = {}
    try:
        with h5py.File(fp_in, "r") as hf:
            data_dict = {krfn(key): val[()] for (key, val) in hf.items()}
    except BlockingIOError:
        logging.warning(f"File {fp_in} is blocked on attempt {retries}")
        time.sleep(RETRY_SLEEP_DUR)
        return load_hdf5_to_dict(fp_in, key_replacement, retries + 1)
    return data_dict


def save_dict_to_hdf5(
    data_dict: Dict,
    fp_out: str,
    key_replacement: dict = None,
    retries: int=0,
) -> None:
    """Saves a dictionary as a hdf5 file at path fp_out"""
    key_replacement = key_replacement if key_replacement is not None else {}
    # key replacement function
    krfn = lambda key: key_replacement[key] if key in key_replacement.keys() else key

    if retries >= MAX_RETRIES:
        raise IOError(f"(sdth) Couldn't open file {fp_out} after {MAX_RETRIES} tries")
    try:
        with h5py.File(fp_out, "w") as hf:
            for key, val in data_dict.items():
                hf.create_dataset(krfn(key), data=val)
    except BlockingIOError:
        logging.warning(f"File {fp_out} is blocked; {retries} retries remaining")
        time.sleep(RETRY_SLEEP_DUR)
        save_dict_to_hdf5(data_dict, fp_out, key_replacement, retries+1)



### Helper functions to operate on Individual Fields ###

def save_field_to_hdf5(
    key: str, data: np.ndarray, fp_out: str, overwrite: bool = True, retries: int = 0
) -> None:
    """Saves an individual array to the specified field in a given hdf5 file
    Note that this operation may squash the old field
    """
    if not os.path.exists(fp_out):
        raise FileNotFoundError
    if retries >= MAX_RETRIES:
        raise IOError(f"(sfth) Couldn't open file {fp_out} after {MAX_RETRIES} tries")
    try:
        with h5py.File(fp_out, "a") as hf:
            # logging.debug(f"sfth df keys before: {hf.keys()}")
            if key in hf.keys() and not overwrite:
                raise KeyError(
                    f"Attempted to write to key {key} in {fp_out} "
                    "which already exists (and overwrite mode=False)"
                )
            elif key in hf.keys() and overwrite:
                # Need to handle the case where the dataset already exists...
                # See the update_field_in_hdf5 function for ideas maybe?
                # pass
                dset = hf.require_dataset(key, shape=data.shape, dtype=data.dtype)
                dset.write_direct(data)
            else:
                # Create new entry
                hf.create_dataset(key, data=data)
            # logging.debug(f"sfth df keys after:  {hf.keys()}")

    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(RETRY_SLEEP_DUR)
        save_field_to_hdf5(key, data, fp_out, retries + 1)

# Provide an alias for naming consistency but leave the original version
# intact to avoid breaking anything
def save_field_in_hdf5(
    key: str, data: np.ndarray, fp_out: str, overwrite: bool = True, retries: int = 0
) -> None:
    """Saves an individual array to the specified field in a given hdf5 file
    Note that this operation may squash the old field!

    Alias for save_field_to_hdf5 for better consistency
    with the other field-specific helper functions
    """
    save_field_to_hdf5(key, data, fp_out, overwrite=overwrite, retries=retries)


def load_field_in_hdf5(
    key: str, fp_out: str, idx_slice=slice(None), retries: int = 0
) -> np.ndarray:
    """Loads an individual field to the specified field in a given hdf5 file"""
    if not os.path.exists(fp_out):
        raise FileNotFoundError("Can't load field %s from %s" % (key, fp_out))
    if retries >= MAX_RETRIES:
        raise IOError(f"(load_field_in_hdf5) Couldn't open file after {MAX_RETRIES} tries")
    try:
        with h5py.File(fp_out, "r") as hf:
            data_loaded = hf[key][()]
            data = (
                data_loaded[idx_slice]
                if isinstance(data_loaded, np.ndarray)
                else data_loaded
            )
        return data

    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(RETRY_SLEEP_DUR)
        return load_field_in_hdf5(key, fp_out, idx_slice, retries + 1)
    # except:
    #     import pdb; pdb.set_trace()


def update_field_in_hdf5(
    key: str, data: np.ndarray, fp_out: str, idx_slice=slice(None), retries: int = 0
) -> None:
    """Saves an individual array to a slice in the specified field in a given hdf5 file
    Note that this operation may squash the old field
    """
    if not os.path.exists(fp_out):
        raise FileNotFoundError
    if retries >= MAX_RETRIES:
        raise IOError(f"(update_field_in_hdf5) Couldn't open file after {MAX_RETRIES} tries")

    try:
        with h5py.File(fp_out, "a") as hf:
            data_loaded = hf[key][()]
            data_loaded[idx_slice] = data
            dset = hf.require_dataset(
                key, shape=data_loaded.shape, dtype=data_loaded.dtype
            )
            dset.write_direct(data_loaded)
    except KeyError:
        # In case the field was not located, just make a new one...
        save_field_to_hdf5(key, data, fp_out, retries)

    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(RETRY_SLEEP_DUR)
        update_field_in_hdf5(key, data, fp_out, idx_slice, retries + 1)

def get_fields_in_hdf5(
    fp_in: str,
    retries: int=0,
    require_file_exist: bool=True,
) -> list:
    """Helper function to get a list of all the fields in a given hdf5 file

    Parameters:
        fp_in (str): file path of a the desired file to check
        retries (int): number of times to retry in case the file is busy
    Outputs:
        all_keys (list): a list of all the keys encountered
            if the file is not found but require_file_exist
            is set to False this will be an empty list
    """
    if not os.path.exists(fp_in):
        raise FileNotFoundError
    if retries >= MAX_RETRIES:
        raise IOError(f"(get_fields_in_hdf5) Couldn't open file after {MAX_RETRIES} tries")

    all_fields = []
    try:
        with h5py.File(fp_in, "r") as hf:
            all_fields = list(hf.keys())
    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(RETRY_SLEEP_DUR)
        return get_fields_in_hdf5(fp_in, retries + 1)
    except FileNotFoundError:
        if require_file_exist:
            raise # let the error propagate up if we require the file to exist
    return all_fields

### Helper functions for directory-wide HDF5 loading ###
# The file-naming/listing helpers (_file_start_idx, list_h5_files,
# get_file_start_index, find_files_index_range) now live in src.data.layout
# since they're about the on-disk file-naming convention rather than HDF5 I/O.

def _get_valid_idcs(arr: np.ndarray) -> np.ndarray:
    """Checks the array for NaNs.
    Parameters:
        arr (np.ndarray): expects shape (N_samples, ...)
    Returns:
        out (np.ndarray): 1-dim array with shape (N_samples,) indicating
            whether any entry for a given sample contains a nan
    """
    out = np.logical_not(np.any(np.isnan(arr), axis=tuple(range(1,arr.ndim))))
    return out


### Directory-wide HDF5 loading functions ###

def load_scobj_dir(
    scobj_dir_name: str,
    truncate_num: int | None = None,
    load_cart: bool = True,
    nan_mode: str = None,
    bpmd_mode: bool = False,
) -> Tuple[Dict, Dict]:
    """Helper function to load a directory of just scattering objects
    Note: if nan_mode is chosen as 'skip' then only the initial truncate_num
    will be loaded *before* the skip is applied.
    That is, there may be a shorter output than what is available

    Can also set bpmd_mode to load just the fields relevant to back-projected
    measurement difference (bpmd) files (i.e., q_cart, gamma_cart)
    """
    nan_mode = nan_mode.lower() if nan_mode is not None else "skip"
    # 1. Determine the relevant files to be loaded
    file_list_scobj = list_h5_files(scobj_dir_name)
    logging.info(f"About to load file_list_scobj={file_list_scobj}")

    n_files = len(file_list_scobj)
    if bpmd_mode:
        keys_to_append = [*KEYS_FOR_BPMD_SAMPLES] # includes sample_completion
    else:
        keys_to_append = [*KEYS_FOR_SCOBJ_SAMPLES, SAMPLE_COMPLETION]
    keys_to_ignore = set()
    if not load_cart:
        keys_to_ignore = {*keys_to_ignore, Q_CART, Q_CART_LPF}
        keys_to_append = [key for key in keys_to_append if key not in keys_to_ignore]

    # 2. Load the first file
    scobj_fp_0 = file_list_scobj[0]
    out_dd = load_hdf5_to_dict(scobj_fp_0)
    for key in keys_to_ignore:
        if key in out_dd.keys():
            del out_dd[key]

    # If we already have enough samples, the loading process is already finished
    # So, exit early
    n_samples_0 = out_dd[SAMPLE_COMPLETION].shape[0]
    # n_samples_0 = out_dd[Q_POLAR].shape[0]
    samples_loaded = n_samples_0

    def get_nan_entries(q_arr):
        nan_idcs = np.any(np.isnan(q_arr), axis=tuple(range(1,q_arr.ndim)))
        return nan_idcs
    def enough_samples_loaded(samples_loaded):
        nonlocal truncate_num
        return False if truncate_num is None else (samples_loaded >= truncate_num)

    # 3. Extend the keys_to_append values
    if truncate_num is not None:
        for key in keys_to_append:
            if key in out_dd.keys():
                tmp_val = out_dd[key]
                out_dd[key] = np.zeros([truncate_num, *tmp_val.shape[1:]], dtype=tmp_val.dtype)
                out_dd[key][:tmp_val.shape[0]] = tmp_val[:min(truncate_num, tmp_val.shape[0])]

    # 4. Load each file
    for i in range(1, n_files):
        if enough_samples_loaded(samples_loaded):
            break
        scobj_fp_i = file_list_scobj[i]
        dd_new = load_hdf5_to_dict(scobj_fp_i)
        # n_samples_i = dd_new[Q_POLAR].shape[0]
        n_samples_i = dd_new[SAMPLE_COMPLETION].shape[0]
        src_idcs = slice(
            0,
            min(n_samples_i, truncate_num - samples_loaded)
            if truncate_num is not None else None
        )
        dst_idcs = slice(
            samples_loaded,
            min(samples_loaded + n_samples_i, truncate_num)
            if truncate_num is not None else None
        )
        if truncate_num is not None:
            for key in keys_to_append:
                out_dd[key][dst_idcs] = dd_new[key][src_idcs]
        else:
            # If truncate_num is None, then the out_dd[key] object
            # has not been set up yet
            for key in keys_to_append:
                out_dd[key] = np.concatenate(
                    [out_dd[key], dd_new[key][src_idcs]],
                    axis=0,
                )

        samples_loaded = min(samples_loaded + n_samples_i, truncate_num) \
            if truncate_num is not None else samples_loaded + n_samples_i

    logging.info(f"out_dd keys: {out_dd.keys()}")
    out_dd_shapes = {k: v.shape for k, v in out_dd.items()}
    logging.info(f"out_dd: {out_dd_shapes}")

    # Apply the nan adjustments at the end
    # if Q_POLAR in out_dd.keys():
    #     nan_idcs = get_nan_entries(out_dd[Q_POLAR])
    nan_idcs = (
        get_nan_entries(out_dd[Q_POLAR])
        if Q_POLAR in out_dd.keys() else
        get_nan_entries(out_dd[Q_CART])
        if Q_CART in out_dd.keys() else
        get_nan_entries(out_dd[GAMMA_CART])
        if GAMMA_CART in out_dd.keys() else
        np.zeros(*dd_new[SAMPLE_COMPLETION].shape, dtype=bool)
        # None
    )
    if nan_idcs is not None:
        valid_idcs = np.logical_not(nan_idcs)
        num_nan_samples = np.sum(nan_idcs)
        if nan_mode == "zero":
            for key in keys_to_append:
                out_dd[key][nan_idcs] = 0
        elif nan_mode == "keep":
            pass
        elif nan_mode == "skip":
            for key in keys_to_append:
                out_dd[key] = out_dd[key][valid_idcs]
        else:
            raise ValueError(
                f"expected nan_mode to be one of {'zero', 'keep', 'skip'}, "
                f"but received '{nan_mode}' instead"
            )
    else:
        logging.debug(f"Skipping nan check since no Q_CART or Q_POLAR or GAMMA_CART were found")
        valid_idcs = np.ones(samples_loaded, dtype=bool)
        num_nan_samples = 0

    # Add in auxiliary info
    aux_dd = {
        "num_loaded_samples": samples_loaded,
        "num_valid_samples":  samples_loaded - num_nan_samples,
        "num_nan_samples":    num_nan_samples,
        "orig_idcs":          valid_idcs,
    }
    out_dd = {**aux_dd, **out_dd}

    logging.info(f"num_loaded_samples: {out_dd['num_loaded_samples']}")
    logging.info(f"num_valid_samples: {out_dd['num_valid_samples']}")
    logging.info(f"num_nan_samples: {out_dd['num_nan_samples']}")

    return out_dd, aux_dd

def load_dir(
    meas_dir_name: str,
    scobj_dir_name: str,
    truncate_num: int | None = None,
    load_cart_and_rs: bool = False,
) -> Dict[str, np.ndarray]:
    """Loads the data from a directory of hdf5 files that we assume to have the same fields, as specified by the generate_measurement_files.py script.

    Args:
        meas_dir_name (str): Directory containing all of the measurement files
        scobj_dir_name (str): Directory containing all of the scattering object files
        truncate_num (int | None, optional): How many samples to load. If set to None, all samples are loaded. Defaults to None.
        load_cart_and_rs (bool, optional): whether to load q_cart, q_cart_lpf and d_rs

    Returns:
        Dict[str, np.ndarray]: contains all keys in the union of KEYS_FOR_TRAINING_SAMPLES and KEYS_FOR_TRAINING_METADATA
    """
    # 1. Determine the relevant files to be loaded

    # Get the list of measurement files
    file_list = list_h5_files(meas_dir_name)

    # Get the list of scattering object files
    file_list_scobj = list_h5_files(scobj_dir_name)
    logging.info(f"Preparing to load file_list_scobj={file_list_scobj}")

    assert len(file_list) == len(
        file_list_scobj
    ), "Can't load directories with different lengths: %i vs %i, %s vs %s" % (
        len(file_list),
        len(file_list_scobj),
        meas_dir_name,
        scobj_dir_name,
    )
    logging.info(f"(load_dir) preparing to load {len(file_list)} files...")

    n_files = len(file_list)

    # 2. Select which fields should be loaded
    # Keys to be appended; also a subset for extracting from the measurement files
    keys_to_append = [*KEYS_FOR_TRAINING_SAMPLES_ALL]  # copy to avoid over-writing it
    keys_to_append_from_meas = [*KEYS_FOR_TRAINING_SAMPLES_MEAS]

    if load_cart_and_rs:
        keys_to_append += [D_RS, Q_CART, Q_CART_LPF]  # include q_cart_lpf
        keys_to_append_from_meas += [D_RS, Q_CART_LPF]
        keys_to_ignore = []
    else:
        keys_to_ignore = [D_RS, Q_CART, Q_CART_LPF]

    # 3. Load the first file to determine the appropriate shapes for the fields
    fp_0_meas = file_list[0]
    out_dd = load_hdf5_to_dict(fp_0_meas)
    if not load_cart_and_rs:
        for kti in keys_to_ignore:
            if kti in out_dd.keys():
                del out_dd[kti]
    # n_samples_0 = out_dd[D_MH].shape[0]
    n_samples_0 = out_dd[Q_POLAR].shape[0]

    fp_0_scobj = file_list_scobj[0]
    out_dd[Q_POLAR] = load_field_in_hdf5(Q_POLAR, fp_0_scobj)
    logging.info(f"(load_dir) first file has been prepared")

    # Set truncate_num to infinity if not specified.
    truncate_num = np.inf if truncate_num is None else truncate_num

    # If we already have enough samples, the loading process is already finished
    # So, exit early
    if n_samples_0 > truncate_num:
        for k in keys_to_append:
            if k in out_dd:
                out_dd[k] = out_dd[k][:truncate_num]
        return out_dd

    # 4. Append the relevant fields for the rest of the files in the directory
    for i in range(1, n_files):
        break_bool = False
        fp_meas = file_list[i]
        fp_scobj = file_list_scobj[i]
        # Temporary dictionary that just holds the fields to be extended
        dd_new = {
            key: load_field_in_hdf5(key, fp_meas) for key in keys_to_append_from_meas
            if key in out_dd.keys()
        }
        dd_new[Q_POLAR] = load_field_in_hdf5(Q_POLAR, fp_scobj)
        if load_cart_and_rs:
            dd_new[Q_CART] = load_field_in_hdf5(Q_CART, fp_scobj)

        new_n_samples = dd_new[SAMPLE_COMPLETION].shape[0]  # number of new samples

        # Check whether to truncate here
        if out_dd[SAMPLE_COMPLETION].shape[0] + new_n_samples > truncate_num:
            # In the case that we have to truncate, we first compute
            # how many samples to keep, and then concatenate the contents
            # of dd_new into out_dd
            n_samples_to_keep = truncate_num - out_dd[SAMPLE_COMPLETION].shape[0]
            dd_new = {key: dd_new[key][:n_samples_to_keep] for key in keys_to_append}
            break_bool = True

        for key in keys_to_append:
            if key in out_dd.keys():
                out_dd[key] = np.concatenate(
                    [out_dd[key], dd_new[key]]
                )  # concatenates along axis 0
        if break_bool:
            # Break out of the for loop if we have to truncate
            break

    # If we choose not to load q_cart or d_rs, then we delete the fields entirely to avoid confusion
    if not load_cart_and_rs:
        if Q_CART_LPF in out_dd.keys():
            del out_dd[Q_CART_LPF]
        if Q_CART in out_dd.keys():
            del out_dd[Q_CART]
        if D_RS in out_dd.keys():
            del out_dd[D_RS]

    logging.info(f"(load_dir) Finished loading the scobj dir ({scobj_dir_name}) and meas dir ({meas_dir_name})")
    return out_dd

# Helper function to process the loaded directory
def load_multifreq_dataset(
    freq_dir_list: List[str],
    truncate_num: int = None,
    key_replacement: dict = None,
    noise_to_sig_ratio: float = None,
    add_noise_to: str = None,
    load_cart: bool = False,
    nan_mode: str = None,
    scobj_only_mode: bool = False,
    noise_seed: int = None,
    noise_norm_mode: str = None,
) -> Tuple[dict, dict]:
    """
    Helper function to load datasets containing multiple frequencies
    Allows for replacing keys to ensure the right naming convention

    Parameters:
        freq_dir_list (List of str): give the different directories corresponding to the different frequencies
        truncate_num (int): the number of samples to be loaded
        key_replacement (dict): key mapping in case old field names need to be overriden
            Note: should not be needed but is left as a courtesy to outdated code
        noise_to_sig_ratio (float): level of noise relative to the signal
        add_noise_to (str): specify whether to add noise to "d_mh" or "d_rs".
            Only adds to one of these because the noise patterns will be different on each
        nan_mode (str): choose between "zero" out nan entries or "skip" entire samples containing a nan
        scobj_only_mode (bool): skip any references to anything not in the scattering_objs files
        noise_seed (int): optional parameter that requests the noise to be produced from
            a specific seed; sample i is generated using (noise_seed+i) for reproducibility

    Outputs:
        dd (dict): dictionary representing the dataset
    """
    N_freqs = len(freq_dir_list)
    dd_list = []
    for dir_name in freq_dir_list:
        logging.info(f"Loading dataset from {dir_name}")
        if scobj_only_mode:
            dd_new, _ = load_scobj_dir(
                dir_name,  # pass as meas  dir
                truncate_num=truncate_num,
                load_cart=load_cart,
            )
        else:
            dd_new = load_dir(
                dir_name,  # pass as scobj dir
                dir_name,  # pass as meas  dir
                truncate_num=truncate_num,
                load_cart_and_rs=load_cart,
            )
        dd_list.append(dd_new)
    logging.info(f"Finished loading the dataset")

    # Set up the dictionary for fixed values and fields that will get multiple frequencies
    dd_all = dict()
    simple_fdk_list = []
    present_fdk_list = []
    for key, val in dd_list[0].items():
        if key not in FREQ_DEPENDENT_KEYS:  # or key in [OMEGA_SF, NU_SF]:
            # If the key is not frequency-dependent we only need this one value
            dd_all[key] = val
        elif key in {OMEGA_SF, NU_SF}:
            # For omega_sf and nu_sf, we just need a scalar per frequency
            dd_all[key] = np.zeros(N_freqs)
            simple_fdk_list.append(key)
        else:
            # All other keys with frequency dependence will need extra space
            # e.g., d_mh needs an index per frequency
            # Assume we always just put the new frequency index in dim 1
            curr_shape = dd_list[0][key].shape
            logging.info(f"Found fdk {key} whose entry has shape {curr_shape}")
            new_shape = tuple([*curr_shape[:1], N_freqs, *curr_shape[1:]])
            dd_all[key] = np.zeros(new_shape, dtype=dd_list[0][key].dtype)
            present_fdk_list.append(key)

    # For each frequency-dependent key in the loaded file, fetch the data from other frequencies
    for fdk in present_fdk_list:
        # Fetch the relevant slice for each frequency (and for each frequency-dependent key)
        for fi in range(N_freqs):
            logging.info(
                f"(key {fdk}) Loading value of shape {dd_list[fi][fdk].shape} into a slice"
                f" of {dd_all[fdk].shape}"
            )
            dd_all[fdk][:, fi] = dd_list[fi][fdk]
    for simple_fdk in simple_fdk_list:
        # Repeat for scalars (i.e., nu_sf and omega_sf)
        for fi in range(N_freqs):
            dd_all[simple_fdk][fi] = dd_list[fi][simple_fdk].item()

    if scobj_only_mode:
        aux_dd = {
            "num_nan_samples": 0,
            "num_loaded_samples": dd_all[Q_POLAR].shape[0],
            "num_valid_samples": dd_all[Q_POLAR].shape[0],
            "orig_idcs": np.full(dd_all[Q_POLAR].shape[0], True, dtype=bool),
            # "orig_idcs": np.arange(dd_all[D_MH].shape[0]),
        }
    else:
        # Decide how to deal with with NaNs; determine using d_mh
        nan_mode = nan_mode.lower() if nan_mode is not None else "skip"
        num_nan_samples = np.sum(np.any(np.isnan(dd_all[D_MH]), axis=(-3,-2,-1)))
        aux_dd = {
            "num_nan_samples": num_nan_samples,
            "num_loaded_samples": dd_all[D_MH].shape[0],
            "num_valid_samples": dd_all[D_MH].shape[0] - num_nan_samples,
            "orig_idcs": np.full(dd_all[D_MH].shape[0], True, dtype=bool),
        }
        our_keys_for_training_samples_all = [*KEYS_FOR_TRAINING_SAMPLES_ALL, Q_CART, Q_CART_LPF]
        def get_valid_idcs(arr):
            return np.logical_not(np.any(np.isnan(arr), axis=tuple(range(1,arr.ndim))))

        # Make sure everything is processed together re nans
        logging.info(f"Computing the auxiliary index info")
        keep_idcs = np.logical_and.reduce([
            get_valid_idcs(dd_all[key])
            for key in TRUNCATABLE_KEYS
            if  key in dd_all
        ])
        nan_idcs = np.logical_not(keep_idcs)
        aux_dd["orig_idcs"] = keep_idcs
        if nan_mode == "keep":
            # Keep the NaNs
            pass
        elif nan_mode == "zero":
            # Zero out for everything
            for key in TRUNCATABLE_KEYS:
                if key in dd_all.keys() and key in our_keys_for_training_samples_all:
                    dd_all[key][nan_idcs] = 0
        elif nan_mode == "skip":
            for key in TRUNCATABLE_KEYS:
                if key in dd_all.keys() and key in our_keys_for_training_samples_all:
                    dd_all[key] = dd_all[key][keep_idcs]
            # Original indices of the included samples
            logging.info(f"orig_idcs: {len(keep_idcs)} entries")
        logging.info(f"num_loaded_samples: {aux_dd['num_loaded_samples']}")
        logging.info(f"num_valid_samples: {aux_dd['num_valid_samples']}")
        logging.info(f"num_nan_samples: {aux_dd['num_nan_samples']}")
        logging.info(f"dd_all[d_mh]: {dd_all[D_MH].shape}")

        # Add noise if applicable
        if noise_to_sig_ratio is not None:
            add_noise_to = add_noise_to.lower() if add_noise_to is not None else "d_mh"
            if noise_to_sig_ratio == 0:
                # If the noise-to-signal ratio is identically zero, skip this step
                pass
            elif add_noise_to == "d_mh":
                dd_all[D_MH] = add_noise_to_d(
                    dd_all[D_MH],
                    noise_to_sig_ratio,
                    noise_seed,
                    norm_mode=noise_norm_mode,
                )
            elif add_noise_to == "d_rs":
                dd_all[D_RS] = add_noise_to_d(
                    dd_all[D_RS],
                    noise_to_sig_ratio,
                    noise_seed,
                    norm_mode=noise_norm_mode,
                )
            elif add_noise_to == "none":
                pass
            else:
                raise ValueError(
                    f"Did not recognize {add_noise_to} as a valid field to add noise to."
                    f" Please enter either 'd_mh' or 'd_rs'."
                )
            logging.info(
                f"Applied noise at {noise_to_sig_ratio:.2f} to field '{add_noise_to}'"
            )

    # Apply key replacement
    key_replacement = key_replacement if key_replacement is not None else {}

    # Replace one key at a time to reduce memory overhead...hopefully...
    for old_key in key_replacement.keys():
        if old_key not in dd_all.keys():
            continue  # skip if key is invalid
        new_key = key_replacement[key]
        if new_key == old_key:
            continue  # skip if no move is required
        dd_all[new_key] = dd_all[old_key]
        del dd_all[old_key]

    # The metadata should be contained in dd_all
    # But hopefully this helps compatibility with other
    metadata_dd = {
        key: dd_all[key]
        for key in KEYS_FOR_EXPERIMENT_INFO_OUT
        if key in dd_all.keys()
    }
    dd_all   = {**aux_dd, **dd_all}
    metadata_dd = {**aux_dd, **metadata_dd}

    return dd_all, metadata_dd

##### Loading slices of a dataset, not necessarily file-by-file #####

def load_single_dir_slice(
    dir_name: str,
    global_idx_start: int = 0,
    global_idx_end: int = None,
    load_keys: Iterable=None,
    ignore_keys: Iterable=None,
    sample_keys: Iterable=None,
) -> dict:
    """Load all the files in a given directory from the desired slice
    Compared to previous implementations, this is meant to be fairly
    agnostic to the field names, though it still offers control by letting
    the user to specify which keys to ignore and which need to be truncated/concatenated

    This is intended for loading single directories.
    Note: it may be worth considering refactoring some of the code above to use this function,
    since it is much simpler and more generic

    Behavior:
        1. Non-concatable keys will be taken from the first valid file
        2. If fields is unspecified, all fields will be loaded, except those in ignore_keys
        3. Fields that are concatable will be concatenated for the return value

    Parameters:
        dir_name (str): directory whose files this function will look through and load
        global_idx_start (int): starting sample index to load, inclusive
        global_idx_end (int): last sample index to load, exclusive to match python conventions
        load_keys (Iterable): optionally can specify which fields to load; defaults to all
        ignore_keys (Iterable): alternately, can specify the fields not to load
        sample_keys(Iterable): the keys corresponding to samples
            the values should be concatenated/truncated as needed
            Note: this always concatenates/truncates along axis 0
    Outputs:
        res_dd (dict):
    """
    global TRUNCATABLE_KEYS, SAMPLE_COMPLETION

    # Basic setup: fetch the relevant files
    valid_file_fps, local_slices = find_files_index_range(
        dir_name, global_idx_start, global_idx_end,
    )
    all_keys = get_fields_in_hdf5(valid_file_fps[0], require_file_exist=True)

    # Basic setup: fetch argument values or set default values
    ignore_keys = ignore_keys if ignore_keys is not None else set()
    sample_keys = sample_keys if sample_keys is not None \
        else {*TRUNCATABLE_KEYS, SAMPLE_COMPLETION}

    # Finish setting up the keys based on what is present
    load_keys = load_keys if load_keys is not None else [
        key for key in all_keys
    ]
    load_keys = [key for key in load_keys if key not in ignore_keys]
    if any((key not in all_keys) for key in load_keys):
        logging.warning(
            f"Not all the requested keys from load_keys were found in the file. "
            f"load_keys: {load_keys} vs. all keys present: {all_keys}. Ignoring "
            f"the missing keys..."
        )
        load_keys = filter(lambda k: k in all_keys, load_keys)
    # Filter sample_keys so it only coincides with keys that are present
    # and does not include the keys we want to ignore...
    sample_keys = [
        key for key in sample_keys
        if (key not in ignore_keys) and (key in load_keys)
    ]

    # First get the keys only used in the first file
    first_file_keys = [key for key in load_keys if key not in sample_keys]
    res_dd = {
        key: load_field_in_hdf5(key, valid_file_fps[0])
        for key in first_file_keys
    }
    for key in sample_keys:
        res_dd[key] = []

    # Next, load all the concatable keys
    for valid_file_fp, load_slice in zip(valid_file_fps, local_slices):
        for key in sample_keys:
            res_dd[key].append(
                load_field_in_hdf5(key, valid_file_fp, idx_slice=load_slice)
            )
    # Flatten the lists of numpy arrays into numpy arrays
    for key in sample_keys:
        res_dd[key] = np.concatenate(res_dd[key], axis=0)

    # Finished!
    return res_dd

def get_multifreq_dset_dirs(
    dset: str,
    kbar_str_list: list,
    base_dir: str=None,
    dir_fmt: str=None,
):
    """Gets the dataset directories in the multi-frequency setting"""
    base_dir = base_dir if base_dir is not None else ""
    dir_fmt = dir_fmt if dir_fmt is not None else "{0}_train_measurements_nu_{1}"
    dir_list = [
        os.path.join(base_dir, dir_fmt.format(dset, kbar_str))
        for kbar_str in kbar_str_list
    ]
    return dir_list

def load_multi_dir_slice(
    dir_list,
    global_idx_start: int=0,
    global_idx_end: int=0,
    load_keys: Iterable=None,
    ignore_keys: Iterable=None,
    sample_keys: Iterable=None,
    freq_dep_keys: Iterable=None,
) -> dict:
    """Load multiple directories with an interface mirroring load_single_dir_slice
    """
    freq_dep_keys = freq_dep_keys if freq_dep_keys is not None else set()
    ignore_keys   = ignore_keys if ignore_keys is not None else set()

    # Load the data first
    dd_single = load_single_dir_slice(
        dir_list[-1],
        global_idx_start=global_idx_start,
        global_idx_end=global_idx_end,
        load_keys=load_keys,
        ignore_keys={*freq_dep_keys, *ignore_keys},
        sample_keys=sample_keys,
    )

    # Only load frequency-dependent keys here
    dd_list = [
        load_single_dir_slice(
            dir_name,
            global_idx_start=global_idx_start,
            global_idx_end=global_idx_end,
            load_keys=freq_dep_keys,
            ignore_keys=ignore_keys,
            sample_keys=sample_keys,
        )
        for dir_name in dir_list
    ]

    # Merge the frequency-dependent key entries
    dict_fdk = {
        fdk: np.stack(
            [dd[fdk] for dd in dd_list],
            axis=1,
        )
        for fdk in freq_dep_keys
    }

    dd_all = {
        **{k:v for (k,v) in dd_single.items() if k not in freq_dep_keys},
        **dict_fdk,
    }
    return dd_all


def nan_handler(
    dd: dict,
    nan_mode: str,
    check_fields: list,
    sample_fields: list=None,
) -> dict:
    """Goes through check_fields to find if any samples have nan values
    Returns a dictionary containing auxiliary information about which samples
    were loaded
    nan_mode expected to be one of ["keep", "zero", "skip"]
    """
    nan_mode = nan_mode.lower() if nan_mode is not None else "keep"
    check_fields = check_fields if check_fields is not None else []
    # sample_fields = sample_fields if sample_fields is not None else []
    sample_fields = sample_fields if sample_fields is not None else [
        *TRUNCATABLE_KEYS,
    ]
    sample_fields = set(filter(lambda f: f in dd.keys(), sample_fields))

    # aux_dd = dict()
    keep_idcs = np.logical_and.reduce([
        _get_valid_idcs(dd[key])
        for key in check_fields
        if  key in dd.keys()
    ])
    nan_idcs = np.logical_not(keep_idcs) # boolean array
    num_tot_samples = keep_idcs.shape[0]
    num_nan_samples = np.sum(nan_idcs)
    num_loaded_samples = num_tot_samples # adjust later

    if nan_mode == "keep":
        # Keep the NaNs
        pass
    elif nan_mode == "zero":
        # Zero out all the fields corresponding to any nan indices
        for key in sample_fields:
            dd[key][nan_idcs] = 0
    elif nan_mode == "skip":
        for key in sample_fields:
            dd[key] = dd[key][keep_idcs]
        num_loaded_samples = num_tot_samples - num_nan_samples
    else:
        raise ValueError(
            f"nan_handler received nan_mode={nan_mode} but expects one of "
            f"['keep', 'zero', 'skip']"
        )

    aux_dd = {
        "orig_idcs": keep_idcs,
        "num_tot_samples": num_tot_samples,
        "num_nan_samples": num_nan_samples,
        "num_loaded_samples": num_loaded_samples,
        "num_valid_samples":  num_tot_samples - num_nan_samples
    }
    return dd, aux_dd

def load_predictions_dataset(
    pred_scobj_dir: str,
    pred_mmg_dir_base: str,
    pred_mmg_dir_list: list = None,
    dset_name: str = None, # i.e., train/val/test
    str_nu_list: list = None,
    truncate_num: int = None,
    nan_mode: str = None,
    # cast_gamma_to_real: bool = True,
) -> Tuple[dict, dict]:
    """Loads in the dataset corresponding to
    the current predictions
    Parameters:
        pred_scobj_dir (str): path to the directory
            containing the scattering objects
        pred_mmg_dir_base (str): path to the base directory
            containing the measurement misfit gradients
        pred_mmg_dir_list (list): can explicitly pass a list of the
            directories relative to the pred_mmg_dir_base to use to fetch
            the relevant mmg data. Will be loaded in the order it is provided,
            regardless of the order of str_nu_list
        dset_name (str): in case pred_mmg_dir_list is not passed,
            this function will infer the relevant directory names
            based off the dset name; expected to be one of ["train", "val", "test"]
        str_nu_list (list): list of nu (or kbar/non-angular wavenumber) values as strings
            for use finding the right directories.
            only used if pred_mmg_dir_list is not provided
        truncate_num (int): number of samples to load; after this many it gets trunctaed
        nan_mode (str): specify how to handle nans that might be present in the files
            expected to be one of ["keep", "zero", "skip"]
            see nan_handler() above for details
        cast_gamma_to_real (bool): cast the gamma values to real if needed
    """
    # Default values
    str_nu_list = str_nu_list if str_nu_list is not None else []
    assert not(pred_mmg_dir_list is None and dset_name is None)

    # First, get the scattering objects in question
    scobj_ignore_keys = [Q_POLAR, Q_CART_LPF, Q_POLAR_LPF, D_RS, D_MH, SAMPLE_COMPLETION]
    pred_scobj_dd = load_single_dir_slice(
        pred_scobj_dir,
        global_idx_start=0,
        global_idx_end=truncate_num,
        ignore_keys=scobj_ignore_keys,
    )

    # Second, get the measurement misfit gradients
    # Get the relevant directories
    if pred_mmg_dir_list is None:
        # build the directory list
        pred_mmg_dir_list = [
            f"{dset_name}_gammas_nu_{nu}"
            for nu in str_nu_list
        ]

    mmg_load_keys = [OMEGA_SF, NU_SF, GAMMA_CART, SAMPLE_COMPLETION]
    pred_mmg_dd_list = []
    for pred_mmg_dir_i in pred_mmg_dir_list:
        pred_mmg_dd = load_single_dir_slice(
            os.path.join(pred_mmg_dir_base, pred_mmg_dir_i),
            global_idx_start=0,
            global_idx_end=truncate_num,
            load_keys=[OMEGA_SF, NU_SF, GAMMA_CART, SAMPLE_COMPLETION],
        )
        pred_mmg_dd_list.append(pred_mmg_dd)
    pred_mmg_keys = pred_mmg_dd_list[0].keys() if len(pred_mmg_dd_list) > 0 else []
    # Assemble the relevant frequency-dependent fields
    # carry around sample completion for debugging purposes
    freq_dependent_keys = [*FREQ_DEPENDENT_KEYS, SAMPLE_COMPLETION]
    pred_dd = pred_scobj_dd # start with scobj fields

    # If pred_mmg_dd_list is empty then pred_mmg_keys will also be empty
    for key in pred_mmg_keys:
        if key not in freq_dependent_keys:
            # Take the freq-independent key from the first loaded file
            pred_dd[key] = pred_mmg_dd_list[0][key]
        else:
            val = pred_mmg_dd_list[0][key]
            # If the number of elements is 1 but at most 1-dim then
            # concatenate along axis 0; otherwise stack along axis 1
            val_ndim = len(val.shape)
            val_size = val.size # np.prod(val.shape, dtype=int)
            should_concat = (val_ndim<=1 and val_size==1)
            combine_axis  = 0 if should_concat else 1
            combine_fn    = np.concatenate if should_concat else np.stack
            pred_dd[key]  = combine_fn(
                [pred_mmg_dd_list[i][key] for i in range(len(pred_mmg_dd_list))],
                axis=combine_axis,
            )
    # nan handling
    pred_dd, aux_dd = nan_handler(
        pred_dd,
        nan_mode=nan_mode,
        check_fields=[GAMMA_CART, Q_CART],
        sample_fields=[GAMMA_CART, Q_CART],
    )

    # build a metadata dd
    metadata_dd = {
        **{k:pred_dd[k] for k in KEYS_FOR_EXPERIMENT_INFO_OUT if k in pred_dd.keys()},
        **aux_dd,
    }
    return pred_dd, metadata_dd
