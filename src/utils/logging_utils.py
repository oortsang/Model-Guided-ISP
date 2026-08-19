import hashlib
import os
import logging
import yaml
import numpy as np
from typing import Any, Dict, Tuple


FMT = "%(asctime)s:MFISNets: %(levelname)s - %(message)s"
TIMEFMT = "%Y-%m-%d %H:%M:%S"


def hash_dict(dictionary: Dict[str, Any]) -> str:
    """Create a hash for a dictionary."""
    dict2hash = ""

    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]

        dict2hash += "%s_%s_" % (str(k), str(v))

    return hashlib.md5(dict2hash.encode()).hexdigest()


def write_result_to_file(
        fp: str,
        missing_str: str = "",
        create_new_dirs: bool=True,
        **trial) -> None:
    """Write a line to a tab-separated file saving the results of a single
        trial.
    Parameters
    ----------
    fp : str
        Output filepath
    missing_str : str
        (Optional) What to print in the case of a missing trial value
    **trial : dict
        One trial result. Keys will become the file header
    Returns
    -------
    None
    """
    header_lst = list(trial.keys())
    header_lst.sort()
    if not os.path.isfile(fp):
        # Help debug in case the directory does not exist
        cont_dir = os.path.dirname(fp)
        cont_dir_exists = os.path.isdir(cont_dir)
        cont_dir_descr = "exists" if cont_dir_exists else "does not exist"
        logging.info(f"(write_result_to_file) containing directory {cont_dir} {cont_dir_descr}")
        if create_new_dirs and not cont_dir_exists:
            os.makedirs(cont_dir, exist_ok=True)
            logging.info(f"(write_result_to_file) created the directory!")

        header_line = "\t".join(header_lst) + "\n"
        with open(fp, "w") as f:
            f.write(header_line)
    trial_lst = [str(trial.get(i, missing_str)) for i in header_lst]
    trial_line = "\t".join(trial_lst) + "\n"
    with open(fp, "a") as f:
        f.write(trial_line)

def load_tab_separated_file_to_dict(file_name):
    """Loads a tab-separated file to a dictionary
    using the first row as a header/field keys
    """
    with open(file_name, "r") as file:
        file_contents = [line.strip().split("\t") for line in file]
    header   = file_contents[0]
    contents = file_contents[1:]
    # file_dd  = {field: [] for field in header}
    tranposed_list = [[] for field in header]
    for i, line in enumerate(contents):
        for j, entry in enumerate(line):
            tranposed_list[j].append(parse_val(entry))
    file_dd = {field: tranposed_list[fi] for fi, field in enumerate(header)}
    return file_dd


def extract_line_by_field(
    file_name: str,
    field: str,
    selection_mode: str = "min",
) -> Tuple[Dict, float]:
    """
    Takes a tab-separated file and extracts the line containing the min/max value of a given field.

    This is used to find the best epoch in a training log, for example.
    Parameters:
        file_name (string/file path): name of the relevant file to retrieve
        field (string): name of the field in question
        selection_mode (string): whether to choose the line with minimum/maximum field value
        verbosity_level (int): indicate a relative level of outputs
    Return Value:
        line_entry (Dict): a lookup-table of the contents in this particular line (to avoid
            concerns about ordering within the header)
        field_value_selected (int/float most likely): the relevant min/max value of the field in question
    """
    with open(file_name, "r") as file:
        file_contents = [line.strip().split("\t") for line in file]
    header = file_contents[0]
    contents = file_contents[1:]

    try:
        field_idx = header.index(field)
    except:
        raise KeyError(f"Unable to locate field '{field}' in the header {header}")
    field_arr = np.array([parse_val(entry[field_idx]) for entry in contents])

    if selection_mode.lower() == "min":
        line_idx = np.argmin(field_arr)
    elif selection_mode.lower() == "max":
        line_idx = np.argmax(field_arr)
    else:
        raise ValueError(
            f"Expected mode keyword as one of ['min', 'max'] to choose the selection direction"
        )
    field_val_selected = field_arr[line_idx]
    line_entry = {
        key: parse_val(contents[line_idx][ki]) for ki, key in enumerate(header)
    }

    return line_entry, field_val_selected


def parse_val(text_val):
    """Parses a text to int, float, or bool if possible"""
    try:
        return int(text_val)
    except:
        pass
    try:
        return float(text_val)
    except:
        pass
    if text_val in ["True", "true"]:
        return True
    elif text_val in ["False", "false"]:
        return False
    else:
        return text_val


def find_best_epoch(
    results_fp: str, val_error_field: str, selection_mode: str = "min"
) -> Dict:
    """
    Find the epoch with the best validation error in a training log.
    Parameters:
        results_fp (str): path to the training log
        val_error_field (str): name of the validation error field in the log
        selection_mode (str): whether to choose the line with minimum/maximum validation error
    Return Value:
        (Dict): the key-value mapping of the best epoch's contents
    """
    line_entry, val_error = extract_line_by_field(
        results_fp, val_error_field, selection_mode=selection_mode
    )
    return line_entry


def update_field_in_yaml_file(
    field_name: str,
    val: dict,
    yaml_fp: str,
) -> dict:
    """Update one field in a yaml file; create it if it does not exist already.
    Designed for use with the central results file for tracking multi-stage runs
    i.e., field_name would be something like freq_idx_1.
    Note that this will squash entries without checking.
    Returns a dictionary containing the loaded fields from the yaml file in question.
    """
    # Load the file if it exists, otherwise create a new file later
    if os.path.exists(yaml_fp):
        with open(yaml_fp, "r") as yaml_file:
            curr_dict = yaml.safe_load(yaml_file)
    else:
        curr_dict = dict()
        # Create the directory if needed
        os.makedirs(os.path.split(yaml_fp)[0], exist_ok=True)

    # Update the entry in question
    curr_dict[field_name] = val
    with open(yaml_fp, "w") as yaml_file:
        yaml.dump(curr_dict, yaml_file, default_flow_style=False)

    # Return the loaded dict
    return curr_dict

def load_yaml_to_dict(
    yaml_fp: str,
) -> dict:
    """Loads an entire yaml file
    Designed for use with the model hyperparameters but should be fairly generic
    Throws a FileNotFound error if the file is not found and a KeyError if the field is not found in the file.
    Returns a dictionary.
    """

    if os.path.exists(yaml_fp):
        with open(yaml_fp, "r") as yaml_file:
            curr_dict = yaml.safe_load(yaml_file)
            return curr_dict
    else:
        raise FileNotFoundError(f"(load_field_in_yaml_file) Unable to locate requested yaml file {yaml_fp}")


def load_field_in_yaml_file(
    field_name: str,
    yaml_fp: str,
) -> dict:
    """Loads a single field from a yaml file.
    Designed for use with the model hyperparameters, e.g. field_name="freq_idx_2".
    Throws a FileNotFound error if the file is not found and a KeyError if the field is not found in the file.
    Returns a dictionary.
    """

    if os.path.exists(yaml_fp):
        with open(yaml_fp, "r") as yaml_file:
            curr_dict = yaml.safe_load(yaml_file)
        if field_name in curr_dict.keys():
            return curr_dict[field_name]
        else:
            raise KeyError(f"(load_field_in_yaml_file) Requested key {field_name} not found; existing keys include: {list(curr_dict.keys())}.")
    else:
        raise FileNotFoundError(f"(load_field_in_yaml_file) Unable to locate requested yaml file {yaml_fp}")

def save_dict_to_yaml(data_dict: dict, yaml_fp: str):
    """Helper function to save a dictionary as a yaml file. Overwrites existing files."""
    with open(yaml_fp, "w") as yaml_file:
        yaml.dump(data_dict, yaml_file, default_flow_style=False)
