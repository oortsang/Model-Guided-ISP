# Script to help comb through the different hyperparameter optimization runs
# Expects tab-separated text files with a header in the first line

import os, glob  # get the relevant files
import yaml
import argparse
import logging
import sys

import numpy as np

from src.utils.logging_utils import FMT, TIMEFMT


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir-base",
        type=str,
        help="Indicate the directory containing all the relevant results files",
    )
    parser.add_argument(
        "--results-file-pattern",
        type=str,
        help="Give a pattern to match the desired model result logs",
    )
    parser.add_argument(
        "--output-summary-fp",
        default=str,
        help="Indicate the filepath for the output summary file",
    )
    parser.add_argument(
        "--field-name",
        default="eval_final_mse",
        type=str,
        help="name of the field to use for selection over",
    )
    parser.add_argument(
        "--selection-mode",
        default="min",
        choices=["min", "max"],
        help="Choose whether to select the min/max value for the given field",
    )
    parser.add_argument(
        "--verbosity-level", default=0, type=int, help="Choose level of outputs"
    )

    parser.add_argument(
        "--models-dir-base",
        type=str,
        default="rlc_data/models",
        help="Indicate the directory containing all the relevant results files",
    )

    a = parser.parse_args()
    return a


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

    return text_val  # unchanged if no other conversions are possible


def extract_line_by_field(
    file_name: str,
    field: str,
    selection_mode: str = "min",
    verbosity_level: int = 0,
) -> tuple[dict, float]:
    """Take a given field in a file and use it to extract the line containing the min/max value
    Parameters:
        file_name (string/file path): name of the relevant file to retrieve
        field (string): name of the field in question
        selection_mode (string): whether to choose the line with minimum/maximum field value
        verbosity_level (int): indicate a relative level of outputs
    Return Value:
        line_entry (dict): a lookup-table of the contents in this particular line (to avoid
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


def extract_lines_from_file_list(
    file_name_list: list[str],
    field: str,
    selection_mode: str = "min",
    verbosity_level: int = 0,
) -> tuple[dict, float]:
    """Take a given field in a file and use it to extract the line containing the min/max value
    Parameters:
        file_name_list (list of string/file path): list of relevant files to process
        field (string): name of the field in question
        selection_mode (string): whether to choose the line with minimum/maximum field value
    Return Value:
        file_name (str): the filename containing the relevant line
        line_entry (dict): a lookup-table of the contents in this particular line (to avoid
            concerns about ordering within the header)
        field_value_selected (int/float most likely): the relevant min/max value of the field in question
    """
    line_entries = []
    field_vals = []
    for file_name in file_name_list:
        # print(f"Processing file {file_name}")
        new_line_entry, new_field_val = extract_line_by_field(
            file_name,
            field,
            selection_mode=selection_mode,
        )
        line_entries.append(new_line_entry)
        field_vals.append(new_field_val)
        # logging.debug(f"File: {file_name}")
        # logging.debug(f"  {field} value: {new_field_val:.5e})")
        logging.debug(f"File: {file_name}; {field} value: {new_field_val:.5e}")
    field_arr = np.array(field_vals)
    file_idx = np.argmin(field_arr)
    file_name = file_name_list[file_idx]
    field_val = field_arr[file_idx].item()
    line_entry = line_entries[file_idx]
    return file_name, line_entry, field_val, field_arr


def extract_display_fields_from_file_list(
    file_name_list: list[str],
    display_fields: list,
    selection_field: str,
    selection_mode: str = "min",
) -> tuple[dict, float]:
    """Extract the lines associated with the selected field for each file
    Parameters:
        file_name_list (list of string/file path): list of relevant files to process
        selection_field (string): name of the field in question
        selection_mode (string): whether to choose the line with minimum/maximum field value
        display_fields (list of strings): specify which fields to keep in the final reported values
    Return Value:
        display_entries (list of lists): nested list containing the display_fields values for each file in the list corresponding to the line chosen by the selection_field and selection_mode parameters.
    """
    # line_entries = []
    display_entries = []
    field_vals = []
    for file_name in file_name_list:
        # print(f"Processing file {file_name}")
        new_line_entry, new_field_val = extract_line_by_field(
            file_name,
            selection_field,
            selection_mode=selection_mode,
        )
        # line_entries.append(new_line_entry)
        display_entries.append(
            [ file_name ] + [
                new_line_entry[key]
                for key in display_fields
                if key in new_line_entry.keys()
            ]
        )
        field_vals.append(new_field_val)
        # logging.debug(f"File: {file_name}; {field} value: {new_field_val:.5e}")
    return display_entries, field_vals


def main(args):
    """Searches through different results files to find the version with the best
    value for a given field. Selects the best model from different hyperparameter
    settings and also takes the best epoch within that run.
    Outputs the model selection to a summary yaml file.
    """
    # Parse the results file dir and pattern; also grab arguments
    file_name_dir = args.results_dir_base.rstrip("/")
    file_name_dir = args.results_dir_base
    file_name_pattern = args.results_file_pattern
    field_name = args.field_name
    selection_mode = args.selection_mode
    # summary_file_out = f"{file_name_dir}/{args.output_summary_fp}"
    summary_file_out = args.output_summary_fp

    # Find the matches
    glob_input = f"{file_name_dir}/{file_name_pattern}"
    logging.debug("Glob input: %s", glob_input)
    file_name_list = glob.glob(glob_input)
    logging.info("Found %i matching file names", len(file_name_list))
    selected_file_name, line_entry, selected_field_val, all_field_vals = (
        extract_lines_from_file_list(
            file_name_list,
            field=field_name,
            selection_mode=selection_mode,
            verbosity_level=args.verbosity_level,
        )
    )
    result_dir, result_file_name = os.path.split(selected_file_name)
    selected_model_dir = os.path.join(
        args.models_dir_base, os.path.splitext(result_file_name)[0]
    )
    selected_epoch = line_entry["epoch"]
    selected_model_fp = f"{selected_model_dir}/epoch_{selected_epoch}.pickle"
    logging.info(f"Selected model file: {selected_model_fp}")

    summary_dict = {
        "selected_result_file": selected_file_name,
        "selected_model_file": selected_model_fp,
        "field_used": field_name,
        "selection_mode": selection_mode,
        "field_val": selected_field_val,
        "log_info": line_entry,
        "epoch": selected_epoch,
        "file_list": {
            fname: float(fval) for (fname, fval) in zip(file_name_list, all_field_vals)
        },
    }

    with open(summary_file_out, "w") as sfile:
        yaml.dump(summary_dict, sfile, default_flow_style=False)

    logging.info(f"The following summary file can be found at {summary_file_out}")
    logging.info(yaml.dump(summary_dict, default_flow_style=False))


if __name__ == "__main__":
    a = setup_args()

    root = logging.getLogger()

    handler = logging.StreamHandler(sys.stderr)
    if a.verbosity_level > 0:
        handler.level = logging.DEBUG
        root.setLevel(logging.DEBUG)
    else:
        handler.level = logging.INFO
        root.setLevel(logging.INFO)

    formatter = logging.Formatter(FMT, datefmt=TIMEFMT)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    main(a)
