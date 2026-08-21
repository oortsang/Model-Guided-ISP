# Similar in function to select_hyperparameter_from_runs.py except
# that this is built to use the centralized yaml files
# The main differences are:
# - the summary files already select the best epoch from the given run
# - the summary files are yamls instead
# - the summary files include the full model file paths
# - (new) provides the option to copy over the model parameters


import os, glob  # get the relevant files
import yaml
import argparse
import logging
import sys
import re
import shutil

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logging_utils import FMT, TIMEFMT, load_yaml_to_dict


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
        "--results-section",
        default="finetune_info",
        type=str,
        help=(
            "Choose which section/stage to read the results from "
            "(i.e., the 'finetune_info' or 'freq_idx_2' section of the "
            "results file yaml)"
        ),
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
        type=str,
        help="Choose whether to select the min/max value for the given field",
    )
    parser.add_argument(
        "--centralize-models",
        default="false",
        choices=["true", "false"],
        help="Whether to copy the selected models over to a centralized location",
    )
    parser.add_argument(
        "--central-model-dir",
        default=None,
        type=str,
        help="Where to copy the selected models if --centralize-models=true",
    )
    parser.add_argument(
        "--central-model-fp-format",
        type=str,
        default="e2e_{0}",
        help="Whether to copy the selected models over to a centralized location",
    )
    parser.add_argument(
        "--verbosity-level", default=1, type=int, help="Choose level of outputs"
    )

    a = parser.parse_args()
    a.centralize_models = (a.centralize_models == "true")
    return a


def main(args):
    """Searches through different results files to find the version with the best
    value for a given field. Selects the best model from different hyperparameter
    settings.
    Outputs the model selection to a summary yaml file.
    """
    verbosity_level = args.verbosity_level

    # Parse the results file dir and pattern; also grab arguments
    file_name_dir = args.results_dir_base.rstrip("/")
    file_name_dir = args.results_dir_base
    file_name_pattern = args.results_file_pattern
    results_section = args.results_section
    field_name = args.field_name
    selection_mode = args.selection_mode
    # summary_file_out = f"{file_name_dir}/{args.output_summary_fp}"
    summary_file_out = args.output_summary_fp
    # Find the matches
    glob_input = f"{file_name_dir}/{file_name_pattern}"
    if verbosity_level >= 1:
        print(f"Glob input: {glob_input}")
    file_name_list = glob.glob(glob_input) # note: this contains the entire path name
    if verbosity_level >= 1:
        print(f"Found {len(file_name_list)} matching file names")

    if verbosity_level >= 1:
        print(f"results_section: {results_section}")
    full_dd_list = [
        load_yaml_to_dict(file_name)
        for file_name in file_name_list
    ]
    dd_list = [dd.get(results_section, None) for dd in full_dd_list]
    all_field_vals = [dd[field_name] for dd in dd_list]
    if verbosity_level >= 1:
        print(f"all_field_vals: {all_field_vals}")
    if selection_mode == "min":
        sel_idx = np.argmin(all_field_vals)
    else:
        sel_idx = np.argmax(all_field_vals)
    if verbosity_level >= 1:
        print(f"Selected idx={sel_idx} with field val {all_field_vals[sel_idx]}")

    sel_dd = dd_list[sel_idx]
    sel_full_dd = full_dd_list[sel_idx]
    selected_field_val = all_field_vals[sel_idx]
    selected_file_name = file_name_list[sel_idx]


    try:
        # Helper that picks out the last string of digits in the section name
        # for sorting purposes
        _idx_sorter = lambda s: int(re.findall("[0-9]+", s)[-1])
        freq_list = sorted(
            [k for k in sel_full_dd.keys() if "freq_idx" in k],
            key=_idx_sorter
        )
        # Grabs from each section
        model_fp_list = {
            k: sel_full_dd[k]["central_model_fp"]
            for k in freq_list
        }
        print(f"Model fp list: {model_fp_list}")
        model_dir = os.path.split(model_fp_list[freq_list[0]])[0]
    except:
        model_fp_list = None
        model_dir = None

    if args.centralize_models:
        # Copy files and update model_fp_list
        # args.central_model_fp_format
        centralized_model_fp_list = []
        central_model_dir = args.central_model_dir
        os.makedirs(central_model_dir, exist_ok=True)
        for model_fp in model_fp_list.values():
            model_fp_filename = os.path.split(model_fp)[1]
            dst_model_fp = os.path.join(
                central_model_dir,
                args.central_model_fp_format.format(model_fp_filename),
            )
            print(f"Copying over {model_fp} to {dst_model_fp}")
            shutil.copy2(model_fp, dst_model_fp)
            centralized_model_fp_list.append(dst_model_fp)
        # Actually I think this is not necessary
        # print(f"TODO: update block_fp_list...")
        # model_fp_list = centralized_model_fp_list # wrong one...

    summary_dict = {
        "grid_search_info": {
            "results_section": results_section,
            "selected_result_file": selected_file_name,
            "field_used": field_name,
            "field_val": selected_field_val,
            "selection_mode": selection_mode,
            "file_list": {
                fname: float(fval) for (fname, fval) in zip(file_name_list, all_field_vals)
            },
            "all_field_vals": all_field_vals,
            "model_fp_list": model_fp_list,
            "model_dir": model_dir,
        },
        **sel_full_dd,
    }
    with open(summary_file_out, "w") as sfile:
        yaml.dump(summary_dict, sfile, default_flow_style=False)

    if verbosity_level >= 1:
        print(f"model_dir: {model_dir}")

    if verbosity_level >= 2:
        logging.info(yaml.dump(summary_dict, default_flow_style=False))
    print(f"The summary file can be found at {summary_file_out}")
    logging.info(f"The summary file can be found at {summary_file_out}")

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
