# Fetch a result entry and add it to a new or existing results file
# Also add several new fields:
# frequency index, frequency value
# target train type (smoothed/original) if known

import re
import argparse
import yaml
import sys, os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logging_utils import FMT, TIMEFMT, find_best_epoch, parse_val, update_field_in_yaml_file


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-results-fp")
    parser.add_argument("--training-results-key")
    parser.add_argument(
        "--new-entries",
        default=None,
        help=(
            "enter a semicolon-separated list of entries to insert, in the form "
            "\"freq_idx=5,entry_name=value\". Will ignore whitespace"
        )
    )
    parser.add_argument("--freq-idx", type=int)
    parser.add_argument("--central-results-fp", default=None, help="centralized yaml file for the results of a run")

    a = parser.parse_args()
    return a

def parse_new_entries(new_entries_str: str) -> dict:
    """Parse the string of the new entries to add
    Expected format: "\"freq_idx=5;entry_name=value;..."
    """
    new_entries_str = new_entries_str.strip() if new_entries_str is not None else None
    if new_entries_str is None or len(new_entries_str) == 0 \
       or "=" not in new_entries_str:
        return dict()

    new_entries_tup_list = [
        [x.strip()  for x in entr.split("=")]
        for entr in new_entries_str.split(";")
    ]
    new_entries_dd = {
        key: parse_val(val)
        for (key, val) in new_entries_tup_list
    }
    return new_entries_dd


def main(args):
    """Fetch the relevant entry
    """
    new_entries_dd = parse_new_entries(args.new_entries)

    best_epoch_dd = find_best_epoch(
        args.training_results_fp,
        args.training_results_key,
        selection_mode="min"
    )
    print(best_epoch_dd.get("epoch", "No field 'epoch' found!"))

    combined_dd = {
        **best_epoch_dd,
        **new_entries_dd,
    }

    if args.central_results_fp is not None:
        # write_result_to_file(args.central_results_fp, **train_dd)
        update_field_in_yaml_file(
            f"freq_idx_{args.freq_idx}",
            combined_dd,
            args.central_results_fp,
        )
    else:
        print(f"Retrieved the data: {combined_dd}")


if __name__ == "__main__":
    a = setup_args()
    main(a)
