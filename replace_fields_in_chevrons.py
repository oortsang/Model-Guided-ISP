# replace_fields_in_chevrons.py: a driver script that applies string replacements
# of the form <<var>> with a corresponding value, specified by the replacement map.
# Default values may be provided as <<var=DEFAULT>>; if --use-defaults=true and a
# value for var is not supplied, the chevrons will be replaced by this default value.
# Intended to facilitate the preparation of a large number of badger config files
# while reducing the amount of boilerplate where the settings are stored.
#
# This file supports several usage patterns:
# 1. Files in/out:
#     python replace_fields_in_chevrons "a:5;b:10" <in-file> <out-file>
# This applies replacements to the contents of input as the output file
# 2. Streams in/out:
#     echo "blah <<a>> <<c=15>>" | python replace_fields_in_chevrons "a:5;b:10"
#     (outputs "blah 5 15" to stdout)
# Outputs to stdout
# 3. Mixing in/out streams/files; make sure to pass input/output files explicitly as
# --in-file/--out-file arguments as needed
# 4. Configuration file for template modification
#     python replace_fields_in_chevrons <config-yaml>
# In this case, the config file is treated as the replacement_map, and
# template is expected to have the following entries:
#     template-file: <the yaml file to use as the input file>
#     out-file: <the destination file>
# This way, the file locations can be easily controllable and reproducible.
#
# Example replacement behavior for input string "Hi <<name>>, haver of <<num-eyes=2>> good eyes"
# - Replacement map "name: Polyphemus; num-eyes: 0" will yield:
#     "Hi Polyphemus, haver of 0 good eyes"
# - Replacement map "name: Nobody" will yield:
#     "Hi Nobody, haver of 2 good eyes"
# - Replacement map "num-eyes: 1" will yield:
#     "Hi , haver of 2 good eyes"
#     (that is, name silently resolves to the empty string by default)
# Note: ignores any unused mappings from replacement_map

import re
import yaml
import argparse
import os, sys

from src.utils.replace_fields_utils import (
    FULL_WILDCARD_PATTERN,
    SMALL_WILDCARD_PATTERN,
    CHEVRON_PATTERN,
    parse_val,
    replace_single_var,
    apply_replacements,
    apply_replacements_to_dict,
    partition_by_completion,
    propagate_replacements,
)

# Read in from stdin if target-contents is None
def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "replacement_map", nargs="?", type=str, default=None,
        help="Replacement map as a semicolon(or newline)-separated string",
    )

    parser.add_argument(
        "--in-file", type=str, default=None,
        help="Input file path; defaults to stdin or the in-file/template-file field in the map file, if applicable"
    )
    parser.add_argument(
        "--out-file", type=str, default=None,
        help="Output file path; defaults to stdout or the out-file field in the map file, if applicable"
    )
    parser.add_argument(
        "--map-file", type=str, default=None,
        help="Basic replacement map as a yaml file"
    )
    parser.add_argument(
        "--recursion-depth", type=int, default=10,
        help="Number of recursions when applying replacements within the map file to itself; defaults to 10"
    )
    parser.add_argument(
        "--use-defaults", choices=["true", "false"], default="true",
        help="Whether to use default values provided in the in-file.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
    )

    a = parser.parse_args()
    a.use_defaults = (a.use_defaults == "true")

    # If we only receive the replacement map then we should treat it
    # as the map file instead
    only_passed_replacement_map = (
        a.replacement_map is not None
        and a.map_file is None
        and a.in_file is None
        and a.out_file is None
    )
    # import pdb; pdb.set_trace()
    if only_passed_replacement_map:
        a.map_file = a.replacement_map
        a.replacement_map = None

    return a


def main(args):
    if args.replacement_map is not None:
        replacement_map_str = args.replacement_map
        # print(f"RECEIVED: {replacement_map_str}\n")
        replacement_str_list = re.split(
            "[;\n]+", replacement_map_str.strip()
        )
        # print(f"SPLIT: {replacement_str_list}\n")
        replacement_pair_list = [
            tuple(parse_val(side.strip(), bool_as_str=True) for side in pair.split(":"))
            for pair in replacement_str_list
        ]
        if args.verbose:
            print(f"replacement_pair_list: {replacement_pair_list}\n")
        replacement_dict = {
            k: v for (k,v) in replacement_pair_list
        }
    else:
        with open(args.map_file, "r") as file:
            replacement_dict = yaml.safe_load(file.read())
        # print(f"yaml.safe_load gives: {replacement_dict}")
        if args.in_file is None:
            if "in-file" in replacement_dict.keys():
                args.in_file = replacement_dict["in-file"]
                print(f"Selected template file {args.in_file}")
            elif "template-file" in replacement_dict.keys():
                args.in_file = replacement_dict["template-file"]
                print(f"Selected template file {args.in_file}")
        if args.out_file is None:
            if "out-file" in replacement_dict.keys():
                args.out_file = replacement_dict["out-file"]
                print(f"Selected output file {args.out_file}")

    # Apply the replacements recursively
    replacement_dict = propagate_replacements(
        {k: str(v) for (k,v) in replacement_dict.items()},
        recursion_depth=args.recursion_depth,
        cleanup=True,
    )
    replacement_dict = {
        k: parse_val(v, bool_as_str=True)
        for (k,v) in replacement_dict.items()
    }

    in_file = args.in_file
    out_file = args.out_file
    if args.verbose:
        print(f"in_file: {in_file}")
        print(f"out_file: {out_file}")

    # Take input from in_file or stdin
    out_str = ""
    if in_file is None:
        # Read from stdin
        for line in sys.stdin:
            # print(line)
            out_str += apply_replacements(
                line, replacement_dict, cleanup=args.use_defaults,
            )
    else:
        # print(f"use_defaults: {args.use_defaults}")
        with open(in_file, "r") as f:
            for li, line in enumerate(f):
                # if "<<selection-field>>" in line:
                #     import pdb; pdb.set_trace()
                try:
                    out_str += apply_replacements(
                        line, replacement_dict, cleanup=args.use_defaults,
                    )
                except:
                    print(f"Note: apply_settings_yaml encountered an error at line {li}")
                    raise

    # Now output to out_file or stdout
    # print(out_str)
    if out_file is None:
        print(out_str)
    else:
        with open(out_file, "w") as f:
            f.write(out_str)

if __name__ == "__main__":
    a = setup_args()
    main(a)
