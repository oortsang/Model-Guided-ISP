import re
import yaml
from typing import Dict, Tuple
import argparse
import os, sys
import re

FULL_WILDCARD_PATTERN  = re.compile(r"<<[^=]+=(.*?)>>")
SMALL_WILDCARD_PATTERN = re.compile(r"<<[^=]+>>")
CHEVRON_PATTERN = re.compile(f"<<.*>>")

def parse_val(text_val, bool_as_str: bool=False, none_as_empty_str: bool=True):
    """Parses a text to int, float, or bool if requested"""
    try:
        return int(text_val)
    except:
        pass
    try:
        return float(text_val)
    except:
        pass
    if text_val in ["True", "true"]:
        return "true" if bool_as_str else True
    elif text_val in ["False", "false"]:
        return "false" if bool_as_str else False
    if text_val is None and none_as_empty_str:
        text_val = ""

    return text_val

def replace_single_var(in_str: str, field_name: str, field_val=None) -> str:
    """Helper function to handle a single variable at a time"""
    tmp_str = in_str[:]
    full_pattern  = re.compile(f"<<{field_name}+=(.*?)>>")
    small_pattern = re.compile(f"<<{field_name}>>")

    if field_val is not None:
        # replace with field_var
        field_val_str = str(field_val)
        tmp_str = re.sub(full_pattern,  field_val_str, tmp_str)
        tmp_str = re.sub(small_pattern, field_val_str, tmp_str)
    else:
        # try to use the default value
        tmp_str = re.sub(full_pattern, r"\1", tmp_str)
        tmp_str = re.sub(small_pattern, "", tmp_str)
    return tmp_str

def apply_replacements(
    in_str: str,
    replacement_dict: dict,
    cleanup: bool=True,
) -> str:
    """Applies the replacements in replacement_dict.
    Assumes the fields for replacement take the form <<var-name>>
    or <<var-name=default-value>>
    if the field value in the replacement_dict is given as None, skip it
    Optionally will recursively apply the replacements to other keys' values
    if those values contain chevrons
    """
    tmp_str = in_str[:]
    for field_name, field_val in replacement_dict.items():
        tmp_str = replace_single_var(tmp_str, field_name, field_val=field_val)
    out_str = tmp_str

    # Clean-up all fields, not just the ones passed
    if cleanup:
        out_str = re.sub(FULL_WILDCARD_PATTERN, r"\1", out_str)
        out_str = re.sub(SMALL_WILDCARD_PATTERN, "", out_str)
    return out_str

def apply_replacements_to_dict(
    in_dict: dict,
    replacement_dict: dict,
    cleanup: bool=False,
):
    """Applies the mappings from replacement_dict
    to the values of in_dict
    """
    out_dict = {
        k: apply_replacements(v, replacement_dict, cleanup=cleanup)
        for (k,v) in in_dict.items()
    }
    return out_dict

def partition_by_completion(replacement_dict: dict) -> Tuple[dict, dict]:
    """For a replacement map, partition the keys into those that are
    completed vs. those containing <<chevron-fields>> in their values
    e.g.
        {"field1": "a", "field2": "2+<<field1>>", "field3": "(<<field2>>)"}
    is split into
        {"field1", "a"} and {"field2": "2+<<field1>>", "field3": "(<<field2>>)"}
    The intention is that we can use completed values to fill in the remaining values
    """
    complete_dict = dict()
    incomplete_dict = dict()
    # chevron_pattern = re.compile("<<.*>>")
    for key, val in replacement_dict.items():
        # if len(re.findall(CHEVRON_PATTERN, val)) == 0:
        if re.search(CHEVRON_PATTERN, val) is None:
            # No replacements to be made
            complete_dict[key] = val
        else:
            incomplete_dict[key] = val
    return incomplete_dict, complete_dict

def propagate_replacements(
    replacement_dict: dict,
    recursion_depth: int=10,
    cleanup: bool=True,
):
    """Applies the replacements to any incomplete fields in the dictionary
    e.g.
        {"field1": "a", "field2": "2+<<field1>>", "field3": "(<<field2>>)"}
    is mapped to
        {"field1": "a", "field2": "2+a", "field3": "(<<field2>>)"}
    after a single round. If you set to recursion_depth to at least
        {"field1": "a", "field2": "2+a", "field3": "(2+a)"}
    """
    # Check for completion in case we can exit early
    # print([re.search(CHEVRON_PATTERN, v) is None for (k,v) in replacement_dict.items()])
    is_complete = all(re.search(CHEVRON_PATTERN, v) is None for (k,v) in replacement_dict.items())
    if is_complete:
        return replacement_dict

    # Base case
    if recursion_depth == 0:
        # Only apply cleanup here if requested
        out_dict = apply_replacements_to_dict(replacement_dict, dict(), cleanup=cleanup)
        return out_dict

    # Recursive step
    # Apply the replacement dict mappings to the values themselves
    incomplete_dict, complete_dict = partition_by_completion(replacement_dict)
    # Apply one round
    updated_dict = {
        **apply_replacements_to_dict(
            incomplete_dict,
            complete_dict,
            cleanup=False,
        ),
        **complete_dict,
    }
    # apply the rest recursively
    return propagate_replacements(
        updated_dict,
        recursion_depth=recursion_depth-1,
        cleanup=cleanup,
    )
