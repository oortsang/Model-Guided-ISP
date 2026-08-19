# Submit the pipeline scripts and also
# grab the pickled taskpipeline
# This is the driver script we call

import re, os, sys, time
import argparse
import importlib.util
# from src.utils.pipeline_utils import (
#     apply_settings_yaml,
#     SoloTask,
#     FrequencyBlock,
#     SequentialTasks,
#     ParallelTasks,
# )
# from src.utils.pipeline_blocks import (
#     TaskPipeline,
# )

import badger

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    bool_choices = ["true", "false"]
    parser.add_argument("config_file")
    parser.add_argument("--command-str", type=str, default="all")
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--sleep-time", type=float, default=0.5)
    parser.add_argument("--dry-run", choices=bool_choices, default="false")
    parser.add_argument(
        "--use-pickled-pipeline",
        choices=["true", "false"],
        default="true",
    )
    a = parser.parse_args()
    a.use_pickled_pipeline = True if a.use_pickled_pipeline == "true" else False
    a.dry_run = True if a.dry_run == "true" else False
    return a


def main(args):
    """Can I do setup in this function??
    I guess I'm importing badger?
    """
    config_fp = args.config_file
    # runpy.run_path(config_fp, run_name="__main__")

    cfg_spec = importlib.util.spec_from_file_location(
        "config_script",
        config_fp,
    )
    cfg_module = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg_module)

    pipeline_output = cfg_module.run_pipeline(
        command_str=args.command_str,
        use_pickled_pipeline=args.use_pickled_pipeline,
        verbosity=args.verbosity,
        sleep_time=args.sleep_time,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    args = setup_args()
    main(args)
