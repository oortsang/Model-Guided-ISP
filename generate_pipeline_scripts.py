# Generate the pipeline scripts and also
# set up the taskpipeline to be pickled
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
    parser.add_argument("config_file")
    parser.add_argument("--verbosity", type=int, default=2)
    # parser.add_argument(
    #     "--create-slurm-scripts",
    #     choices=["true", "false"],
    #     default="true",
    # )
    a = parser.parse_args()
    # a.create_slurm_scripts = True if a.create_slurm_scripts == "true" else False
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

    pipeline_output = cfg_module.generate_pipeline(verbosity=args.verbosity)


if __name__ == "__main__":
    args = setup_args()
    main(args)
