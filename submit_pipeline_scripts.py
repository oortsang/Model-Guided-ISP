# Submit the pipeline scripts and also
# grab the pickled taskpipeline
# This is the driver script we call

# Example usage:
# $ python submit_pipeline_scripts.py experiments/mini_runs/mini_hpscnn.py --command-str="f1 f2 f3" --use-pickled-pipeline=true
#
# This driver script performs the automated submission of components of the
# pipeline to Slurm. By default, it will re-use the pickled pipeline created
# by generate_pipeline_scripts.py. If the pipeline components have
# been generated yet, this driver script will generate them.
#
# To facilitate running specific steps of this pipeline (since sometimes runs
# may get interrupted or something may have crashed), the configuration files
# take a flexible "command string" argument, so you do not need to start over
# from the beginning every time.
# There are several modes:
# - "all": submits all tasks from the prepared pipeline
# - "f1 f2 f3 ...": it is possible to specify whole frequency blocks to submit
# at a time. The leading f characters are optional but clarify that these are
# frequency indices rather than actual k values. That is, regardless of the choice
# of frequencies, the frequency blocks will be numbered 1, 2, 3, 4, ...; this is
# true even if a frequency is repeated, for example if k_1=k_2.
# - "f1te f2sn f3 f4s": within each frequency block, it is possible to specify which
# sub-tasks to submit with a post-fix. These post-fixes need not be specified for
# every frequency block. The default codes are as follows:
#   - t: train, used for FYNet and MFISNet-Refinement
#   - e: eval, used for FYNet and MFISNet-Refinement
#   - s: solver, used for HPS-CNN's refinement blocks (Measurement Misfit Gradients)
#   Note: currently, the pipeline does not support running the solver for a single
#   dataset, but this can be done by calling badger on the underlying yaml within
#   the pipeline_scripts/<current-run>/ directory.
#   - n: neural network; this is a fused train/eval interface (MMG Update Blocks)
#   (FYNet and MFISNet-Refinement included split train/eval subtasks as previously
#   we performed frequency-wise hyperparameter searches. This was a bit cumbersome,
#   and we found it was was not necessary for the HPS-CNN's simpler 2D CNN blocks,
#   so we combined the train/eval operations into a single task)
# - "none": submits nothing. Handy for combining with --dry-run=true just to see
# the code summary for the whole pipeline without accidentally submitting anything.
# The subtask codes can be found in src/utils/pipeline_blocks.py. Regardless of which
# command string is used ("all", "none", or a selection), the pipeline summary
# will always display the character codes for each subtask up front.
#
# Other arguments of note:
# - --dry-run: this argument allows you to verify the pipeline submission
# plan before going through with the slurm job submissions.
# - --sleep-time: this argument lets you to adjust the amount of time between slurm
# job submissions. Depending on the cluster setup, submitting jobs too quickly may
# sometimes cause problems.
#
# Example output
# # Reports the commands received
# Received command_str=f1 f2 f3 f4s
# Full pipeline code str: 1te 2sn 3sn 4sn 5sn 6sn 7sn 8sn 9sn 10sn
# Full pipeline code obj:
# 1te 2sn 3sn 4sn 5sn 6sn 7sn 8sn 9sn 10sn: MMG Pipeline
#   1te: f1
#     t: train-fynet
#       TrainFYNet train-fynet
#     e: eval-fynet
#       EvalFYNet eval-fynet
#   2sn: f2
#     s: run-mmg-solver
#       RunMMGSolverDataset run-mmg-solver-train
#       RunMMGSolverDataset run-mmg-solver-val
#       RunMMGSolverDataset run-mmg-solver-test
#     n: train-mmgublock
#       TrainEvalMMGUBlock train-mmgublock
# [...] (ommitted, but the entire pipeline is displayed)
# # Displays the plan of tasks to be run
# ~~ Run plan ~~~
# SequentialTasks Selected Tasks Pipeline (f1 f2 f3)
#   FrequencyBlock f1
#     TrainFYNet train-fynet
#     EvalFYNet eval-fynet
#   FrequencyBlock f2
#     RunMMGSolver run-mmg-solver
#       RunMMGSolverDataset run-mmg-solver-train
#       RunMMGSolverDataset run-mmg-solver-val
#       RunMMGSolverDataset run-mmg-solver-test
#     TrainEvalMMGUBlock train-mmgublock
#   FrequencyBlock f3
#     RunMMGSolver run-mmg-solver
#       RunMMGSolverDataset run-mmg-solver-train
#       RunMMGSolverDataset run-mmg-solver-val
#       RunMMGSolverDataset run-mmg-solver-test
#     TrainEvalMMGUBlock train-mmgublock
#   FrequencyBlock f4
#     RunMMGSolver run-mmg-solver
#       RunMMGSolverDataset run-mmg-solver-train
#       RunMMGSolverDataset run-mmg-solver-val
#       RunMMGSolverDataset run-mmg-solver-test
# ~~~~~~~~~~~~~~~
# # Reports the slurm job ids, space-separated for easy copy/paste
# # after scancel if needed (without needing to cancel all your
# # other jobs that may be running)
# f1 jobs: 2523544 2523545
# f2 jobs: 2523546 2523547 2523548 2523549
# f3 jobs: 2523550 2523551 2523552 2523553
# f4 jobs: 2523554 2523555 2523556
# All pipeline jobs encountered: 2523544 2523545 2523546 2523547 2523548 2523549 2523550 2523551 2523552 2523553 2523554 2523555 2523556


import re, os, sys, time
import argparse
import importlib.util
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

    cfg_spec = importlib.util.spec_from_file_location(
        "config_script", config_fp,
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
    return pipeline_output

if __name__ == "__main__":
    args = setup_args()
    main(args)
