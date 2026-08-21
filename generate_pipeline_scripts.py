# Generate the pipeline scripts and also
# set up the taskpipeline to be pickled
# This is the driver script we call


# Example usage:
# $ python generate_pipeline_scripts.py experiments/mini_runs/mini_hpscnn.py --verbosity=2
#
# This driver script parses the configuration file, creating a pipeline of tasks
# that can be submitted in part or in whole; also allows for the inspection and
# editing of the underlying bash scripts and badger config yamls being submitted
# to Slurm for easier debugging.
#
# Sample output:
# # First, the script displays the sequence of tasks being specified in the python config file
# ~~~ Pipeline Outline ~~~
# SequentialTasks MMG Pipeline
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
# [...]
#
# # Next, the script reports the inputs/outputs for each frequency block and other
# # potentially useful values that may change frequently between scripts
# ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~Frequency Block f1~~~
# f1 train-fynet dataset dir   dataset
# f1 train-fynet train targets original
# f1 eval-fynet output pred dir mg_data/mmg_pipeline/predictions/2026-08-20_eval_original_train_original_fynet_f1_for_mini_hpscnn_run
# f1 eval-fynet dataset dir     dataset
# f1 eval-fynet eval targets    original
# ~~Frequency Block f2~~~
# f2 run-mmg-solver input_pred_scobj_dir  mg_data/mmg_pipeline/predictions/2026-08-20_eval_original_train_original_fynet_f1_for_mini_hpscnn_run
# f2 run-mmg-solver output_mmg_rel_dir    mg_data/mmg_pipeline/predictions/2026-08-20_hps_mmg_f2_for_mini_hpscnn_run
# f2 run-mmg-solver output_name_format    {output_dir}/{dset}_gammas_nu_{nu_sf}/gammas_{input_label}.h5
# f2 run-mmg-solver-train num_samples     10
# f2 run-mmg-solver-val num_samples     10
# f2 run-mmg-solver-test num_samples     10
# f2 train-mmgublock ref dataset dir       dataset
# f2 train-mmgublock input pred scobj dir  mg_data/mmg_pipeline/predictions/2026-08-20_eval_original_train_original_fynet_f1_for_mini_hpscnn_run
# f2 train-mmgublock input pred mmg dir    mg_data/mmg_pipeline/predictions/2026-08-20_hps_mmg_f2_for_mini_hpscnn_run
# f2 train-mmgublock output pred scobj dir mg_data/mmg_pipeline/predictions/2026-08-20_eval_original_train_original_mmgu_f2_for_mini_hpscnn_run
# f2 train-mmgublock train targets         original
# f2 train-mmgublock eval targets          original
# [...]

import re, os, sys, time
import argparse
import importlib.util
import badger

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--verbosity", type=int, default=2)
    a = parser.parse_args()
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

    pipeline_output = cfg_module.generate_pipeline(verbosity=args.verbosity)
    return pipeline_output

if __name__ == "__main__":
    args = setup_args()
    main(args)
