# generate_pipeline_scripts.py calls generate_pipeline
# submission occurs separately and involves calling run_pipeline

import re, os, sys, time, copy
import pickle

from src.utils.pipeline_utils import (
    # Helper functions
    map_dict_vals_to_str,
    pretty_dict_to_str,
    apply_settings_yaml,
    # Classes
    SoloTask,
    FrequencyBlock,
    SequentialTasks,
    ParallelTasks,
    pretty_dict_to_str,
    str_tuple_to_list,
    copy_and_name,
    TaskPipeline,
    # Verbosity levels
    # Generation
    VLVL_IO_INFO,
    VLVL_INIT_CONFIG,
    VLVL_SCRIPTS_INFO,
    VLVL_EFF_CTX,
    # Running
    VLVL_ALL_JOBS,
    VLVL_RUN_PLAN,
    VLVL_BLOCK_JOBS,
    # VLVL_SBATCH_CMD,
)
from src.utils.pipeline_blocks import (
    TrainFYNet,
    EvalFYNet,
    TrainMMGUBlock,
    TrainMRefBlock,
    EvalMRefBlock,
    TrainMRefPipeline,
    EvalMRefPipeline,
    RunMMGSolver,
    MMGTaskPipeline,
    # MPSRTaskPipeline,
    get_standard_mmg_settings,
    load_system_setup,
    setup_basic_pipeline_tasks,
    setup_basic_mmg_pipeline,
    WHOLE_RUN_NAME,
    FREQ_IDX,
)
from src.utils.replace_fields_utils import (
    apply_replacements,
    apply_replacements_to_dict,
    propagate_replacements,
)


# Commonly updated stuff
RUN_DATE_VAL = "2025-10-10"

STR_NU_LIST_VAL    = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
WHOLE_RUN_NAME_VAL = "mref_Nk_10_(1,2,3,4,5,6,7,8,9,10)_ntr_1000_inf_noise_0.90_targets_smoothed_epochs_300"
WANDB_PROJECT_VAL  = "2025-09-03_mmg_pipeline"

# Debugging settings...
# print(f"NOTE: #EPOCHS AND #TRAINING SAMPLES HAVE BEEN SIGNIFICANTLY LOWERED")
NUM_TRAIN = 1000
NUM_VAL   = 1000
NUM_TEST  = 1000

NUM_EPOCHS = 300
NUM_E2E_EPOCHS = 100
MREF_TARGETS = "smoothed"
NOISE_LEVEL  = "0.90"

def setup_pipeline(verbosity: int = 3, generate_scripts: bool=True, **kwargs) -> dict:
    """Shared setup function between generate_pipeline and run_pipeline
    (shared so it's easier to fetch the pipeline when running the pipeline, but
    we also don't need to re-generate the scripts)
    """
    global RUN_DATE_VAL, WHOLE_RUN_NAME, STR_NU_LIST_VAL, WANDB_PROJECT_VAL
    global NUM_TRAIN, NUM_VAL, NUM_TEST, MREF_TARGETS, NOISE_LEVEL
    ### Common settings ###
    system_setup = load_system_setup()
    rlc_repo_dir = system_setup["file-paths"]["repo-dir"]
    rlc_data_dir = system_setup["file-paths"].get("data-rel-dir", "rlc_data")
    standard_settings = get_standard_mmg_settings(rlc_repo_dir, rlc_data_dir)
    # Sets the default templates and jobs/logs directories

    str_nu_list = STR_NU_LIST_VAL
    data_input_nus = "(" + ",".join(str_nu_list) + ")"
    Nk = len(str_nu_list)
    shard_size = 1000 # makes sense when we use only 1000 samples total
    common_settings = {
        **standard_settings,
        "run-date": RUN_DATE_VAL,
        "model-date": RUN_DATE_VAL,
        "whole-run-name": WHOLE_RUN_NAME_VAL,
        "pipeline-scripts-rel-dir": "pipeline_scripts",
        "scripts-dir":              "<<pipeline-scripts-rel-dir>>/<<run-date>>_<<whole-run-name>>",
        "central-run-dir":          "<<central-rel-dir>>/<<run-date>>_<<whole-run-name>>",
        "central-model-dir":        "<<central-run-dir>>/models/",
        "central-model-format":     "model_params_<<freq-idx>>.pickle",
        "central-results-fp":       "<<central-run-dir>>/summary.yaml",
        # For use during the eval e2e block though...
        "central-model-e2e-format": "e2e_{0}", # use with .format(model_fp)

        # Meta.. where to save this task pipeline?
        "scripts-task-pipeline-fp": "<<scripts-dir>>/task_pipeline.pickle",

        # General stuff
        # "rlc-repo":                 rlc_repo_dir,
        # "rlc-data":                 rlc_data_dir,
        # "dataset-rel-dir":          "dataset", # for FYNet/base
        # "ref-dataset-rel-dir":      "dataset", # for MMG/refinement-based stuff
        # "ref-dataset-dir":          "dataset", # for MMG/refinement-based stuff
        # "dataset-dir":              "dataset", # just in case
        # "templates-rel-dir":        "pipeline_templates",
        # "predictions-rel-dir":      f"{rlc_data_dir}/mmg_pipeline/predictions",
        # "results-rel-dir":          f"{rlc_data_dir}/mmg_pipeline/results",
        # "models-rel-dir":           f"{rlc_data_dir}/mmg_pipeline/models",
        # "central-rel-dir":          f"{rlc_data_dir}/mmg_pipeline/central_run_info",

        # Slurm stuff
        # "partition": "gpu",
        # "num-cpu": 2,
        # "num-gpu": 1,
        "exclude-node": '""',
        # "mem": "50G",
        # "mail-type": "NONE",
        # "mail-user": "NONE",
        "wandb-project": WANDB_PROJECT_VAL,
        # "wandb-entity": "recursive-linearization",
        # "wandb-mode": "online",

        # Data settings
        "data-input-nus": data_input_nus,
        "str_nu_list": str_nu_list,
        "nu_list": [float(str_nu) for str_nu in str_nu_list],
        "Nk": Nk,
        "num-freqs": Nk,
        "num-train": NUM_TRAIN,
        "num-val":   NUM_VAL,
        "num-test":  NUM_TEST,
        "noise-level": NOISE_LEVEL,
        "use-noise-seed": "true",
        # "noise-seed-offset": 3221,
        "noise-seed-base-train": 10000000,
        "noise-seed-base-val":   20000000,
        "noise-seed-base-test":  30000000,
        "noise-seed-train": "10000*<<freq-idx>>+3221+<<noise-seed-base-train>>",
        "noise-seed-val":   "10000*<<freq-idx>>+3221+<<noise-seed-base-val>>",
        "noise-seed-test":  "10000*<<freq-idx>>+3221+<<noise-seed-base-test>>",
        "noise-seed-format-rule": "{noise_seed_base}+{input_label}",
        "n-epochs":           NUM_EPOCHS,
        "num-epochs":         NUM_EPOCHS,
        "n-epochs-per-log":   5,
        "num-epochs-per-log": 5,
        "log-train-subset-frac": 1,
        "train-targets": MREF_TARGETS,
        "eval-targets":  MREF_TARGETS,

        # General optimization stuff
        "lr-decrease-factor": 1,
        "output-pred-shard-size": 250,
    }


    # Replace fields like <<field>> if they were defined by other keys in the dictionary
    common_settings = propagate_replacements(
        map_dict_vals_to_str(common_settings),
        cleanup=False,
    )
    # Prepare the list of noise seeds for all
    dset_list = ["train", "val", "test"]
    noise_seed_list_dict = {}
    for dset in dset_list:
        noise_seed_dset_list = [
            (10000 * fi + 3221 + int(common_settings[f"noise-seed-base-{dset}"]))
            for fi in range(1, 1+Nk)
        ]
        noise_seed_list_dict[dset] = noise_seed_dset_list
        common_settings[f"noise-seed-list-{dset}"] = " ".join(map(str, noise_seed_dset_list))

    print(f"noise_seed_list_dict = {noise_seed_list_dict}")
    noise_seed_all_list = ['"' + " ".join(map(str,noise_seed_list_dict[dset])) + '"' for dset in dset_list]
    print(f"noise_seed_all_list = {noise_seed_all_list}")
    noise_seed_all_arr = "(" + " ".join(noise_seed_all_list) + ")"
    common_settings[f"noise-seed-all-arr"] = noise_seed_all_arr
    print(f"noise_seed_all_arr = {noise_seed_all_arr}")

    # scripts_task_pipeline_fp = os.path.join(rlc_repo_dir, common_settings["scripts-task-pipeline-fp"])

    def process_block_settings(in_settings: dict):
        """Apply common_settings to the input settings dict
        then propagate any fields as necessary
        """
        nonlocal common_settings
        tmp_settings = apply_replacements_to_dict(
            in_dict=map_dict_vals_to_str(in_settings),
            replacement_dict=common_settings,
            cleanup=False,
        )
        out_settings = propagate_replacements(
            replacement_dict=map_dict_vals_to_str(tmp_settings),
            cleanup=False
        )
        return out_settings

    ### Block-wise settings ###
    train_fynet_settings = {
        # Architecture
        "n-cnn-1d": 3,
        "n-cnn-channels-1d": 24,
        "kernel-size-1d": 40,
        "n-cnn-2d": "3",
        "n-cnn-channels-2d": "24",
        # "n-cnn-2d": "3, 4",
        # "n-cnn-channels-2d": "24, 36",
        "kernel-size-2d": 5,

        # Optimization (note; these are different settings from before...)
        "batch-size": 16,
        "lr-init": "3e-4",
        "eta-min": "1e-5",
        "eta-min-format-rule": '"1e-5"',
        "init-mode": "he-normal",
        "weight-decay-base": "1e-3",
        "freq-lvl": "<<freq-idx>>",
        "seed": 29384,

        # copy/pasted
        "level-base-name": "train_fynet_f<<freq-idx>>_for_<<whole-run-name>>",
        "level-type": "fynet",
        "fynet-model-name-format": (
            "{run_date}"
            "_<<level-base-name>>"
            "_targets_{tt}_ntrain_{ntrain}_noise_{ntsr}_nus_{nus}"
            "_arch_{nlc1d}_{nlc2d}_{ncc1d}_{ncc2d}_{k1d}_{k2d}"
            # "_arch_{ncc1d}_{ncc2d}_{k1d}_{k2d}"
            "_opt_{lr}_{wd}_{im}"
        ),
        "job-name-format": (
            "arch_{nlc1d}_{nlc2d}_{ncc1d}_{ncc2d}_{k1d}_{k2d}_"
            # "arch_{ncc1d}_{ncc2d}_{k1d}_{k2d}_"
            "opt_{lr}_{wd}_{im};"
            "{run_date}_<<level-base-name=eval_fynet>>"
            "_targets_{tt}_ntrain_{ntrain}_noise_{ntsr}_nus_{nus}_..."
        ),


        # Slurm stuff
        "logs-rel-dir": "logs/mmg_pipeline/train_fynet",
        "jobs-rel-dir": "jobs/mmg_pipeline/train_fynet",

        # new stuff
        "model-base-name": "train_fynet_f<<freq-idx>>_for_<<whole-run-name>>",
        # "output-pred-scobj-dir": "<<predictions-rel-dir>>/<<run-date>>_dummy_f<<freq-idx>>_for_<<whole-run-name>>",
        "block-badger-fp": "<<scripts-dir>>/f<<freq-idx>>_badger_train_fynet.yaml",

        # Train targets
        "train_targets": "original",
        "eval_targets": "original",

    }
    train_fynet_settings = process_block_settings(train_fynet_settings)

    eval_fynet_settings = {
        "model-base-name": train_fynet_settings["model-base-name"],
        "block-badger-fp": "<<scripts-dir>>/f<<freq-idx>>_badger_eval_fynet.yaml",
        "output-pred-scobj-dir": (
            # "<<predictions-rel-dir>>/<<output-common-name-format>>"
            "<<predictions-rel-dir>>/"
            "<<run-date>>_eval_<<eval-targets>>_train_<<train-targets>>_fynet_f<<freq-idx>>_for_<<whole-run-name>>"
            # "<<predictions-rel-dir>>/<<run-date>>_eval_SOMETHING_train_SOMETHING_"
            # "fynet_f<<freq-idx>>_for_<<whole-run-name>>"
        ),

        # "central-results-fp": "<<central-run-dir>>/train_fynet_f<<freq-idx>>_summary.yaml", # hope this works...
        # "central-results-fp": "<<central-run-dir>>/summary.yaml", # seems to work

        # Things copied from the mpsr pipeline
        "level-type": "fynet",
        "level-base-name": "eval_fynet_f<<freq-idx>>_for_<<whole-run-name>>",
        "delete-unused-models": "false",
        "eval-batch-size": 100,
        "train_targets": MREF_TARGETS,
        "eval_targets": MREF_TARGETS,
        "output-common-name-format": "{run_date}_eval_{et}_train_{tt}_fynet_f<<freq-idx>>_for_<<whole-run-name>>",
        "results-file-pattern-format": (
            "{model_date}_<<model-base-name>>"
            "_targets_{tt}_ntrain_<<num-train>>_noise_{ntsr}_nus_<<nu-sf>>"
            "_arch_*_opt_*.txt"
        ),

        # Running settings
        "samples-per-chunk": shard_size,
        "eval-seed": "1001",
        "mem": "50G",
        "time-limit": "1:00:00",
        "logs-rel-dir": "logs/mmg_pipeline/eval_fynet",
        "jobs-rel-dir": "jobs/mmg_pipeline/eval_fynet",
        # This hack is meant to use bash to figure out the conditional since
        # I don't think that badger handles that...
        "field-name": "$(if [ {tt} = smoothed ]; then echo val_rel_l2; else echo val_final_rel_l2; fi)",
        "selection-mode": "min",
        "verbosity-level": "1",
    }
    eval_fynet_settings = process_block_settings(eval_fynet_settings)

    train_mrefblock_settings = {
        "level-type": "mref",
        "level-base-name": "train_mref_f<<fi>>_for_<<whole-run-name>>", # NOTE this drops <<prev-level-type>>
        "use-pred-d-mh": "false", # Set to false for MFISNet-Refinement mode
        "train-targets": MREF_TARGETS,
        "eval-targets":  MREF_TARGETS,
        "fi": "<<freq-idx>>",
        "model-base-name": "train_mref_f<<freq-idx>>_for_<<whole-run-name>>",

        # I/O stuff
        # "predictions-input-name": "<<prev-output-meas-dir>>",
        "predictions-input-name": "<<last-output-pred-scobj-dir>>",
        "block-badger-fp": "<<scripts-dir>>/f<<freq-idx>>_badger_train_mref.yaml",

        "field-name": "$(if [ {tt} = smoothed ]; then echo val_rel_l2; else echo val_final_rel_l2; fi)",
        "selection-mode": "min",
        "verbosity-level": "1",

        "mpsr-model-name-format": (
            "{run_date}"
            "_<<level-base-name>>"
            "_targets_{tt}_ntrain_{ntrain}_noise_{ntsr}_nus_{nus}"
            "_arch_{nlc1d}_{nlc2d}_{ncc1d}_{ncc2d}_{k1d}_{k2d}_{sc1pf}_{uc2d}_{neco}"
            "_opt_{lr}_{wd}_{im}_{lrdf}"
        ),
        "job-name-format": (
            "arch_{nlc1d}_{nlc2d}_{ncc1d}_{ncc2d}_{k1d}_{k2d}_{sc1pf}_{uc2d}_{neco}"
            "_opt_{lr}_{wd}_{im}_{lrdf};"
            "{run_date}_<<level-base-name>>"
            "_targets_{tt}_ntrain_{ntrain}_noise_{ntsr}_nus_{nus}_..."
        ),

        # Running settings
        "samples-per-chunk": shard_size,
        "mem": "50G",
        "time-limit": "4:00:00",
        "logs-rel-dir": "logs/mmg_pipeline/train_mref",
        "jobs-rel-dir": "jobs/mmg_pipeline/train_mref",

        # Architecture settings
        "set-c1d-per-freq": '"false"',
        "use-cnns-2d": '"both"', # matches the MFISNet-Refinement paper
        "n-cnn-1d": "3",
        "n-cnn-2d": "3",
        # "n-cnn-2d": "3,4",
        "n-cnn-channels-1d": "24",
        "n-cnn-channels-2d": "24",
        # "n-cnn-channels-2d": "24,36",
        "kernel-size-1d": "40",
        "kernel-size-2d": "5", # "7" worked better in the (4,16) setting but unclear here...
        "embedding-mode": "none", # drop the embedding layers
        "n-emb-channels-out": "0",

        # Optimization/training settings
        "seed": 35675,
        # "num-epochs": NUM_EPOCHS,
        "batch-size": 16,
        "lr-init": '"3e-4"',
        "eta-min-format-rule": '"1e-5"', # alternately could tether to lr-init as {lr_init}
        "lr-decrease-factor": '"1"',
        "weight-decay": '"1e-3"', # '"1e-3", "5e-3"'
        "init-mode": '"he-normal"',
    }
    train_mrefblock_settings = process_block_settings(train_mrefblock_settings)

    eval_mrefblock_settings = {
        "model-base-name": train_mrefblock_settings["model-base-name"],
        "level-type": "mref",
        "level-base-name": "eval_mref_f<<fi>>_for_<<whole-run-name>>",
        "fi": "<<freq-idx>>",
        "block-badger-fp": "<<scripts-dir>>/f<<freq-idx>>_badger_eval_mref.yaml",

        "use-pred-d-mh": "false", # MFISNet-Refinement mode
        "train-targets": MREF_TARGETS,
        "eval-targets":  MREF_TARGETS,
        "field-name": (
            "$(if [ {tt} = smoothed ]; then echo eval_rel_l2; "
            "else echo eval_final_rel_l2; fi)"
        ),
        # "predictions-input-name": "<<prev-output-meas-dir>>" # ??
        "predictions-input-name": "<<last-output-pred-scobj-dir>>",
        # "output-pred-scobj-dir"
        "output-pred-scobj-dir": (
            "<<predictions-rel-dir>>/<<run-date>>_"
            "eval_<<eval-targets>>_train_<<train-targets>>_"
            "mref_f<<freq-idx>>_for_<<whole-run-name>>"
        ),
        # "predictions-output-name-format": "<<output-common-name-format>>",
        "predictions-output-name-format": "<<output-pred-scobj-dir>>",
        "output-common-name-format": (
            "{run_date}_eval_{et}_train_{tt}_mref"
            "_f<<fi>>_for_<<whole-run-name>>"
        ),

        "results-file-pattern-format": (
            "{model_date}_<<model-base-name>>"
            "_targets_{tt}_ntrain_<<num-train>>_noise_{ntsr}_nus_<<nu-sf>>"
            "_arch_*_opt_*.txt"
        ),

        "selection-mode": "min",
        "eval-batch-size": 100,
        # "eval-batch-size": "<<log-batch-size>>",
        # "delete-unused-models": "true",
        "delete-unused-models": "false", # maybe switch this at some point

        # Running settings
        "eval-seed": "1001",
        "samples-per-chunk": shard_size,
        "mem": "40G",
        "time-limit": "4:00:00",
        "logs-rel-dir": "logs/mmg_pipeline/eval_mref",
        "jobs-rel-dir": "jobs/mmg_pipeline/eval_mref",
    }

    eval_mrefblock_settings = process_block_settings(eval_mrefblock_settings)

    train_mref_pipe_settings = {
        "freq-idx": "e", # Treat the index as e...
        "fi":       "e",
        "level-type":   "mrefpipe",
        "level-base-name": "train_e2e_mrefpipe_for_<<whole-run-name>>",
        "block-badger-fp": "<<scripts-dir>>/f<<freq-idx>>_badger_train_e2e_mrefpipe.yaml",

        "train-targets": "original",
        "eval-targets": "original",
        "use-pred-d-mh": "false", # MFISNet-Refinement mode
        "selection-mode": "min",
        "selection-field": (
            "'$(if [ {tt} = smoothed ]; then echo eval_rel_l2; "
            "else echo eval_final_rel_l2; fi)'"
        ),

        "d-rs-loss-weight": "0.0",
        "e2e-date":      RUN_DATE_VAL,
        "model-date":    RUN_DATE_VAL,

        # Data/logging setup
        "e2e-num-train": NUM_TRAIN,
        "e2e-num-val":   NUM_VAL,
        "e2e-num-test":  NUM_TEST,
        "e2e-log-train-subset-frac": "1",
        "num-batches-per-log": "'0'",
        "num-epochs-per-log":  "5",

        # Optimization parameters
        "noise-level": "<<noise-level>>",
        "jax-mem-alloc-mb": 0,
        "e2e-batch-size": 16,
        "e2e-num-epochs": NUM_E2E_EPOCHS,

        "e2e-lr-init-base": "1e-4",
        "e2e-eta-min-format-rule": "1e-5",
        "e2e-weight-decay-base": "1e-3",


        # Naming for the model
        "base-model-name": "<<whole-run-name>>",
        "e2e-common-name": "train_e2e_<<whole-run-name>>",
        # "e2e-model-name-details-format": "opt_{ne}_{bs}_{lr}_{wd}_ntsr_{ntsr}",
        # "e2e-model-name-format": "{prefix}_{details}",
        # "e2e-model-fp-format":   "e2e_{0}"

        # I/O stuff
        "in-central-results-fp":    "<<central-results-fp>>",
        "out-results-yaml-format":  "<<results-rel-dir>>/{e2e_model_name}.yaml",
        "out-train-results-format": "<<results-rel-dir>>/{e2e_model_name}.txt",
        "out-model-dir-format":     "<<models-rel-dir>>/{e2e_model_name}",

        "save-all-model-weights":   "false",

        # Slurm stuff
        "mem": "50G",
        "time-limit": "8:00:00",
        "logs-rel-dir": "logs/mmg_pipeline/train_e2e_mref",
        "jobs-rel-dir": "jobs/mmg_pipeline/train_e2e_mref",

        # HPS settings that I probably won't use
        "pde-solver-type": "hps",
        "hps-l": "3",
        "hps-p": "16",
    }
    train_mref_pipe_settings = process_block_settings(train_mref_pipe_settings)

    eval_mref_pipe_settings = {
        "freq-idx": "e", # Treat the index as e...
        "fi":       "e",
        "level-type":   "mrefpipe",
        "level-base-name": "eval_e2e_mrefpipe_for_<<whole-run-name>>",
        "block-badger-fp": "<<scripts-dir>>/f<<freq-idx>>_badger_eval_e2e_mrefpipe.yaml",

        "train-targets": "original",
        "eval-targets": "original",
        "use-pred-d-mh": "false", # MFISNet-Refinement mode
        "selection-mode": "min",
        "selection-field": (
            "'$(if [ {tt} = smoothed ]; then echo eval_rel_l2; "
            "else echo eval_final_rel_l2; fi)'"
        ),

        "d-rs-loss-weight": "0.0",
        "e2e-date":      RUN_DATE_VAL,
        "model-date":    RUN_DATE_VAL,

        # Data/logging setup
        "e2e-dsets": "train val test",
        "e2e-dsets-num-samples": " ".join(map(str, [NUM_TRAIN, NUM_VAL, NUM_TEST])),
        # "e2e-dsets-noise-seeds": " ".join(map(str, [NUM_TRAIN, NUM_VAL, NUM_TEST])),
        # "e2e-num-train": NUM_TRAIN,
        # "e2e-num-val":   NUM_VAL,
        # "e2e-num-test":  NUM_TEST,
        # "e2e-log-train-subset-frac": "1",
        "num-batches-per-log": "'0'",
        "num-epochs-per-log":  "5",

        # Runtime parameters
        "noise-level": "<<noise-level>>",
        "jax-mem-alloc-mb": 0,
        "e2e-eval-batch-size": 100,
        "e2e-eval-seed": "1001",

        # Naming for the model
        # "base-model-name": "<<whole-run-name>>",
        "e2e-common-name": "eval_e2e_<<whole-run-name>>",
        "e2e-model-name": "<<e2e-date>>_<<e2e-common-name>>",

        # I/O stuff
        "output-dset-summary-fp": "<<central-run-dir>>/e2e_summary_${dset}.yaml",
        "output-central-summary-fp": "<<central-run-dir>>/e2e_summary.yaml",
        "output-pred-scobj-dir": (
            "<<predictions-rel-dir>>/<<run-date>>_"
            "eval_<<eval-targets>>_train_<<train-targets>>_"
            "e2e_for_<<whole-run-name>>"
        ),

        # Hyperparameter selection, if relevant...
        "in-results-pattern": (
            "<<e2e-date>>_train_e2e_<<whole-run-name>>"
            "_opt_*_ntsr_<<noise-level>>.yaml"
        ),
        "hyperparam-summary-fp": (
            "<<central-run-dir>>/e2e_summary.yaml"
        ),
        "centralize-models": "true",
        "central-model-fp-format": "e2e_{0}",

        # Slurm stuff
        "mem": "80G",
        "time-limit": "8:00:00",
        "logs-rel-dir": "logs/mmg_pipeline/eval_e2e_mref",
        "jobs-rel-dir": "jobs/mmg_pipeline/eval_e2e_mref",

        # HPS settings that I probably won't use
        "use-pde-args": "true",
        "pde-solver-type": "hps",
        "hps-l": "3",
        "hps-p": "16",
    }
    eval_mref_pipe_settings = process_block_settings(eval_mref_pipe_settings)


    ### Print settings ###
    if verbosity >= VLVL_INIT_CONFIG:
        indent = 2
        common_settings_str = pretty_dict_to_str(
            common_settings,
            dict_label="Common settings",
            indent_width=indent,
        )
        print(f"Received the following settings...")
        print(common_settings_str)
        settings_list = [
            (train_fynet_settings,     "Train FYNet settings"),
            (eval_fynet_settings,      "Eval FYNet settings"),
            (train_mrefblock_settings, "Train MRef settings"),
            (eval_mrefblock_settings,  "Eval MRef settings"),
            (train_mref_pipe_settings, "Train MRef pipeline settings"),
            (eval_mref_pipe_settings,  "Eval MRef pipeline settings"),
        ]
        for settings, name in settings_list:
            print(pretty_dict_to_str(settings, name, indent_width=indent))


    ### Set up the blocks and pipeline ###
    train_fynet     = TrainFYNet("train-fynet", train_fynet_settings)
    eval_fynet      = EvalFYNet("eval-fynet", eval_fynet_settings)
    train_mrefblock = TrainMRefBlock("train-mrefblock", train_mrefblock_settings)
    eval_mrefblock  = EvalMRefBlock("eval-mrefblock", eval_mrefblock_settings)
    train_mrefblock_last = TrainMRefBlock(
        "train-mrefblock",
        {
            **train_mrefblock_settings,
            "train-targets": "original",
            "eval-targets": "original",
        },
    )
    eval_mrefblock_last = EvalMRefBlock(
        "eval-mrefblock",
        {
            **eval_mrefblock_settings,
            "train-targets": "original",
            "eval-targets": "original",
        },
    )
    train_mref_pipe = TrainMRefPipeline(
        "train-e2e-mref-pipe",
        train_mref_pipe_settings,
    )
    eval_mref_pipe = EvalMRefPipeline(
        "eval-e2e-mref-pipe",
        eval_mref_pipe_settings,
    )

    init_block = FrequencyBlock("f1", [train_fynet, eval_fynet])
    # iter_block = FrequencyBlock("fi", [run_mmg_solver, train_mmgublock])
    iter_block = FrequencyBlock("fi", [train_mrefblock, eval_mrefblock])
    iter_block_last = FrequencyBlock("fi", [train_mrefblock_last, eval_mrefblock_last])

    e2e_block = SequentialTasks("fe", [train_mref_pipe, eval_mref_pipe])
    e2e_block.freq_idx = "e"

    # Manual version
    # pipeline_tasks = [
    #     copy_and_name("f1", init_block),
    #     copy_and_name("f2", iter_block),
    #     copy_and_name("f3", iter_block_last),
    # ]
    pipeline_tasks = [
        *setup_basic_pipeline_tasks(
            init_block, iter_block, Nk-2,
        ),
        copy_and_name(f"f{Nk}", iter_block_last),
        copy_and_name("fe", e2e_block),
    ]
    main_pipeline = MMGTaskPipeline(
        "MMG Pipeline",
        pipeline_tasks,
        common_settings,
    )

    # Ensure that the last training block always targets original objects...
    # May need to change this when I add e2e training
    # Do this for both training and evaluation blocks...
    # main_pipeline.task_list[-1].task_list[-2].settings["train-targets"] = "original"
    # main_pipeline.task_list[-1].task_list[-2].settings["eval-targets"]  = "original"
    # main_pipeline.task_list[-1].task_list[-1].settings["train-targets"] = "original"
    # main_pipeline.task_list[-1].task_list[-1].settings["eval-targets"]  = "original"

    if verbosity >= 1:
        print(f"~~~ Pipeline Outline ~~~")
        print(str(main_pipeline))
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~")

    ### Set up context and generate scripts ###
    context = {
        "data-input-nus": data_input_nus,
        "freq-idx": 1,
        "nu-sf": str_nu_list[0],
        # "last-output-pred-scobj-dir": "rlc_data/predictions/mpsr/2025-07-15_eval_smoothed_train_smoothed_mref_f4_for_mref_Nk_8_(1,2,3,4,5,6,7,8)_ntr_10000_hps",
        # "last-output-pred-mmg-dir": "rlc_data/mmg_pipeline/predictions/2025-08-22_mmg_from_mref_f4_ntr_10000",
        # Now we use FYNet so these *shouldn't* be needed...
        "last-output-pred-scobj-dir": "ERROR",
        "last-output-pred-mmg-dir": "ERROR",
    }
    context_init = {**context}

    if generate_scripts:
        context_after = main_pipeline.gen_scripts(context, settings=dict(), verbosity=verbosity)
    else:
        context_after = {**context_init}

    save_object = {
        "pipeline": main_pipeline,
        "context-init": context_init,
        "context-after": context_after, # in case we need it??
        "common-settings": common_settings,
        "train-fynet-settings": train_fynet_settings,
        # "run-mmg-solver-settings": run_mmg_solver_settings,
        # "train-mmgublock-settings": train_mmgublock_settings,
        "train-mrefblock-settings": train_mrefblock_settings,
    }

    return save_object

def generate_pipeline(verbosity: int = 3, **kwargs):
    """This is the main function called by
    generate_pipeline_scripts.py
    """
    save_object = setup_pipeline(verbosity=verbosity, generate_scripts=True, **kwargs)
    scripts_task_pipeline_fp = save_object["common-settings"]["scripts-task-pipeline-fp"]

    # Pickle the save_object dictionary; intended for immediate use
    print(f"Saving pipeline object to... {scripts_task_pipeline_fp}")
    with open(scripts_task_pipeline_fp, "wb") as pipeline_file:
        pickle.dump(save_object, pipeline_file, pickle.DEFAULT_PROTOCOL)

def run_pipeline(command_str: str, use_pickled_pipeline: bool=False, verbosity: int=3, **kwargs):
    """Runs the pipeline passed in to it
    i.e., submit jobs to slurm
    Can optionally grab the pickled pipeline -- this lets us to avoid over-writing existing scripts,
    in case there were any manual changes in the meantime
    """
    # does not currently allow us to control the sub-sections...
    # main_pipeline.submit_scripts(sleep_time=0.5, verbosity=verbosity)
    save_object = setup_pipeline(
        verbosity=max(0,verbosity-2),
        # verbosity=0,
        generate_scripts=(not use_pickled_pipeline),
        **kwargs
    )

    scripts_task_pipeline_fp = save_object["common-settings"]["scripts-task-pipeline-fp"]
    context = save_object["context-init"]

    # Get the pipeline either from save_object or from the pickle file
    do_generate_scripts = not use_pickled_pipeline
    if use_pickled_pipeline:
        try:
            with open(scripts_task_pipeline_fp, "rb") as pipeline_file:
                new_save_object = pickle.load(pipeline_file)
                # overwrite the save object thing when loading from pipeline
                save_object = new_save_object
        except:
            print(f"Failed to load pickled pipeline from {scripts_task_pipeline_fp}; falling back to the newly-generated version...")
            do_generate_scripts = True
    pipeline = save_object["pipeline"]

    # If needed, generate the scripts
    if do_generate_scripts:
        pipeline.gen_scripts(context, settings=dict(), verbosity=max(0,verbosity-2))

    dry_run    = kwargs.get("dry_run", False)
    sleep_time = kwargs.get("sleep_time", 0.5)
    pipeline.submit_scripts(
        command_str,
        sleep_time=sleep_time,
        verbosity=verbosity,
        dry_run=dry_run,
    )
