# generate_pipeline_scripts.py calls generate_pipeline
# submission occurs separately and involves calling run_pipeline
# This is a variant of the pipeline specifically for evaluating
# the MMG pipeline
#
# Evaluation-only (post-e2e) config for the mini_mfisnet_refinement_smoothed.py
# run, targeting OOD contrast 1.5.

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
    MMGTaskPipeline,
    EvalMRefPipeline,
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
STR_NU_LIST_VAL    = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
WANDB_PROJECT_VAL  = "none"

# Incoming model info -- matches mini_mfisnet_refinement_smoothed.py's own
# RUN_DATE_VAL/WHOLE_RUN_NAME_VAL (the e2e-fine-tuned model we're evaluating)
MODEL_DATE_VAL = "2026-08-20"
INCOMING_WHOLE_RUN_NAME_VAL = "mini_mref_smoothed"
INC_NOISE_LEVEL = "0.0"
NUM_TRAIN = 10
NUM_VAL   = 10
NUM_TEST  = 10

RUN_DATE_VAL = "2026-08-21"
WHOLE_RUN_NAME_VAL = "eval_mref_post_e2e_contrast_1.5_mini_mref_smoothed"

# Eval data info
EVAL_DATASET = os.path.join(load_system_setup()["file-paths"]["ood-dataset"], "2025-09-23_ood_dataset_contrast_1.5")
EVAL_NUM_TRAIN = 0
EVAL_NUM_VAL   = 0
EVAL_NUM_TEST  = 10
EVAL_DSET_LIST = ["test"]
EVAL_DSET_NUMS = list(map(str, [EVAL_NUM_TEST]))
NOISE_LEVEL = "0.0"


def setup_pipeline(verbosity: int = 3, generate_scripts: bool=True, **kwargs) -> dict:
    """Shared setup function between generate_pipeline and run_pipeline
    (shared so it's easier to fetch the pipeline when running the pipeline, but
    we also don't need to re-generate the scripts)
    """
    global RUN_DATE_VAL, WHOLE_RUN_NAME, STR_NU_LIST_VAL, WANDB_PROJECT_VAL
    global MODEL_DATE_VAL, INCOMING_WHOLE_RUN_NAME_VAL, NOISE_LEVEL
    global NUM_TRAIN, NUM_VAL, NUM_TEST, EVAL_DSET_LIST
    global EVAL_NUM_TRAIN, EVAL_NUM_VAL, EVAL_NUM_TEST, EVAL_DSET_NUMS
    global INC_NOISE_LEVEL, INCOMING_WHOLE_RUN_NAME_VAL, EVAL_DATASET
    ### Common settings ###
    system_setup = load_system_setup()
    rlc_repo_dir = system_setup["file-paths"]["repo-dir"]
    rlc_data_dir = system_setup["file-paths"].get("data-rel-dir", "rlc_data")
    standard_settings = get_standard_mmg_settings(rlc_repo_dir, rlc_data_dir)
    # Sets the default templates and jobs/logs directories

    str_nu_list = STR_NU_LIST_VAL
    data_input_nus = "(" + ",".join(str_nu_list) + ")"
    Nk = len(str_nu_list)
    shard_size = 250

    # Settings for the pipeline I'm trying to evaluate
    incoming_pipeline_settings = {
        "inc-model-date": MODEL_DATE_VAL,
        "inc-whole-run-name": INCOMING_WHOLE_RUN_NAME_VAL,
        "inc-fynet-base-name":     "train_fynet_f<<freq-idx>>_for_<<inc-whole-run-name>>",
        "inc-mref-base-name":      "train_mref_f<<freq-idx>>_for_<<inc-whole-run-name>>",
        "inc-train-targets":       "smoothed",
        "inc-eval-targets":        "smoothed",
        "inc-noise-level":         INC_NOISE_LEVEL,
        "inc-num-train":           NUM_TRAIN,
        "inc-num-val":             NUM_VAL,
        "inc-num-test":            NUM_TEST,

        # Directory stuff
        "inc-central-rel-dir":   f"{rlc_data_dir}/mmg_pipeline/central_run_info",
        "inc-central-run-dir":   "<<inc-central-rel-dir>>/<<inc-model-date>>_<<inc-whole-run-name>>",
        "inc-central-model-dir": "<<inc-central-run-dir>>/models/",
    }
    incoming_pipeline_settings = propagate_replacements(
        map_dict_vals_to_str(incoming_pipeline_settings),
        cleanup=False,
    )

    common_settings = {
        **standard_settings,
        **incoming_pipeline_settings,
        "run-date": RUN_DATE_VAL,
        "whole-run-name": WHOLE_RUN_NAME_VAL,
        "pipeline-scripts-rel-dir": "pipeline_scripts",
        "scripts-dir":              "<<pipeline-scripts-rel-dir>>/<<run-date>>_<<whole-run-name>>",
        "central-run-dir":          "<<central-rel-dir>>/<<run-date>>_<<whole-run-name>>",
        "central-model-dir":        "<<central-run-dir>>/models/",
        "central-model-format":     "model_params_<<freq-idx>>.pickle",

        # Meta.. where to save this task pipeline?
        "scripts-task-pipeline-fp": "<<scripts-dir>>/task_pipeline.pickle",

        # General stuff
        "dataset-rel-dir":          EVAL_DATASET, # for FYNet/base
        "ref-dataset-rel-dir":      EVAL_DATASET, # for MMG/refinement-based stuff
        "ref-dataset-dir":          EVAL_DATASET, # for MMG/refinement-based stuff
        "dataset-dir":              EVAL_DATASET, # just in case

        # Slurm stuff
        "exclude-node": '""',
        "wandb-project": WANDB_PROJECT_VAL,

        # Data settings
        "data-input-nus": data_input_nus,
        "str_nu_list": str_nu_list,
        "nu_list": [float(str_nu) for str_nu in str_nu_list],
        "Nk": Nk,
        "num-freqs": Nk,
        "num-train":      NUM_TRAIN,
        "num-val":        NUM_VAL,
        "num-test":       NUM_TEST,
        "noise-level":    NOISE_LEVEL,
        "eval-num-train": EVAL_NUM_TRAIN,
        "eval-num-val":   EVAL_NUM_VAL,
        "eval-num-test":  EVAL_NUM_TEST,
        "eval-dset-list": " ".join(EVAL_DSET_LIST),
        "eval-dset-nums": " ".join(EVAL_DSET_NUMS),
        # The e2e fine-tuning stage always targets original (unsmoothed)
        # objects, so we evaluate against original targets here too --
        # matches train_mref_pipe_settings/eval_mref_pipe_settings in
        # mini_mfisnet_refinement_smoothed.py
        "train-targets": "original",
        "eval-targets":  "original",

        # noise seed and stuff
        "use-noise-seed": "true",
        "noise-seed-base-train": 10000000,
        "noise-seed-base-val":   20000000,
        "noise-seed-base-test":  30000000,
        "noise-seed-train": "10000*<<nu-sf>>+3221+<<noise-seed-base-train>>",
        "noise-seed-val":   "10000*<<nu-sf>>+3221+<<noise-seed-base-val>>",
        "noise-seed-test":  "10000*<<nu-sf>>+3221+<<noise-seed-base-test>>",
        "noise-seed-format-rule": "{noise_seed_base}+{input_label}",

        # General optimization stuff
        "lr-decrease-factor": 1,
        "output-pred-shard-size": shard_size,
    }
    # Replace fields like <<field>> if they were defined by other keys in the dictionary
    common_settings = propagate_replacements(
        map_dict_vals_to_str(common_settings),
        cleanup=False,
    )
    scripts_task_pipeline_fp = os.path.join(rlc_repo_dir, common_settings["scripts-task-pipeline-fp"])

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
    dset_list = EVAL_DSET_LIST

    eval_mref_pipe_settings = {
        **incoming_pipeline_settings,
        "freq-idx": "e", # Treat the index as e...
        "fi":       "e",
        "level-type":   "mrefpipe",
        "level-base-name": "eval_e2e_mrefpipe_for_<<whole-run-name>>",
        "block-badger-fp": "<<scripts-dir>>/f<<freq-idx>>_badger_eval_e2e_mrefpipe.yaml",

        "model-date":    MODEL_DATE_VAL,
        "eval-date":     RUN_DATE_VAL,

        "train-targets": "original",
        "eval-targets": "original",
        "use-pred-d-mh": "false", # MFISNet-Refinement mode

        # Data/logging setup
        "e2e-dsets": " ".join(EVAL_DSET_LIST),
        "e2e-dsets-num-samples": "<<eval-dset-nums>>",

        "noise-level": "<<noise-level>>",
        "jax-mem-alloc-mb": 0,
        "e2e-eval-batch-size": 100,
        "e2e-eval-seed": "1001",

        # Name for the model, then I/O stuff
        "e2e-common-name": "<<whole-run-name>>",
        "e2e-model-name": "<<eval-date>>_<<e2e-common-name>>",

        "output-pred-scobj-dir": (
            "<<predictions-rel-dir>>/<<run-date>>_"
            "eval_<<eval-targets>>_train_<<train-targets>>_"
            "e2e_for_<<whole-run-name>>"
        ),
        "output-dset-summary-fp": "<<central-run-dir>>/e2e_summary_${dset}.yaml",
        "output-central-summary-fp": "<<central-run-dir>>/e2e_summary.yaml",
        "hyperparam-summary-fp": (
            "<<inc-central-run-dir>>/e2e_summary.yaml"
        ),

        "num-train": "<<eval-num-train>>",
        "num-val":   "<<eval-num-val>>",
        "num-test":  "<<eval-num-test>>",

        # Running settings
        "samples-per-chunk": shard_size,
        "eval-seed": "1001",
        "mem": "80G",
        "time-limit": "1:00:00",
        "logs-rel-dir": "logs/mmg_pipeline/eval_e2e_mref",
        "jobs-rel-dir": "jobs/mmg_pipeline/eval_e2e_mref",

        # Selection stuff
        # Most of these should be ignored due to select-hyperparameters being set to false...
        "select-hyperparameters": "false",
        "centralize-models": "false",
        "selection-mode": "min",
        "selection-field": (
            "'$(if [ {tt} = smoothed ]; then echo eval_rel_l2; "
            "else echo eval_final_rel_l2; fi)'"
        ),
        "field-name": "$(if [ {tt} = smoothed ]; then echo val_rel_l2; else echo val_final_rel_l2; fi)",
        "verbosity-level": "1",

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
            (eval_mref_pipe_settings, "Eval MRef Model-Pipeline settings"),
        ]
        for settings, name in settings_list:
            print(pretty_dict_to_str(settings, name, indent_width=indent))


    ### Set up the blocks and pipeline ###
    eval_mref_pipe = EvalMRefPipeline("eval-mref-pipe", eval_mref_pipe_settings)
    e2e_block = SequentialTasks("fe", [eval_mref_pipe])
    e2e_block.freq_idx = "e"

    pipeline_tasks = [
        copy_and_name("fe", e2e_block),
    ]

    main_pipeline = MMGTaskPipeline(
        "MMG Pipeline",
        pipeline_tasks,
        {**common_settings, **incoming_pipeline_settings},
    )


    if verbosity >= 1:
        print(f"~~~ Pipeline Outline ~~~")
        print(str(main_pipeline))
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~")

    ### Set up context and generate scripts ###
    context = {
        "data-input-nus": data_input_nus,
        "freq-idx": "e",
        "nu-sf": str_nu_list[0],
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
        "eval-mref-pipe-settings": eval_mref_pipe_settings,
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
    save_object = setup_pipeline(
        verbosity=max(0,verbosity-2),
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
