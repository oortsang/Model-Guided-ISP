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
    RunMMGSolver,
    MMGTaskPipeline,
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
RUN_DATE_VAL = "2025-10-09"

# STR_NU_LIST_VAL    = ["1", "4.5", "8"]
STR_NU_LIST_VAL    = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
WHOLE_RUN_NAME_VAL = "mmgu_Nk_10_(1,2,3,4,5,6,7,8,9,10)_train_original_ntr_1000_inf_noise_0.20_epochs_300"
WANDB_PROJECT_VAL  = "2025-09-16_mmg_Nk_10_early_runs"

NUM_TRAIN = 1000
NUM_VAL   = 1000
NUM_TEST  = 1000
NUM_EPOCHS = 300
NOISE_LEVEL = "0.20"

def setup_pipeline(verbosity: int = 3, generate_scripts: bool=True, **kwargs) -> dict:
    """Shared setup function between generate_pipeline and run_pipeline
    (shared so it's easier to fetch the pipeline when running the pipeline, but
    we also don't need to re-generate the scripts)
    """
    global RUN_DATE_VAL, WHOLE_RUN_NAME, STR_NU_LIST_VAL, WANDB_PROJECT_VAL
    global NUM_TRAIN, NUM_VAL, NUM_TEST, NOISE_LEVEL
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
        "noise-seed-base-train": 10000000,
        "noise-seed-base-val":   20000000,
        "noise-seed-base-test":  30000000,
        "noise-seed-train": "10000*<<nu-sf>>+3221+<<noise-seed-base-train>>",
        "noise-seed-val":   "10000*<<nu-sf>>+3221+<<noise-seed-base-val>>",
        "noise-seed-test":  "10000*<<nu-sf>>+3221+<<noise-seed-base-test>>",
        "noise-seed-format-rule": "{noise_seed_base}+{input_label}",
        "n-epochs": NUM_EPOCHS,
        "n-epochs-per-log": 5,
        "log-train-subset-frac": 1,
        "train-targets": "original",
        "eval-targets":  "original",

        # General optimization stuff
        "lr-decrease-factor": 1,
        "output-pred-shard-size": 250,
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
    train_fynet_settings = {
        # Architecture
        "n-cnn-1d": 3,
        "n-cnn-channels-1d": 24,
        "kernel-size-1d": 40,
        "n-cnn-2d": 3,
        "n-cnn-channels-2d": 24,
        "kernel-size-2d": 5,
        # Optimization (note; these are different settings from before...)
        "batch-size": 16,
        "lr-init": "3e-4",
        "eta-min": "1e-5",
        "init-mode": "he-normal",
        "weight-decay-base": "1e-3",
        "n-cnn-channels-2d": 36,
        "freq-lvl": "<<freq-idx>>",
        "seed": 29384,

        # copy/pasted
        "level-base-name": "train_fynet_f<<freq-idx>>_for_<<whole-run-name>>",
        "level-type": "fynet",
        "fynet-model-name-format": (
            "{run_date}"
            "_<<level-base-name>>"
            "_targets_{tt}_ntrain_{ntrain}_noise_{ntsr}_nus_{nus}"
            "_arch_{ncc1d}_{ncc2d}_{k1d}_{k2d}"
            "_opt_{lr}_{wd}_{im}"
        ),
        "job-name-format": (
            "arch_{ncc1d}_{ncc2d}_{k1d}_{k2d}_"
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
            "<<predictions-rel-dir>>/"
            "<<run-date>>_eval_<<eval-targets>>_train_<<train-targets>>_fynet_f<<freq-idx>>_for_<<whole-run-name>>"
        ),
        "central-results-fp": "<<central-run-dir>>/summary.yaml", # hope this works...

        # Things copied from the mpsr pipeline
        "level-type": "fynet",
        "level-base-name": "eval_fynet_f<<freq-idx>>_for_<<whole-run-name>>",
        "delete-unused-models": "false",
        "eval-batch-size": 100,
        "train_targets": "original",
        "eval_targets": "original",
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
        # A bit of a hack in case of multiple train targets
        "field-name": "$(if [ {tt} = smoothed ]; then echo val_rel_l2; else echo val_final_rel_l2; fi)",
        "selection-mode": "min",
        "verbosity-level": "1",
    }
    eval_fynet_settings = process_block_settings(eval_fynet_settings)

    run_mmg_solver_settings = {
        "hps-l": 3,
        "hps-p": 16,
        "hps-comp-domain-factor": 1.1,
        "first-chunk-start": 0,
        "samples-per-chunk": shard_size,
        "write-every-n": 50,
        "noise-seed-format-rule": "{noise_seed_base}+{input_label}",

        # badger/io
        "output-name-format": "{output_dir}/{dset}_gammas_nu_{nu_sf}/gammas_{input_label}.h5",
        "output-mmg-rel-dir": "<<predictions-rel-dir>>/<<run-date>>_hps_mmg_f<<freq-idx>>_for_<<whole-run-name>>",
        "block-badger-fp": "<<scripts-dir>>/f<<freq-idx>>_badger_mmg_solver_<<dset>>.yaml",

        # slurm stuff
        "logs-rel-dir": "logs/mmg_pipeline/mmg_solver",
        "jobs-rel-dir": "jobs/mmg_pipeline/mmg_solver",
        "log-name-format": "<<run-date>>_mmg_solver_f<<freq-idx>>_for_<<whole-run-name>>_nu_{nu_sf}_dset_{dset}_chunk_{input_label}",
        "job-name-format": "mmg_solver_nu_{nu_sf}_dset_{dset}_chunk_{input_label};<<run-date>>_",
    }
    run_mmg_solver_settings = process_block_settings(run_mmg_solver_settings)

    dset_list =  ["train", "val", "test"]
    train_mmgublock_settings = {
        "cleanup": False,
        # Architecture
        "n-cnn-layers-2d": 4,
        "n-cnn-channels-2d": 36,
        "kernel-size-2d": 5,
        "learn-cnn-scale": "false",
        "init-cnn-scale": 1.0,

        # Optimization
        "batch-size": 16,
        "lr-init-base": "3e-4",
        "eta-min-base": "1e-5",
        "weight-decay-base": "1e-3",
        "freq-lvl": "<<freq-idx>>",
        "seed": 29384,
        # Misc.
        "selection-field": "val_cart_rel_l2",
        "selection-mode":  "min",
        "log-batch-size": 50,

        # Badger/script file location
        "block-badger-fp": "<<scripts-dir>>/f<<freq-idx>>_badger_train_mmgublock.yaml",
        # slurm stuff
        "logs-rel-dir": "logs/mmg_pipeline/train_mmgu",
        "jobs-rel-dir": "jobs/mmg_pipeline/train_mmgu",
        "log-name-format": "{model_name}",
        "job-name-format": "{model_hyperparams};<<run-date>>_{model_base_name}_{model_extras}_",

        ### Model names ###
        "model-base-name": "train_mmgu_f<<freq-idx>>_for_<<whole-run-name>>",
        "model-hyperparam-name-format": "arch_{ncl2}_{ncc2}_{k2d}_opt_{lr}_{wd}_{bs}",
        "model-extras-name-format": "targets_{tt}_ntrain_{ntrain}_noise_{ntsr}",
        "model-name-format": "<<run-date>>_{model_base_name}_{model_extras}_{model_hyperparams}",

        ### I/O ###
        "save-all-model-weights": "false",
        "central-summary-fp":   "<<central-run-dir>>/summary.yaml",
        "train-results-format": "<<results-rel-dir>>/{model_name}.txt",
        "model-weights-format": "<<models-rel-dir>>/{model_name}", # directory

        "central-model-dir": "<<central-run-dir>>/models/",
        "central-model-format": "model_params_<<freq-idx>>.pickle",

        "output-pred-shard-size": shard_size,
        "output-pred-save": "true",
        "output-pred-format": (
            "<<predictions-rel-dir>>/"
            "<<run-date>>_eval_{et}_train_{tt}_mmgu_f<<freq-idx>>_for_<<whole-run-name>>"
        ),
        # Only set block-eval-targets during runtime
        "output-pred-format-internal": (
            "<<predictions-rel-dir>>/"
            "<<run-date>>_eval_<<eval-targets>>_train_<<train-targets>>_mmgu_f<<freq-idx>>_for_<<whole-run-name>>"
        ),
    }
    train_mmgublock_settings = process_block_settings(train_mmgublock_settings)

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
            (train_fynet_settings, "Train FYNet settings"),
            (eval_fynet_settings, "Eval FYNet settings"),
            (train_mmgublock_settings, "Train MMGUBlock settings"),
            (run_mmg_solver_settings, f"Run MMGSolver settings (dsets: {dset_list})"),
        ]
        for settings, name in settings_list:
            print(pretty_dict_to_str(settings, name, indent_width=indent))


    ### Set up the blocks and pipeline ###
    train_fynet     = TrainFYNet("train-fynet", train_fynet_settings)
    eval_fynet      = EvalFYNet("eval-fynet", eval_fynet_settings)
    run_mmg_solver  = RunMMGSolver("run-mmg-solver", dset_list, run_mmg_solver_settings)
    train_mmgublock = TrainMMGUBlock("train-mmgublock", train_mmgublock_settings)

    init_block = FrequencyBlock("f1", [train_fynet, eval_fynet])
    iter_block = FrequencyBlock("fi", [run_mmg_solver, train_mmgublock])

    # Manual version
    # pipeline_tasks = [
    #     copy_and_name("f1", init_block),
    #     copy_and_name("f2", iter_block),
    #     copy_and_name("f3", iter_block),
    # ]
    pipeline_tasks = setup_basic_pipeline_tasks(
        init_block, iter_block, Nk-1,
    )
    main_pipeline = MMGTaskPipeline(
        "MMG Pipeline",
        pipeline_tasks,
        common_settings,
    )

    if verbosity >= 1:
        print(f"~~~ Pipeline Outline ~~~")
        print(str(main_pipeline))
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~")

    ### Set up context and generate scripts ###
    context = {
        "data-input-nus": data_input_nus,
        "freq-idx": 1,
        "nu-sf": str_nu_list[0],
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
        "run-mmg-solver-settings": run_mmg_solver_settings,
        "train-mmgublock-settings": train_mmgublock_settings,
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
