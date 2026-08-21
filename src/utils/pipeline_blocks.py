# pipeline_blocks
# Offer the sorts of blocks that I would probably use
# Since the setup is


import os, sys, copy, re
import yaml

from .pipeline_utils import (
    # Helper functions
    map_dict_vals_to_str,
    pretty_dict_to_str,
    apply_settings_yaml,
    copy_and_name,
    # Classes
    CodeObj,
    SoloTask,
    TaskGroup,
    FrequencyBlock,
    SequentialTasks,
    ParallelTasks,
    TaskPipeline
)
from .replace_fields_utils import (
    apply_replacements,
    apply_replacements_to_dict,
    propagate_replacements,
    parse_val
)

import badger

# Keys for re-use
FREQ_IDX = "freq-idx"
NU_SF = "nu-sf"
DSET = "dset"
TASK = "task"
LAST_TASK = "last-task"
WHOLE_RUN_NAME = "whole-run-name"
OUTPUT_PRED_FORMAT = "output-pred-format"

# MMG pipeline keys
OUTPUT_PRED_MMG_DIR   = "output-pred-mmg-dir"
OUTPUT_PRED_SCOBJ_DIR = "output-pred-scobj-dir"
LAST_OUTPUT_PRED_MMG_DIR = "last-output-pred-mmg-dir"
LAST_OUTPUT_PRED_SCOBJ_DIR = "last-output-pred-scobj-dir"

# other stuff
REF_DATASET_REL_DIR = "ref-dataset-rel-dir"

# Verbosity cutoffs
from .pipeline_utils import (
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

def prep_containing_dir(file_fp: str) -> None:
    """Little helper function to prepare directories for files"""
    dir_name, file_name = os.path.split(file_fp)
    os.makedirs(dir_name, exist_ok=True)

def common_gen_scripts(
    task: SoloTask,
    ctx: dict,
    fixed_settings: dict,
    in_badger_template_field: str,
    block_badger_field: str = None,
    setup_field_mapping: dict = dict(),
    save_field_mapping: dict = dict(),
    printable_io_fields: dict = dict(),
    verbosity: int=2,
) -> dict:
    """Helper function to consolidate the boilerplate code
    Also saves relevant variables to the task object

    For setup, initially copies eff_ctx[setup_field_mapping[key]]
    over to eff_ctx[setup_field_mapping[val]]

    Copies eff_ctx[save_field_mapping[key]] over to ctx[save_field_mapping[val]]
    and returns the updated copy of task
    Parameters:
        task (SoloTask): task object
        ctx (dict): dynamic context being passed in
        fixed_settings (dict): settings given to the task block in question
        in_badger_template_field (str): name of the field where to find the
            name of the badger template file
        block_badger_field (str): name of the field where to find the
            name of the output badger config for this block
        setup_field_mapping (dict): a mapping to help set up eff_ctx
            before calling badger, copies eff_ctx[src]
            to eff_ctx[dst] for each (dst, src) mapping
        save_field_mapping (dict): a mapping to update the context for the next block
            after calling badger, copies eff_ctx[src]
            to ctx[dst] for each (dst, src) mapping
            (basically, gives source->destination pairs)
            ctx is then returned
        printable_io_fields (dict): a collection of I/O fields to be printed if
            verbosity >= VLVL_IO_INFO. Each key is the eff_ctx field name
            and each value is the label for that field
        verbosity (int): level of outputs to print
    Returns:
        ctx (dict): updated context dictionary
        eff_ctx (dict): context dictionary used to generate the files, in case
            additional processing is required afterwards
    """
    # 0. Fetch the effective context
    eff_ctx = propagate_replacements(
        map_dict_vals_to_str(
            {**fixed_settings, **task.settings, **ctx}
        ),
        cleanup=False,
    )

    # 1. Locate the badger files
    block_badger_field = (
        block_badger_field
        if block_badger_field is not None
        else "block-badger-fp"
    )
    in_badger_template_fp = eff_ctx[in_badger_template_field]
    block_badger_fp       = eff_ctx[block_badger_field]

    # 2. Finish setup with eff_ctx_setup_mapping
    for (dst, src) in setup_field_mapping.items():
        eff_ctx[dst] = eff_ctx[src]

    # 3. Print out info
    freq_idx = eff_ctx.get(FREQ_IDX, 0) # just for printing purposes
    prefix = f"f{freq_idx} {task.name}"
    if verbosity >= VLVL_EFF_CTX:
        pretty_eff_ctx = pretty_dict_to_str(eff_ctx, f"{prefix} received eff_ctx...", indent_width=0)
        print(pretty_eff_ctx)
    if verbosity >= VLVL_IO_INFO:
        # Justify the labels for better readability
        justwidth = max(len(val) for val in printable_io_fields.values())
        for (field, label) in printable_io_fields.items():
            field_str = eff_ctx.get(field, "(missing from eff_ctx)")
            label_str = label.ljust(justwidth)
            print(f"{prefix} {label_str} {field_str}")
    if verbosity >= VLVL_SCRIPTS_INFO:
        # print(f"{prefix} in_base_template_fp   {in_base_template_fp}")
        print(f"{prefix} in_badger_template_fp {in_badger_template_fp}")
        print(f"{prefix} block_badger_fp       {block_badger_fp}")

    # 4. Create the badger file
    cleanup = parse_val(eff_ctx.get("cleanup", False), bool_as_str=False)
    prep_containing_dir(block_badger_fp)
    apply_settings_yaml(
        in_badger_template=in_badger_template_fp,
        out_badger_yaml=block_badger_fp,
        settings=eff_ctx,
        cleanup=True,
    )

    # 5. Call badger
    task_scripts = badger.run_from_python(
        block_badger_fp,
        parsimony=12,
        silence=True
    )[1]
    if verbosity >= VLVL_SCRIPTS_INFO:
        print(f"{prefix} block scripts {task_scripts}")

    # 6. Updates
    # 6a. Update block values
    task.block_badger_fp = block_badger_fp
    task.task_scripts    = task_scripts

    # 6b. Update the context
    ctx[LAST_TASK] = task.name
    for (dst, src) in save_field_mapping.items():
        ctx[dst] = eff_ctx[src]

    return ctx, eff_ctx

class TrainFYNet(SoloTask):
    code: str = "t"
    def __init__(self, name: str, settings: dict):
        super().__init__(name, settings)

    def gen_scripts(self, ctx: dict, fixed_settings: dict=dict(), verbosity: int=2):
        """This function should generate (and save) the badger yaml
        and possibly also the slurm scripts
        Can edit ctx; should not edit fixed_settings
        """
        ctx, eff_ctx = common_gen_scripts(
            self,
            ctx,
            fixed_settings,
            in_badger_template_field="badger-template-train-fynet",
            block_badger_field="block-badger-fp",
            setup_field_mapping = {},
            save_field_mapping = {},
            printable_io_fields= {
                "dataset-rel-dir": "dataset dir",
                "train-targets": "train targets",
                # "eval-targets": "eval targets",
            },
            verbosity=verbosity,
        )
        # save mapping e.g. {OUTPUT_PRED_SCOBJ_DIR, LAST_OUTPUT_PRED_SCOBJ_DIR}
        return ctx

class EvalFYNet(SoloTask):
    code: str = "e"
    def __init__(self, name: str, settings: dict):
        super().__init__(name, settings)

    def gen_scripts(self, ctx: dict, fixed_settings: dict=dict(), verbosity: int=2):
        """This function should generate (and save) the badger yaml
        and possibly also the slurm scripts
        Can edit ctx; should not edit fixed_settings
        """
        ctx, eff_ctx = common_gen_scripts(
            self,
            ctx,
            fixed_settings,
            in_badger_template_field="badger-template-eval-fynet",
            block_badger_field="block-badger-fp",
            setup_field_mapping={
                "output-pred-dir": OUTPUT_PRED_SCOBJ_DIR,
            },
            save_field_mapping={
                LAST_OUTPUT_PRED_SCOBJ_DIR: OUTPUT_PRED_SCOBJ_DIR,
            },
            printable_io_fields={
                OUTPUT_PRED_SCOBJ_DIR: "output pred dir",
                "dataset-rel-dir": "dataset dir",
                "eval-targets": "eval targets",
            },
            verbosity=verbosity,
        )
        return ctx

class TrainEvalMMGUBlock(SoloTask):
    code: str = "n" # use nn to differentiate from the split t/e blocks
    def __init__(self, name: str, settings: dict):
        super().__init__(name, settings)

    def gen_scripts(self, ctx: dict, fixed_settings: dict=dict(), verbosity: int=2):
        """This function should generate (and save) the badger yaml
        and possibly also the slurm scripts
        Can edit ctx; should not edit fixed_settings
        """
        ctx, eff_ctx = common_gen_scripts(
            self,
            ctx,
            fixed_settings,
            in_badger_template_field="badger-template-train-mmgu",
            block_badger_field="block-badger-fp",
            setup_field_mapping={
                "input-pred-scobj-dir": LAST_OUTPUT_PRED_SCOBJ_DIR,
                "input-pred-mmg-dir": LAST_OUTPUT_PRED_MMG_DIR,
                "output-pred-dir": "output-pred-format-internal",
            },
            save_field_mapping={
                LAST_OUTPUT_PRED_SCOBJ_DIR: "output-pred-dir",
            },
            printable_io_fields={
                REF_DATASET_REL_DIR:    "ref dataset dir",
                "input-pred-scobj-dir": "input pred scobj dir",
                "input-pred-mmg-dir":   "input pred mmg dir",
                "output-pred-dir":      "output pred scobj dir",
                "train-targets": "train targets",
                "eval-targets": "eval targets",
            },
            verbosity=verbosity,
        )
        return ctx

TrainMMGUBlock = TrainEvalMMGUBlock # to avoid breaking things...

class EvalMMGUBlock(SoloTask):
    code: str = "e" # just evaluation...
    def __init__(self, name: str, settings: dict):
        super().__init__(name, settings)

    def gen_scripts(self, ctx: dict, fixed_settings: dict=dict(), verbosity: int=2):
        """This function should generate (and save) the badger yaml
        and possibly also the slurm scripts
        Can edit ctx; should not edit fixed_settings
        """
        # 0. Fetch the effective context
        eff_ctx = map_dict_vals_to_str(
            {
                **fixed_settings,
                **self.settings,
                **ctx,
                # "train-targets": "smoothed(just testing!!)", # this works btw
            }
        )
        eff_ctx = propagate_replacements(eff_ctx, cleanup=False)
        freq_idx = eff_ctx.get(FREQ_IDX, 0)

        # 1. Decide where to place the badger script
        in_base_template_fp   = eff_ctx["base-template-eval-mmgu"]
        in_badger_template_fp = eff_ctx["badger-template-eval-mmgu"]
        block_badger_fp       = eff_ctx["block-badger-fp"]

        # 2. Set up inputs/outputs
        ref_dataset_dir      = eff_ctx[REF_DATASET_REL_DIR]
        input_pred_scobj_dir = eff_ctx[LAST_OUTPUT_PRED_SCOBJ_DIR]
        input_pred_mmg_dir   = eff_ctx[LAST_OUTPUT_PRED_MMG_DIR]
        output_pred_dir      = eff_ctx["output-pred-format-internal"]

        # 3. Print out info
        if verbosity >= VLVL_EFF_CTX:
            pretty_eff_ctx = pretty_dict_to_str(eff_ctx, f"f{freq_idx} {self.name} received eff_ctx...", indent_width=0)
            print(pretty_eff_ctx)
        if verbosity >= VLVL_IO_INFO:
            print(f"f{freq_idx} {self.name} dataset dir          {ref_dataset_dir}")
            print(f"f{freq_idx} {self.name} input pred scobj dir {input_pred_scobj_dir}")
            print(f"f{freq_idx} {self.name} input pred mmg dir   {input_pred_mmg_dir}")
            print(f"f{freq_idx} {self.name} out pred dir         {output_pred_dir}")
        if verbosity >= VLVL_SCRIPTS_INFO:
            print(f"f{freq_idx} {self.name} in_base_template_fp   {in_base_template_fp}")
            print(f"f{freq_idx} {self.name} in_badger_template_fp {in_badger_template_fp}")
            print(f"f{freq_idx} {self.name} block_badger_fp       {block_badger_fp}")
            # print(f"task {self.name} received effective settings\n{pretty_dict_to_str(eff_ctx,indent_width=2)}")

        # 4. Create the badger file
        # 4a. update eff_ctx as needed
        eff_ctx["input-pred-scobj-dir"] = input_pred_scobj_dir
        eff_ctx["input-pred-mmg-dir"] = input_pred_mmg_dir
        eff_ctx["output-pred-dir"] = output_pred_dir
        # 4b. call the replacement function
        cleanup = parse_val(eff_ctx.get("cleanup", False), bool_as_str=False)
        prep_containing_dir(block_badger_fp)
        apply_settings_yaml(
            in_badger_template=in_badger_template_fp,
            out_badger_yaml=block_badger_fp,
            settings=eff_ctx,
            cleanup=True,
        )

        # 5. Call badger
        task_scripts = badger.run_from_python(
            block_badger_fp,
            parsimony=12,
            silence=True
        )[1]
        if verbosity >= VLVL_SCRIPTS_INFO:
            print(f"f{freq_idx} {self.name} block scripts {task_scripts}")

        # 6. Updates
        # 6a. Update block values
        self.block_badger_fp = block_badger_fp
        self.task_scripts    = task_scripts

        # 6b. Update the context
        ctx[LAST_TASK] = self.name
        ctx[LAST_OUTPUT_PRED_SCOBJ_DIR] = output_pred_dir

        return ctx

# Settings I know I want to set...
# use-pred-d-mh: false
class TrainMRefBlock(SoloTask):
    code: str = "t" # just training here
    def __init__(self, name: str, settings: dict):
        super().__init__(name, settings)

    def gen_scripts(self, ctx: dict, fixed_settings: dict=dict(), verbosity: int=2):
        """This function should generate (and save) the badger yaml
        and possibly also the slurm scripts
        Can edit ctx; should not edit fixed_settings
        """
        ctx, eff_ctx = common_gen_scripts(
            self,
            ctx,
            fixed_settings,
            in_badger_template_field="badger-template-train-refinement-block",
            block_badger_field="block-badger-fp",
            setup_field_mapping={
                "input-pred-scobj-dir": LAST_OUTPUT_PRED_SCOBJ_DIR,
            },
            save_field_mapping={
                # LAST_OUTPUT_PRED_SCOBJ_DIR: OUTPUT_PRED_SCOBJ_DIR,
            },
            printable_io_fields={
                # OUTPUT_PRED_SCOBJ_DIR: "output pred dir",
                REF_DATASET_REL_DIR: "ref dataset dir",
                LAST_OUTPUT_PRED_SCOBJ_DIR: "input pred scobj dir",
                "train-targets": "train targets",
                # "eval-targets": "eval targets",
            },
            verbosity=verbosity,
        )
        return ctx

class EvalMRefBlock(SoloTask):
    code: str = "e" # just evaluation here
    def __init__(self, name: str, settings: dict):
        super().__init__(name, settings)

    def gen_scripts(self, ctx: dict, fixed_settings: dict=dict(), verbosity: int=2):
        """This function should generate (and save) the badger yaml
        and possibly also the slurm scripts
        Can edit ctx; should not edit fixed_settings
        """
        ctx, eff_ctx = common_gen_scripts(
            self,
            ctx,
            fixed_settings,
            in_badger_template_field="badger-template-eval-refinement-block",
            block_badger_field="block-badger-fp",
            setup_field_mapping={
                "input-pred-scobj-dir": LAST_OUTPUT_PRED_SCOBJ_DIR,
                "output-pred-dir": OUTPUT_PRED_SCOBJ_DIR,
            },
            save_field_mapping={
                LAST_OUTPUT_PRED_SCOBJ_DIR: OUTPUT_PRED_SCOBJ_DIR,
            },
            printable_io_fields={
                REF_DATASET_REL_DIR:    "ref dataset dir",
                "input-pred-scobj-dir": "input pred scobj dir",
                OUTPUT_PRED_SCOBJ_DIR:  "output pred dir",
                "eval-targets": "eval targets",
            },
            verbosity=verbosity,
        )
        return ctx

class TrainMRefPipeline(SoloTask):
    code: str = "t" # just training here
    def __init__(self, name: str, settings: dict):
        super().__init__(name, settings)
        self.freq_idx = "e"

    def gen_scripts(self, ctx: dict, fixed_settings: dict=dict(), verbosity: int=2):
        """This function should generate (and save) the badger yaml
        and possibly also the slurm scripts
        Can edit ctx; should not edit fixed_settings
        """
        ctx, eff_ctx = common_gen_scripts(
            self,
            ctx,
            fixed_settings,
            in_badger_template_field="badger-template-train-model-pipeline",
            block_badger_field="block-badger-fp",
            setup_field_mapping={},
            save_field_mapping={},
            printable_io_fields={
                REF_DATASET_REL_DIR: "ref dataset dir",
                LAST_OUTPUT_PRED_SCOBJ_DIR: "pred dataset dir",
                "train-targets": "train targets",
            },
            verbosity=verbosity,
        )
        return ctx


class EvalMRefPipeline(SoloTask):
    code: str = "e" # just evaluation here
    def __init__(self, name: str, settings: dict):
        super().__init__(name, settings)
        self.freq_idx = "e"

    def gen_scripts(self, ctx: dict, fixed_settings: dict=dict(), verbosity: int=2):
        """This function should generate (and save) the badger yaml
        and possibly also the slurm scripts
        Can edit ctx; should not edit fixed_settings
        """
        ctx, eff_ctx = common_gen_scripts(
            self,
            ctx,
            fixed_settings,
            in_badger_template_field="badger-template-eval-model-pipeline",
            block_badger_field="block-badger-fp",
            setup_field_mapping={
            },
            save_field_mapping={
            },
            printable_io_fields={
                REF_DATASET_REL_DIR: "ref dataset dir",
                OUTPUT_PRED_SCOBJ_DIR: "output pred dir",
                LAST_OUTPUT_PRED_SCOBJ_DIR: "pred dataset dir",
                "eval-targets": "eval targets",
            },
            verbosity=verbosity,
        )
        return ctx

class RunMMGSolverDataset(SoloTask):
    code: str = "d"
    def __init__(self, name: str, dset: str, settings: dict):
        super().__init__(name, settings)
        self.dset = dset

    def gen_scripts(self, ctx: dict, fixed_settings: dict=dict(), verbosity: int=2):
        """This function should generate (and save) the badger yaml
        and possibly also the slurm scripts
        Can edit ctx; should not edit fixed_settings
        """
        # 0. Fetch the effective context
        eff_ctx = map_dict_vals_to_str(
            {
                **fixed_settings,
                **self.settings,
                **ctx,
                DSET: self.dset,
                "num-samples": f"<<num-{self.dset}>>",
            }
        )
        eff_ctx  = propagate_replacements(eff_ctx, cleanup=False)
        freq_idx = eff_ctx.get(FREQ_IDX, 0)
        nu_sf    = eff_ctx[NU_SF]
        noise_seed_key = f"noise-seed-{self.dset}"
        if noise_seed_key in eff_ctx.keys():
            noise_seed = eff_ctx[noise_seed_key]
            eff_ctx["noise-seed-base"] = noise_seed

        # 1. Decide where to place the badger script
        in_base_template_fp   = eff_ctx["base-template-mmg-solver"]
        in_badger_template_fp = eff_ctx["badger-template-mmg-solver"]
        block_badger_fp       = eff_ctx["block-badger-fp"]

        # 2. Set up inputs/outputs
        ref_dataset_dir      = eff_ctx[REF_DATASET_REL_DIR]
        input_pred_scobj_dir = eff_ctx[LAST_OUTPUT_PRED_SCOBJ_DIR]
        output_mmg_rel_dir   = eff_ctx["output-mmg-rel-dir"]
        # output_name_format   = eff_ctx["output-name-format"]
        # f"{dset}_gammas_nu_{nu_sf}"

        # 3. Print out info
        if verbosity >= VLVL_IO_INFO:
            print(f"f{freq_idx} {self.name} num_samples     {eff_ctx['num-samples']}")
        if verbosity >= VLVL_SCRIPTS_INFO:
            print(f"f{freq_idx} {self.name} block_badger_fp {block_badger_fp}")

        if verbosity >= VLVL_EFF_CTX:
            pretty_eff_ctx = pretty_dict_to_str(eff_ctx, f"{self.name} received eff_ctx...", indent_width=0)
            print(pretty_eff_ctx)

        # 4. Apply replacement
        # 4a. Update context as needed
        eff_ctx["input-scobj-dir"] = input_pred_scobj_dir

        # 4b. call the replacement function
        cleanup = parse_val(eff_ctx.get("cleanup", False), bool_as_str=False)
        prep_containing_dir(block_badger_fp)
        apply_settings_yaml(
            in_badger_template=in_badger_template_fp,
            out_badger_yaml=block_badger_fp,
            settings=eff_ctx,
            cleanup=False,
        )

        # 5. Call badger
        task_scripts = badger.run_from_python(
            block_badger_fp,
            parsimony=12,
            silence=True
        )[1]
        if verbosity >= VLVL_SCRIPTS_INFO:
            print(f"f{freq_idx} {self.name} block scripts {task_scripts}")

        # 6. Updates
        # 6a. Update block
        self.block_badger_fp = block_badger_fp
        self.task_scripts    = task_scripts

        # 6b. Update context
        ctx[LAST_TASK] = self.name
        ctx[LAST_OUTPUT_PRED_MMG_DIR] = output_mmg_rel_dir

        return ctx

class RunMMGSolver(ParallelTasks):
    code: str = "s"
    def __init__(self, name: str, dset_list: list, settings: dict):
        task_list = [
            RunMMGSolverDataset(f"{name}-{dset}", dset, settings)
            for dset in dset_list
        ]
        super().__init__(name, task_list, settings)

    def gen_scripts(self, ctx: dict, fixed_settings: dict, verbosity: int=2):
        """Generate scripts; call RunMMGSolverDataset for each relevant dataset"""
        # 0. Fetch the effective context
        eff_ctx = map_dict_vals_to_str(
            {
                **fixed_settings,
                **self.settings,
                **ctx,
            }
        )
        eff_ctx = propagate_replacements(eff_ctx, cleanup=False)

        # 1. Fetch badger info for printing purposes
        in_base_template_fp   = eff_ctx["base-template-mmg-solver"]
        in_badger_template_fp = eff_ctx["badger-template-mmg-solver"]

        freq_idx = eff_ctx.get(FREQ_IDX, 0)
        output_mmg_rel_dir   = eff_ctx["output-mmg-rel-dir"]
        ref_dataset_dir      = eff_ctx[REF_DATASET_REL_DIR]
        input_pred_scobj_dir = eff_ctx[LAST_OUTPUT_PRED_SCOBJ_DIR]
        output_mmg_rel_dir   = eff_ctx["output-mmg-rel-dir"]
        output_name_format   = eff_ctx["output-name-format"]

        # 2. Print outputs
        if verbosity >= VLVL_IO_INFO:
            print(f"f{freq_idx} {self.name} input_pred_scobj_dir  {input_pred_scobj_dir}")
            print(f"f{freq_idx} {self.name} output_mmg_rel_dir    {output_mmg_rel_dir}")
            print(f"f{freq_idx} {self.name} output_name_format    {output_name_format}")
        if verbosity >= VLVL_SCRIPTS_INFO:
            print(f"f{freq_idx} {self.name} in_base_template_fp   {in_base_template_fp}")
            print(f"f{freq_idx} {self.name} in_badger_template_fp {in_badger_template_fp}")

        # 3. Do the actual run here
        ctx = super().gen_scripts(ctx, fixed_settings, verbosity=verbosity)

        # 4. Update the block and context as necessary
        ctx[LAST_OUTPUT_PRED_MMG_DIR] = output_mmg_rel_dir
        ctx[LAST_TASK] = self.name

        return ctx

    def get_codes(self):
        """Here, I prefer not to expose the train/val/test distinction
        """
        return (self.code, CodeObj(self.code, self.task_list, self))

    def get_tasks_from_command(self, command_str: str):
        """Get the relevant commands from this string"""
        if self.code in command_str:
            return self
        else:
            return None

### Treat TaskPipeline a bit differently since it's the wrapper ###
# Under the hood it uses a SequentialTasks object
class MMGTaskPipeline(TaskPipeline):
    """Special case of the TaskPipeline for the MMG-style pipeline
    Expects a sequence of FrequencyBlock objects
    # The only thing to override is the script submission behavior
    """
    def submit_scripts(
        self,
        command_str: str,
        sleep_time: float=0.5,
        verbosity: int=2,
        dry_run: bool=False,
    ):
        """Submits the internally-stored list of slurm jobs
        Submits the job dependencies in such a way that each task in the task list
        is dependent on the previous one
        Differs from SequentialTasks' submission for printing purposes
        """
        # 1. Parse the command str to get the relevant subset of tasks to run
        if verbosity >= 1:
            print(f"Received command_str={command_str}")

        # The code corresponding to running the entire pipeline
        # (computed unconditionally so "all"/"none" can also display it)
        full_code_str, full_code_obj = self.sequential_tasks.get_codes(delim=" ")
        if verbosity >= 1+VLVL_RUN_PLAN:
            print(f"Full pipeline code str: {full_code_str}")
            print(f"Full pipeline code obj:\n{str(full_code_obj)}")

        if command_str == "all":
            selected_task_list = self.task_list
            selected_tasks = self.sequential_tasks
            if verbosity >= VLVL_RUN_PLAN:
                print(f"Running everything")
        elif command_str == "none":
            selected_task_list = []
            selected_tasks = "(none)"
            if verbosity >= VLVL_RUN_PLAN:
                print(f"Running nothing")
        else:
            # Process the commands by going through the full pipeline and executing
            # the relevant tasks/subtasks when relevant
            command_str   = command_str.strip(" ")
            command_queue = command_str.split(" ")
            next_command  = command_queue.pop(0)
            selected_task_list = []
            for code_obj in full_code_obj.children:
                # Get the frequency block under consideration
                freq_block_task = code_obj.curr_task
                freq_idx = freq_block_task.freq_idx

                # Pattern: optionally start each block with f
                # Then, the \b and [^0-9]+ are to ensure we don't spuriously match
                # in case the freq_idx is contained within next_command but only in
                # the sense of a string; e.g., freq_idx=1, next_command="f10sn" or "f10" or "10"
                # would be an example of a pattern that should not be matched
                # task_matches  = re.findall(f"\\b[^0-9]+{freq_idx}\\b", next_command) # doesn't capture trailing subtasks
                task_matches  = re.findall(f"\\b[^0-9]*{freq_idx}[^0-9]*\\b", next_command)
                is_task_match = len(task_matches) > 0
                if is_task_match:
                    # Extract the code describing which sub-tasks of the frequency block to run
                    # e.g., just the PDE solver or just the neural network or both
                    # note: will run all available sub-tasks if none are specified explicitly
                    # Example pattern behavior: "f1et" -> "et" or "10" -> ""
                    # Include freq_idx for the e2e case where I set it to e
                    # subtask_code = re.sub(f"[f]?[0-9{freq_idx}]+", "", next_command)
                    subtask_code = re.sub(f"[f]?{freq_idx}", "", next_command, count=1)

                    selected_block = None
                    if len(subtask_code) == 0:
                        # Use the original block with no adjustment
                        selected_block = freq_block_task
                    else:
                        # Collect the relevant subtasks...
                        subtask_queue = list(subtask_code)
                        next_subtask_code = subtask_queue.pop(0)
                        fb_subtask_list  = freq_block_task.task_list
                        selected_subtask_list = []
                        # import pdb; pdb.set_trace()
                        for fb_subtask in fb_subtask_list:
                            # for each subtask from the full selection of subtasks
                            # attempt to match against the requested code
                            fb_subtask_code, fb_subtask_obj = fb_subtask.get_codes()

                            # In case of a match...
                            if fb_subtask_code == next_subtask_code:
                                # Add to selected_subtask_list
                                selected_subtask_list.append(fb_subtask_obj.curr_task)

                                # Pop from the queue and keep going
                                if len(subtask_queue) == 0:
                                    break
                                next_subtask_code = subtask_queue.pop(0)

                        selected_block = FrequencyBlock(
                            name=f"{freq_block_task.name} (post-selection)",
                            task_list=selected_subtask_list,
                            settings=freq_block_task.settings,
                        )

                    # Add the selected block/task
                    if verbosity >= VLVL_BLOCK_JOBS:
                        print(f"Selected: {selected_block}")
                    selected_task_list.append(selected_block)

                    if len(command_queue) == 0:
                        break
                    next_command = command_queue.pop(0)
            if verbosity >= VLVL_BLOCK_JOBS:
                print(f"Selected task list: {selected_task_list}")

        # Preview the run plan
        if verbosity >= VLVL_RUN_PLAN:
            selected_tasks = SequentialTasks(
                f"Selected Tasks Pipeline ({command_str})",
                selected_task_list,
                self.common_settings
            )
            print("~~~ Run plan ~~~")
            print(selected_tasks)
            print("~~~~~~~~~~~~~~~~")

        if dry_run:
            print(f"Exiting early since this is a dry run")
            sys.exit(0)

        # 2. Submit the relevant jobs
        job_dependency_list = []

        all_job_id_list = []
        curr_dep_list = job_dependency_list
        for i, task in enumerate(selected_task_list, start=1):
            task_job_id_list, new_job_id_list = task.submit_scripts(
                curr_dep_list,
                sleep_time=sleep_time,
                verbosity=verbosity,
            )
            curr_dep_list = new_job_id_list
            all_job_id_list += task_job_id_list
            if verbosity >= VLVL_ALL_JOBS:
                print(f"{task.name} jobs: {' '.join(map(str, task_job_id_list))}")

        if verbosity >= VLVL_ALL_JOBS:
            print(f"All pipeline jobs encountered: {' '.join(map(str, all_job_id_list))}")
        return all_job_id_list, all_job_id_list

SYSTEM_SETUP_FP = os.path.join(os.path.dirname(__file__), "..", "..", "system_setup.yaml")

def load_system_setup(config_fp: str = SYSTEM_SETUP_FP) -> dict:
    """Load repo-wide defaults (rlc-repo location, slurm/wandb defaults, ...)
    from system_setup.yaml so individual pipeline configs don't need to
    hardcode them. Returns {} if the file isn't present, so callers can fall
    back to their own defaults.
    """
    config_fp = os.path.abspath(config_fp)
    if not os.path.isfile(config_fp):
        return {}
    with open(config_fp, "r") as f:
        return yaml.safe_load(f) or {}

### Can write an alternate version for refinement-block settings or other stuff that come up later ###
def get_standard_mmg_settings(repo_dir: str = None, rlc_data_dir: str = None):
    """Get the basic/standard settings for the mmg pipeline
    """
    system_setup = load_system_setup()
    file_paths = system_setup.get("file-paths", {})

    if repo_dir is None:
        if "repo-dir" in file_paths:
            repo_dir = file_paths["repo-dir"].rstrip("/")
        else:
            raise ValueError(
                "No valid repo directory passed to get_standard_mmg_settings! "
                "Please either pass by argument or place under system_setup.yaml "
                "under a file-paths section."
            )

    if rlc_data_dir is None:
        rlc_data_dir = file_paths.get("data-rel-dir", "rlc_data")
    rlc_data_dir = rlc_data_dir.rstrip("/") if rlc_data_dir is not None else "rlc_data"

    # Overrideable settings fetched from system_setup.yaml, if present
    # (mail-user/mail-type and badger-* settings are intentionally left out;
    # they're rarely changed and can be added back in if someone wants them)
    system_settings = {
        **system_setup.get("default-slurm-settings", {}),
        **system_setup.get("wandb-settings", {}),
    }

    dataset_dir = file_paths.get("main-dataset", "dataset")

    standard_settings = {
        # General stuff
        "rlc-repo":                 repo_dir,
        "rlc-data":                 rlc_data_dir,
        "dataset-rel-dir":          dataset_dir, # for FYNet/base
        "ref-dataset-rel-dir":      dataset_dir, # for MMG/refinement-based stuff
        "ref-dataset-dir":          dataset_dir, # for MMG/refinement-based stuff
        "dataset-dir":              dataset_dir, # just in case
        "templates-rel-dir":        "pipeline_templates",
        "predictions-rel-dir":      f"{rlc_data_dir}/mmg_pipeline/predictions",
        "results-rel-dir":          f"{rlc_data_dir}/mmg_pipeline/results",
        "models-rel-dir":           f"{rlc_data_dir}/mmg_pipeline/models",
        "central-rel-dir":          f"{rlc_data_dir}/mmg_pipeline/central_run_info",
        "hps-sd-mat-dir":           file_paths.get("hps-sd-mat-dir", f"{rlc_data_dir}/HPS_SD_matrices"),
        # Slurm stuff
        "logs-rel-dir-base":        "logs/mmg_pipeline",
        "jobs-rel-dir-base":        "jobs/mmg_pipeline",
        "partition":                system_settings.get("partition", "gpu"),
        "badger-fake-submission":   "true",
        "badger-overwrite-scripts": "true",
        "mail-type":                "NONE",
        "mail-user":                "NONE",
        "num-gpu":                  system_settings.get("num-gpu", "1"),
        "num-cpu":                  system_settings.get("num-cpu", "2"),
        "mem":                      system_settings.get("mem", "50G"),
        "time-limit":               system_settings.get("time-limit", "4:00:00"),
        "wandb-entity":             system_settings.get("wandb-entity", "none"),
        "wandb-mode":               system_settings.get("wandb-mode", "offline"),

        # Template locations
        "base-template-train-fynet":   "<<templates-rel-dir>>/base_template_train_fynet.jinja",
        "badger-template-train-fynet": "<<templates-rel-dir>>/badger_template_train_fynet.yaml",
        "base-template-eval-fynet":    "<<templates-rel-dir>>/base_template_eval_fynet.jinja",
        "badger-template-eval-fynet":  "<<templates-rel-dir>>/badger_template_eval_fynet.yaml",

        "base-template-mmg-solver":    "<<templates-rel-dir>>/base_template_mmg_solver.jinja",
        "badger-template-mmg-solver":  "<<templates-rel-dir>>/badger_template_mmg_solver.yaml",

        "base-template-train-mmgu":    "<<templates-rel-dir>>/base_template_train_mmgublock.jinja",
        "badger-template-train-mmgu":  "<<templates-rel-dir>>/badger_template_train_mmgublock.yaml",
        "base-template-eval-mmgu":     "<<templates-rel-dir>>/base_template_eval_mmgublock.jinja",
        "badger-template-eval-mmgu":   "<<templates-rel-dir>>/badger_template_eval_mmgublock.yaml",

        # MFISNet-Refinement per-frequency block stuff
        "base-template-train-refinement-block":    "<<templates-rel-dir>>/base_template_train_refinement_block.jinja",
        "badger-template-train-refinement-block":  "<<templates-rel-dir>>/badger_template_train_refinement_block.yaml",
        "base-template-eval-refinement-block":     "<<templates-rel-dir>>/base_template_eval_refinement_block.jinja",
        "badger-template-eval-refinement-block":   "<<templates-rel-dir>>/badger_template_eval_refinement_block.yaml",

        # Model Pipeline e2e
        "base-template-train-model-pipeline":    "<<templates-rel-dir>>/base_template_train_model_pipeline.jinja",
        "badger-template-train-model-pipeline":  "<<templates-rel-dir>>/badger_template_train_model_pipeline.yaml",
        "base-template-eval-model-pipeline":     "<<templates-rel-dir>>/base_template_eval_model_pipeline.jinja",
        "badger-template-eval-model-pipeline":   "<<templates-rel-dir>>/badger_template_eval_model_pipeline.yaml",
    }
    standard_settings = map_dict_vals_to_str(standard_settings)
    return standard_settings

def setup_basic_pipeline_tasks(
    init_block: FrequencyBlock,
    iter_block: FrequencyBlock,
    iter_count: int,
) -> list:
    """Sets up a basic pipeline but only does the tasks
    """
    init_block_copy = copy_and_name("f1", init_block)
    iter_block_list = [
        copy_and_name(f"f{i}", iter_block)
        for i in range(2, 2+iter_count)
    ]
    return [init_block_copy, *iter_block_list]

def setup_basic_mmg_pipeline(
    init_block: FrequencyBlock,
    iter_block: FrequencyBlock,
    iter_count: int,
    pipeline_name: str = None,
    common_settings: dict = dict(),
) -> MMGTaskPipeline:
    """Sets up a basic looped pipeline that has a different initial block
    However, the iter_block gets deepcopied so each occurrence is different
    iter_count is the number of times to repeat the iter_block, so it should be
    one less than the total number of blocks
    """
    pipeline_name   = pipeline_name if pipeline_name is not None else "Basic Pipeline"
    pipeline_tasks = setup_basic_pipeline_tasks(init_block, iter_block, iter_count)

    basic_pipeline = MMGTaskPipeline(
        pipeline_name,
        pipeline_tasks,
        common_settings,
    )
    return basic_pipeline
