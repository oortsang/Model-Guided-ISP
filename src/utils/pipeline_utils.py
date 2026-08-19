# pipeline_utils.py
# more general utilities for the pipeline
# Contains helper functions and basic Task/TaskGroup classes
# along with the TaskPipeline class

import re, os, sys, time
import copy
import subprocess

from typing import Iterable

from src.utils.replace_fields_utils import apply_replacements

import numpy as np
rng = np.random.default_rng(234)

import badger

BADGER_YAMLS_SUBDIR = "badger_yamls"
PIPELINE_PLAN_FILE  = "pipeline_plan.yaml"
PIPELINE_PICKLE_FILE = "task_pipeline.pickle"

INDENT_STEP = 2

COMMAND_STR_ALL = "all"

# Verbosity cutoffs
# Generation
VLVL_IO_INFO      = 2 # Print the I/O information such as script and scattering object locations
VLVL_INIT_CONFIG  = 3 # Print the initial configuration received from the pipeline script
VLVL_SCRIPTS_INFO = 4 # Print the paths of the slurm shell scripts generated
VLVL_EFF_CTX      = 5 # Print the effective context at each block (quite verbose)
# Running
VLVL_RUN_PLAN     = 1 # Pretty output indicating which jobs are to be run
VLVL_ALL_JOBS     = 2 # Display all slurm job ids submitted
VLVL_BLOCK_JOBS   = 3 # Display the slurm job ids for each block
# VLVL_SBATCH_CMD   = 4 # show the sbatch command used for each submission

##### Helper functions #####
# File organization
def setup_pipeline_dir(pipeline_dir: str, exist_ok: bool) -> dict:
    """Sets up a directory for use storing pipeline stuff
    """

    # Create a new directory; can throw errors depending on exist_ok
    # if the directory already exists (e.g. if you want to avoid
    # overwriting stuff)
    os.makedirs(pipeline_dir, exist_ok=exist_ok)

    # Set up the pipeline_plan file
    badger_dir = os.path.join(pipeline_dir, BADGER_YAMLS_SUBDIR)
    os.makedirs(badger_dir, exist_ok=exist_ok)

    # Prepare the other file names without creating them yet
    pipeline_plan_fp = os.path.join(pipeline_dir, PIPELINE_PLAN_FILE)
    pipeline_pickle_fp = os.path.join(pipeline_dir, PIPELINE_PICKLE_file)

    paths_dict = {
        "pipeline_dir": pipeline_dir,
        "badger_yamls_dir": badger_dir,
        "pipeline_plan_fp": pipeline_plan_fp,
        "pipeline_pickle_fp": pipeline_pickle_fp,
    }
    return paths_dict

# Formatting helpers
def any_instance(x, type_list: Iterable) -> bool:
    return any(isinstance(x, type_i) for type_i in type_list)

def map_dict_vals_to_str(dd: dict) -> dict:
    """Converts all the values of a dictionary to string type
    """

    out_dd = {
        k: str(v) if not any_instance(v, [bool])
        else ("true" if v else "false") if isinstance(v, bool)
        else v
        for (k, v) in dd.items()
    }
    return out_dd

def indenter(msg_lines: list|str, indent_width: int) -> list:
    """Handle indentation"""
    indent_str = indent_width*" "
    if isinstance(msg_lines, list):
        out_lines = [indent_str+line for line in msg_lines]
    else:
        out_lines = [indent_str + msg_lines]
    return out_lines

def pretty_dict_to_str(
    dd: dict,
    dict_label: str=None,
    indent_width: int=0,
    include_header: bool=False,
    justify_right: bool=False,
) -> str:
    justify_len = max(len(k) for k in dd.keys())+2
    out_lines = []

    if dict_label is not None:
        out_lines    += indenter(dict_label, indent_width)
        indent_width  = indent_width + INDENT_STEP

    if include_header:
        header     = "Keys" .ljust(justify_len) + "Values"
        out_lines += indenter(header, indent_width)

    justify_fn = "rjust" if justify_right else "ljust"
    out_lines += indenter(
        [
            getattr(f"{k}: ", justify_fn)(justify_len) + str(v)
            for (k,v) in dd.items()
        ],
        indent_width,
    )
    out_str = "\n".join(out_lines)
    return out_str

def apply_settings_yaml(
    in_badger_template: str,
    out_badger_yaml: str,
    settings: dict,
    cleanup: bool=False,
    verbosity: int=2,
) -> str:
    """This should be roughly equivalent to the task in replace_fields_in_chevrons.py
    Returns a copy of the output as a string
    """
    # if verbosity >= VLVL_IO_INFO:
    #     print(
    #         f"(apply_settings_yaml) would apply settings from "
    #         f"{in_badger_template} to {out_badger_yaml}"
    #     )
    out_str = ""
    with open(in_badger_template, "r") as f:
        for li, line in enumerate(f):
            try:
                out_str += apply_replacements(
                    line, settings, cleanup=cleanup,
                )
            except:
                print(f"Note: apply_settings_yaml encountered an error at line {li}")
                raise
        with open(out_badger_yaml, "w") as f:
            f.write(out_str)
    return out_str

def str_tuple_to_list(str_tuple: str) -> list:
    return str_tuple.strip("()").split(",")

def submit_slurm_job(
    script: str,
    job_dependency_str: None,
    sleep_time: float=0.5,
    fake_submit: bool=False,
) -> int:
    """Submit submission_script to slurm, with the dependencies on job_dependency_str
    Also sleep afterwards to avoid problems with slurm submissions

    Returns the job id of the slurm job
    """
    global rng
    
    # print(f"Received job_dependency_str={job_dependency_str}")
    dep_str = f"--dependency={job_dependency_str}" if job_dependency_str is not None else None
    arg_list = [
        "sbatch",
        f"--parsable",
        f"{script}",
    ]
    if dep_str is not None:
        arg_list.insert(2, dep_str)

    if fake_submit:
        sim_job_id = str(rng.integers(0, 1000))
        print(f"Simulating a submission of {' '.join(arg_list)}; simulated output {sim_job_id}")
        return sim_job_id

    # for real submissions
    output = subprocess.run(arg_list, shell=False, capture_output=True)
    output_utf8 = output.stdout.decode("UTF-8")
    output_job_id = re.findall("[0-9]+", output_utf8)[-1]

    # print(f"Received output: {output_job_id}")
    time.sleep(sleep_time)
    return output_job_id


def slurm_job_dependency_list_to_str(job_id_list: list) -> str:
    """Helper function to convert a list of slurm job ids into the string for use with sbatch
    Use afterok to wait until ~after~ the jobs finish ~okay~ (i.e., wait for all the
    jobs in job_id_list)
    """
    if job_id_list is None or len(job_id_list) == 0:
        return None
    job_dependency_str = "afterok:" + ":".join([str(job_id) for job_id in job_id_list])
    return job_dependency_str

def submit_slurm_job_list(script_list, job_dependency_list: list=None, sleep_time: float=0.5) -> list:
    """Batched version of submit_slurm_job"""
    job_dependency_str = slurm_job_dependency_list_to_str(job_dependency_list)
    output_job_id_list = [
        submit_slurm_job(script, job_dependency_str, sleep_time=sleep_time)
        for script in script_list
    ]
    return output_job_id_list

### Helper object for tracking the codes
class CodeObj:
    code: str
    children: list = []
    # curr_task = None # will be type Task, but that is not defined yet
    def __init__(self, code, children, curr_task=None):
        self.code      = code
        self.children  = children
        self.curr_task = curr_task

    def get_str_lines(self, indent: int = 0) -> str:
        """Gets the string representation but separated by line to simplify indentation
        """
        # obj_type = type(self).__name__
        # msg = f"{code} {self.name}"
        # out_lines = indenter(msg.split("\n"), indent)
        curr_task_name = f": {self.curr_task.name}" if self.curr_task is not None else ""
        msg_lines = indenter([self.code+curr_task_name], indent)
        if len(self.children) > 0:
            children_lines = sum([c.get_str_lines(indent+INDENT_STEP) for c in self.children], start=[])
        else:
            children_lines = []
        out_lines = msg_lines + children_lines
        return out_lines

    def __str__(self):
        return "\n".join(self.get_str_lines())

##### Task-related classes #####
class Task:
    name: str = "Generic Task"
    code: str = "g"
    settings: dict = {}
    task_scripts: list
    def __init__(self, name: str, settings: dict):
        self.name = name
        self.settings = {**settings}
        self.task_scripts = []

    def apply(self, *args, **kwargs):
        print(f"task {self.name} received arguments as {args} and {kwargs}")

    def gen_scripts(self, ctx: dict, base_settings: dict = None, verbosity: int = 2) -> dict:
        eff_ctx = {
            **base_settings,
            **self.settings,
            **ctx,
        }
        if verbosity >= VLVL_EFF_CTX:
            print(f"task {self.name} received effective context {eff_ctx}")
        ctx[f"{self.name.replace(' ', '-')}-was-here"] = "true"
        return ctx

    def get_str_lines(self, indent: int = 0) -> str:
        """Gets the string representation but separated by line to simplify indentation
        """
        obj_type = type(self).__name__
        msg = f"{obj_type} {self.name}"
        out_lines = indenter(msg.split("\n"), indent)

        return out_lines

    def __str__(self):
        return "\n".join(self.get_str_lines())

    def submit_scripts(
        self,
        job_dependency_list: list,
        sleep_time: float=0.5,
        verbosity: int=2,
    ) -> list:
        """Submits the internally-stored list of slurm jobs
        Returns a list of all the job ids encountered as well as the new ones
        """
        if len(self.task_scripts) == 0:
            return job_dependency_list

        # job_dependency_str = slurm_job_dependency_list_to_str(job_dependency_list)
        # import pdb; pdb.set_trace()
        new_job_id_list = submit_slurm_job_list(
            self.task_scripts,
            job_dependency_list,
            sleep_time=sleep_time,
        )
        all_job_id_list = job_dependency_list + new_job_id_list
        return all_job_id_list, new_job_id_list

    def get_codes(self, delim: str = ""):
        """Get the codes corresponding to each of the tasks in task_list
        """
        # code_obj = {"code": self.code, "children": []}
        code_obj = CodeObj(self.code, [self], self)
        # return ([self.code], self.code)
        return (self.code, code_obj)

    def get_tasks_from_command(self, command_str):
        """Get the relevant commands from this string"""
        print(f"{self.name} (code {self.code}) received command_str={command_str}")
        if self.code in command_str:
            return self
        else:
            return None


class SoloTask(Task):
    pass # just let this be a wrapper for the sake of type clarity

def copy_and_name(name: str, block: Task) -> Task:
    """deepcopy the block in question, then update its name
    Code should be valid for any Task or TaskGroup object
    """
    block_copy = copy.deepcopy(block)
    block_copy.name = name
    return block_copy


### Grouping tasks together ###
class TaskGroup(Task):
    """Intended as a generic type where you are grouping together tasks
    """
    code: str = None # try to avoid using this generic version directly from TaskGroup
    def __init__(self, name: str, task_list: list, settings: dict = dict()):
        self.name      = name
        self.task_list = task_list
        self.settings  = {**settings}
        self.task_scripts = []

    def gen_scripts(self, ctx: dict, base_settings: dict, verbosity: int = 2) -> dict:
        for i, task in enumerate(self.task_list, start=1):
            ctx = task.gen_scripts(ctx, {**self.settings, **base_settings}, verbosity=verbosity)
        return ctx

    def get_str_lines(self, indent: int = 0):
        obj_type = type(self).__name__
        msg = indenter(f"{obj_type} {self.name}", indent)
        # Concatenate the list of lists with sum
        task_list_lines = msg + sum(
            [
                task_i.get_str_lines(indent+INDENT_STEP)
                for task_i in self.task_list
            ],
            start=[],
        )
        return task_list_lines

    def submit_scripts(
        self,
        job_dependency_list: list,
        sleep_time: float=0.5, 
        verbosity: int=2,
    ) -> list:
        """Submits the internally-stored list of slurm jobs
        This version should not be used, since submission procedure varies between task group styles
        """
        raise NotImplementedError(
            f"{self.name} ({type(self)}) has not implemented a submit_scripts method"
        )

    def get_codes(self, delim: str = ""):
        """Get the codes corresponding to each of the tasks in task_list
        """
        code_obj_list = []
        code_str_list = []
        for task in self.task_list:
            task_code_str, task_code_obj = task.get_codes()
            # code_list.append(task_code_str)
            code_obj_list.append(task_code_obj)
            code_str_list.append(task_code_str)
        code_str = delim.join(code_str_list)
        code_obj = CodeObj(code_str, code_obj_list, self)
        return code_str, code_obj

    def get_tasks_from_command(self, command_str):
        """I don't think there's a good generic way to handle this"""
        raise NotImplementedError


class SequentialTasks(TaskGroup):
    """Package a group of tasks together that should be run in sequence
    In the future, this will affect slurm behavior
    """
    def __init__(self, name: str, task_list: list, *args, **kwargs):
        super().__init__(name, task_list, *args, **kwargs)

    def gen_scripts(self, ctx: dict, base_settings: dict, verbosity: int = 2) -> dict:
        """Run tasks and augment freq-idx"""
        ctx = super().gen_scripts(ctx, base_settings, verbosity=verbosity)
        return ctx

    def submit_scripts(
        self,
        job_dependency_list: list,
        sleep_time: float=0.5,
        verbosity: int=2,
    ) -> list:
        """Submits the internally-stored list of slurm jobs
        Submits the job dependencies in such a way that each task in the task list
        is dependent on the previous one
        """
        curr_dep_list = job_dependency_list
        # all_job_id_list = [*job_dependency_list]
        all_job_id_list = []
        for task in self.task_list:
            task_job_id_list, new_dep_list = task.submit_scripts(
                curr_dep_list,
                sleep_time=sleep_time,
            )
            curr_dep_list    = new_dep_list
            all_job_id_list += new_dep_list
        new_job_id_list = curr_dep_list
        if verbosity >= VLVL_BLOCK_JOBS:
            print(f"SequentialTasks {self.name} all  jobs: {' '.join(all_job_id_list)}")
            print(f"SequentialTasks {self.name} tail jobs: {' '.join(new_job_id_list)}")
        return all_job_id_list, new_job_id_list

    # def get_tasks_from_command(self, command_str):
    #     code_list, code_str = self.get_codes()

    #     if code_str == COMMAND_STR_ALL:
    #         return self # just use everything

    #     selected_task_list = []
    #     command_list = command_str.split(" ")
    #     print(f"{self.name} command list {command_list}")


class ParallelTasks(TaskGroup):
    """Package a group of tasks together that should be allowed to run in parallel
    In the future, this will affect slurm behavior
    """
    def __init__(self, name: str, task_list: list, *args, **kwargs):
        super().__init__(name, task_list, *args, **kwargs)

    def gen_scripts(self, ctx: dict, base_settings: dict, verbosity: int = 2) -> dict:
        """Run tasks and augment freq-idx"""
        ctx = super().gen_scripts(ctx, base_settings, verbosity=verbosity)
        return ctx

    def submit_scripts(
        self,
        job_dependency_list: list,
        sleep_time: float=0.5,
        verbosity: int=2,
    ) -> list:
        """Submits the internally-stored list of slurm jobs
        Submits the job dependencies in such a way that each task in the task list
        only depends on the incoming job dependencies; so each of these can run in parallel.
        """
        curr_dep_list = []
        for task in self.task_list:
            _, new_dep_list = task.submit_scripts(
                job_dependency_list,
                sleep_time=sleep_time,
            )
            curr_dep_list += new_dep_list
        new_job_id_list = curr_dep_list
        all_job_id_list = job_dependency_list + new_job_id_list
        all_job_id_list = new_job_id_list
        if verbosity >= VLVL_BLOCK_JOBS:
            print(f"ParallelTasks {self.name} all  jobs: {' '.join(all_job_id_list)}")
            print(f"ParallelTasks {self.name} tail jobs: {' '.join(new_job_id_list)}")
        return all_job_id_list, new_job_id_list

# This has more special behavior than anything else in this file...
class FrequencyBlock(SequentialTasks):
    """Collect tasks corresponding to a single frequency block
    After these have finished running, augment the freq-idx
    and also save a last-freq-idx value
    """
    def __init__(self, name: str, task_list: list, *args, **kwargs):
        super().__init__(name, task_list, *args, **kwargs)
        # self.freq_idx = self.kwargs[freq_idx] if "freq_idx" in kwargs.keys() else None
        self.freq_idx = None

    def gen_scripts(self, ctx: dict, base_settings: dict, verbosity: int=2) -> dict:
        """Run tasks and augment freq-idx"""
        str_nu_list = str_tuple_to_list(ctx.get("data-input-nus"))
        freq_idx = ctx.get("freq-idx", 1)
        if verbosity >= VLVL_IO_INFO:
            print(f"~~~Frequency Block f{freq_idx}~~~")
        ctx["nu-sf"] = str_nu_list[freq_idx-1] # update the frequency
        ctx = super().gen_scripts(ctx, base_settings, verbosity)

        ctx["freq-idx"] = freq_idx + 1 # augment the frequency index after this
        ctx["last-freq-idx"] = freq_idx
        if freq_idx < len(str_nu_list):
            ctx["nu-sf"]    = str_nu_list[freq_idx-0] # update the frequency

        self.freq_idx = freq_idx
        return ctx

    def get_codes(self, delim: str = "") -> tuple:
        """Get the code corresponding to this block's contents
        """
        delim =  ""
        freq_idx = self.freq_idx
        if freq_idx is None:
            raise ValueError(f"FrequencyBlock {self.name}'s method get_codes called, but freq_idx has not been set!")

        code_str, code_obj = super().get_codes(delim)
        fic = str(freq_idx) # frequency index char/code

        # new_code_list = [fic, *code_list]
        new_code_str  = fic + code_str
        new_code_obj = CodeObj(new_code_str, code_obj.children, self)
        return (new_code_str, new_code_obj)


class TaskPipeline():
    def __init__(self, name: str, task_list: list, common_settings: dict = None):
        self.name = name if name is not None else "TaskPipeline"
        self.task_list = task_list
        self.common_settings = {**common_settings} if common_settings is not None else dict()
        self.sequential_tasks = SequentialTasks(
            self.name,
            self.task_list,
            self.common_settings,
        )

    def gen_scripts(self, ctx: dict, settings: dict = None, verbosity: int = 2):
        """Can pass settings to override the common settings previously set
        """
        ctx = self.sequential_tasks.gen_scripts(
            ctx,
            base_settings=settings,
            verbosity=verbosity,
        )
        return ctx

    def get_str_lines(self, indent: int = 0):
        task_list_lines = self.sequential_tasks.get_str_lines(indent)
        return task_list_lines

    def __str__(self):
        return "\n".join(self.get_str_lines())

    def submit_scripts(self, command_str: str, sleep_time: float=0.5, verbosity: int=2):
        """Submits the internally-stored list of slurm jobs
        Submits the job dependencies in such a way that each task in the task list
        is dependent on the previous one
        Differs from SequentialTasks' submission for printing purposes
        """
        if command_str == "all":
            selected_task_list = self.task_list
            # selected_tasks = self.sequential_tasks
            if verbosity >= 1:
                print(f"Running everything")
        elif command_str == "none":
            selected_task_list = []
            if verbosity >= 2:
                print(f"Running nothing")
        else:
            raise ValueError(
                f"{self.name} (TaskPipeline)'s method submit_scripts was called "
                f"with command_str={command_str}, but this is not currently supported "
                f"for the generic implementation of TaskPipeline"
            )    
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
