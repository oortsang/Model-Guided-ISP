# src/data/layout.py
# Help manage the file system layout of the dataset
# for a more user-friendly interface
# Note: not fully integrated into existing code
# since this is a newer addition

from dataclasses import dataclass, field
from typing import Optional
import os
import re

import h5py
import numpy as np

from src.data.data_naming_constants import SAMPLE_COMPLETION

### Helpers for the on-disk file-naming convention ###
# Every scattering-object/measurement data file is named like
# "..._{start_idx}.h5", where start_idx is the global sample index of the
# first sample stored in that file. These helpers live here (rather than in
# data_io.py) because they're about how files are laid out/named on disk,
# not about reading/writing HDF5 contents -- data_io.py builds on top of them.

def _file_start_idx(filename: str) -> int:
    """Extracts the leading sample index from a file name of the form
    '..._{number}.h5'
    """
    try:
        stem = filename[:-len(".h5")] if filename.endswith(".h5") else filename
        return int(stem.split("_")[-1])
    except (ValueError, IndexError):
        raise ValueError(
            f"_file_start_idx: unable to parse an index from filename '{filename}'"
        )


def list_h5_files(dir_name: str) -> list[str]:
    """Lists the indexed .h5 files in a directory, sorted by their leading
    sample index, and returned as full paths.
    """
    fnames = [
        fname for fname in os.listdir(dir_name)
        if len(re.findall("[0-9]+", fname.replace(".h5", ""))) >= 1
    ]
    fnames = sorted(fnames, key=_file_start_idx)
    return [os.path.join(dir_name, fname) for fname in fnames]


def get_file_start_index(file_path: str, use_file_name: bool = True) -> int:
    """There may be multiple scattering object or measurement files in a
    directory, so this figures out the starting global sample index of a
    given file: either by trusting the file name (fast), or by tallying up
    the sample counts of the files that precede it in the directory (slow,
    in case the file names can't be trusted).
    """
    if use_file_name:
        start_index = _file_start_idx(os.path.basename(file_path))
    else:
        dir_name, file_name = os.path.split(file_path)
        file_list = list_h5_files(dir_name)
        sample_count = 0
        for file_i in file_list:
            if os.path.basename(file_i) == file_name:
                break
            with h5py.File(file_i, "r") as hf:
                sample_count += hf[SAMPLE_COMPLETION].shape[0]
        start_index = sample_count
    return start_index


def find_files_index_range(
    dir_name: str,
    global_idx_start: int = 0,
    global_idx_end: Optional[int] = None,
) -> tuple[list[str], list[slice]]:
    """For a given directory, return the list of files (and corresponding
    local slices into each) needed to cover global sample indices
    [global_idx_start, global_idx_end) of the directory as a whole.
    """
    file_list = list_h5_files(dir_name)
    if not file_list:
        return [], []

    file_index_list = [
        get_file_start_index(fp, use_file_name=True) for fp in file_list
    ]

    # Also tally on the total sample count so slices can be computed for
    # the last file, and so global_idx_end can default to "everything"
    with h5py.File(file_list[-1], "r") as hf:
        last_file_n_samples = hf[SAMPLE_COMPLETION].shape[0]
    sample_count = file_index_list[-1] + last_file_n_samples
    file_index_list.append(sample_count)
    global_idx_end = global_idx_end if global_idx_end is not None else sample_count

    file_index_arr = np.array(file_index_list)
    file_index_starts = file_index_arr[:-1]
    file_index_ends = file_index_arr[1:]

    # Which files overlap the requested [global_idx_start, global_idx_end) range
    valid_files_bools = np.logical_and(
        file_index_ends > global_idx_start,
        file_index_starts <= global_idx_end,
    )
    valid_starts = file_index_starts[valid_files_bools]
    valid_ends = file_index_ends[valid_files_bools]
    valid_file_fps = [
        fp for (fp, keep) in zip(file_list, valid_files_bools) if keep
    ]

    # Slices, in each file's own local index space
    local_slices = [
        slice(max(fs, global_idx_start) - fs, min(fe, global_idx_end) - fs)
        for (fs, fe) in zip(valid_starts, valid_ends)
    ]
    return valid_file_fps, local_slices


@dataclass
class RefDatasetLayout:
    base_dir: str # Safest to give an absolute path
    nu_str_list: list[str] # store as string in case of decimals
    subset_list: list[str]

    # Templates for the scobj/meas directory names
    scobj_dir_tmpl: str = "{subset_name}_scattering_objs"
    meas_dir_tmpl: str = "{subset_name}_measurements_nu_{nu}"

    # work_dir: str = None # where mmg could be found
    # mmg_dir_tmpl: str = "{subset_name}_gammas_nu_{nu}"

    @classmethod
    def from_basic_config(
        cls,
        base_dir: str="dataset",
        **kwargs,
    ):
        """Default setup with nu=1,2,...,10 and train/val/test sets"""
        nu_list = range(1,1+10)
        nu_str_list = [str(nu) for nu in nu_list]
        return cls(
            base_dir=base_dir,
            nu_str_list=nu_str_list,
            subset_list=["train", "val", "test"],
            **kwargs,
        )

    # def set_work_dir(self, work_dir: str):
    #     """Set the directory for finding """
    #     self.work_dir = work_dir

    def _check_subset_name(self, subset_name: str) -> None:
        """Helper to ensure subset is recognized"""
        if subset_name not in self.subset_list:
            raise ValueError(
                f"DatasetLayoutMultifreq does not recognize subset "
                f"{subset_name}; expected one of {self.subset_list}."
            )

    def get_scobj_dir(self, subset_name: str) -> list[str]:
        self._check_subset_name(subset_name)
        ref_scobj_dir = os.path.join(
            self.base_dir,
            self.scobj_dir_tmpl.format(subset_name=subset_name)
        )
        return ref_scobj_dir

    def get_meas_dir(
        self, subset_name: str, nu_str: str=None, check_subset: bool=True,
    ) -> list[str]:
        if check_subset:
            self._check_subset_name(subset_name)
        meas_dir_name = os.path.join(
            self.base_dir,
            self.meas_dir_tmpl.format(subset_name=subset_name, nu=nu_str)
        )
        return meas_dir_name

    def get_all_meas_dirs(self, subset_name: str) -> list[str]:
        self._check_subset_name(subset_name)
        all_meas_dirs = [
            self.get_meas_dir(
                subset_name=subset_name,
                nu_str=nu_str,
                check_subset=False, # redundant
            )
            for nu_str in self.nu_str_list
        ]
        return all_meas_dirs

    def get_scobj_files(self, subset_name: str) -> list[str]:
        self._check_subset_name(subset_name)
        return list_h5_files(self.get_scobj_dir(subset_name))

    def get_meas_files(self, subset_name: str, nu_str: str) -> list[str]:
        self._check_subset_name(subset_name)
        return list_h5_files(self.get_meas_dir(subset_name, nu_str))

# @dataclass
# class PredsDatasetLayout:
#     """Dataset layout manager for predictions
#     Note: designed for a single frequency at a time,
#     otherwise this is likely unnecessary.

#     Hmmmm, in practice I had not unified these within the predictions directory
#     """
#     ref_dset_layout: RefDatasetLayout
#     subset_list: list[str]

#     work_dir: str = None # where mmg could be found
#     pred_scobj_dir_tmpl: str = "{subset_name}_gammas_nu_{nu}"
#     pred_mmg_dir_tmpl: str = "{subset_name}_gammas_nu_{nu}"

#     def _check_subset_name(self, subset_name: str) -> None:
#         """Helper to ensure subset is recognized"""
#         if subset_name not in self.subset_list:
#             raise ValueError(
#                 f"DatasetLayoutMultifreq does not recognize subset "
#                 f"{subset_name}; expected one of {self.subset_list}."
#             )

#     def get_pred_mmg_dir(self, subset_name: str, nu_str: str):
#         """For use when splitting a training/eval run across multiple jobs
#         and saving
#         """
#         self._check_subset_name(subset_name)
#         pred_mmg_dir = os.path.join(
#             predictions_dir,
#             self.mmg_dir_tmpl.format(subset_name=subset_name, nu=nu_str),
#         )
#         return pred_pred_mmg_dir

#     def get_pred_scobj_dir(self, subset_name: str, nu_str: str):
#         """For use when splitting a training/eval run across multiple jobs
#         and saving
#         """
#         self._check_subset_name(subset_name)
#         pred_mmg_dir = os.path.join(
#             predictions_dir,
#             self.mmg_dir_tmpl.format(subset_name=subset_name, nu=nu_str),
#         )
#         return pred_pred_mmg_dir
