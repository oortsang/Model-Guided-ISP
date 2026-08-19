# shrink.py
# Introduced 2026-08-03 to shrink the dataset size

from dataclasses import dataclass
from typing import Optional, Iterable, Union
import collections
import functools
import logging
import multiprocessing as mp
import os

import h5py
import numpy as np

from src.data.data_naming_constants import (
    Q_CART,
    Q_POLAR,
    Q_CART_LPF,
    Q_POLAR_LPF,
    SAMPLE_COMPLETION,
    D_MH,
    TRUNCATABLE_KEYS,
)
from src.data.layout import RefDatasetLayout, find_files_index_range


# Default codec applied to newly-compressed datasets when compress=True is
# requested. Only ever applied to fields that arrive uncompressed (see
# copy_and_shrink_hdf5) -- fields already compressed in the source are
# always copied through with their existing filter untouched.
#
# This is HDF5's *built-in* byte-shuffle + gzip
# On real data, gzip level 6 + shuffle gets ~24-29% smaller
DEFAULT_SHUFFLE = True
DEFAULT_GZIP_LEVEL = 6
DEFAULT_COMPRESSION = dict(compression="gzip", compression_opts=DEFAULT_GZIP_LEVEL)

# Chunk size (in samples along axis 0) used for newly-compressed datasets.
# Benchmarked against h5py's auto-chunking and a single whole-array chunk on
# a real scattering-object file: 10 samples/chunk matched the best
# compression ratio of any option tried while giving the fastest and most
# consistent read times (whole-array chunking compressed marginally better
# but was ~30% slower to read, since the whole chunk must be decompressed
# before any of it is usable).
COMPRESSION_CHUNK_SAMPLES = 10

@dataclass
class CopyJob:
    """One planned per-file copy. sample_slice is applied only to fields in
    TRUNCATABLE_KEYS (i.e. fields that are indexed per-sample); everything
    else is copied in full. subset_name/group_label ("scobj" or
    "nu=<nu_str>") exist purely for verbosity-level progress reporting in
    run_copy_jobs -- they don't affect what gets copied.
    """
    src_fp: str
    dst_fp: str
    drop_fields: set
    sample_slice: slice
    subset_name: str
    group_label: str
    compress: bool = False

def get_scobj_drop_fields(
    drop_lpf: bool = True,
    drop_polar: bool = True,
) -> set:
    """Get the set of fields to drop for scattering object files.
    These fields can be regenerated later (at the cost of a few minutes on
    a GPU), so dropping them is a reversible space-saving choice.
    """
    drop_list = []
    if drop_lpf:
        drop_list += [Q_CART_LPF, Q_POLAR_LPF]
    if drop_polar:
        drop_list += [Q_POLAR, Q_POLAR_LPF]

    drop_fields = set(drop_list)
    return drop_fields

def get_meas_drop_fields(
    drop_scobj: bool = True,
    drop_d_mh: bool = False,
) -> set:
    """Get the set of fields to drop for measurement files.
    drop_scobj removes fields that are redundant with the scattering-object
    files (q_cart, q_polar, and their LPF variants), since those are already
    stored there.
    """
    drop_list = []
    if drop_scobj:
        drop_list += [
            Q_CART,
            Q_POLAR,
            Q_CART_LPF,
            Q_POLAR_LPF,
        ]
    if drop_d_mh:
        drop_list += [D_MH]

    drop_fields = set(drop_list)
    return drop_fields

def _dst_already_sufficient(
    dst_fp: str, expected_keys: set, sample_slice: slice, src_check_len: int,
) -> bool:
    """Checks whether an existing dst_fp already has exactly the expected
    fields and at least as many samples as sample_slice would request, so
    that re-copying it would be redundant.
    """
    with h5py.File(dst_fp, "r") as dst_check:
        existing_keys = set(dst_check.keys())
        if existing_keys != expected_keys or SAMPLE_COMPLETION not in existing_keys:
            return False
        _, needed_n, _ = sample_slice.indices(src_check_len)
        existing_n = dst_check[SAMPLE_COMPLETION].shape[0]
        return existing_n >= needed_n

def copy_and_shrink_hdf5(
    src_fp: str,
    dst_fp: str,
    drop_fields: Iterable[str] = (),
    sample_slice: slice = slice(None),
    sample_keys: Optional[Iterable[str]] = None,
    copy_batch_size: int = 1_000,
    exists_ok: bool = False,
    skip_if_sufficient: bool = False,
    compress: bool = False,
) -> bool:
    """Copies the source hdf5 file into the destination while dropping some
    of the fields (to shrink the space usage), and optionally truncating
    the sample-indexed fields to `sample_slice` (mainly intended for
    building small datasets for testing purposes).

    Parameters:
        src_fp (str): path to the source hdf5 file
        dst_fp (str): path to the destination hdf5 file (will be created;
            refuses to overwrite an existing file unless exists_ok=True)
        drop_fields (Iterable[str]): names of fields to skip entirely
        sample_slice (slice): slice (in this file's local index space) to
            apply to sample-indexed fields; defaults to keeping everything
        sample_keys (Iterable[str]): which fields are considered sample-indexed
            (and thus subject to sample_slice); defaults to TRUNCATABLE_KEYS
        copy_batch_size (int): number of rows to copy at a time, to avoid
            pulling an entire large dataset into memory at once
        exists_ok (bool): if False (default), raise FileExistsError instead
            of overwriting an existing dst_fp -- protects against clobbering
            data due to a mistaken/overlapping destination path
        skip_if_sufficient (bool): if True, and dst_fp already exists with
            exactly the expected fields and at least as many samples as
            requested, skip entirely (no read, no write) instead of
            re-copying -- lets a re-run with a bigger truncate_num avoid
            redoing files it already finished. This check runs before (and
            is independent of) the exists_ok guard, since skipping never
            touches dst_fp. Off by default.
        compress (bool): if True, apply DEFAULT_COMPRESSION (HDF5's native
            byte-shuffle + gzip filters, both built into libhdf5 core) to
            array fields that arrive uncompressed from the source. Fields
            that are already compressed are always copied through with
            their existing filter unchanged -- this never re-compresses or
            double-compresses. Reading the result back needs no special
            setup at all -- gzip requires no plugin, no HDF5_PLUGIN_PATH,
            nothing; plain h5py.File(...) calls anywhere in the codebase
            (e.g. src/data/data_io.py) just work. Off by default --
            unchanged behavior.

    Returns:
        bool: True if dst_fp was (re)written, False if it was skipped.
    """
    drop_set = set(drop_fields)
    sample_key_set = (
        set(sample_keys) if sample_keys is not None else set(TRUNCATABLE_KEYS)
    )

    with h5py.File(src_fp, "r") as src:
        expected_keys = {
            name for name, obj in src.items()
            if isinstance(obj, h5py.Dataset) and name not in drop_set
        }

        if (
            skip_if_sufficient
            and os.path.exists(dst_fp)
            and SAMPLE_COMPLETION in src
            and _dst_already_sufficient(
                dst_fp, expected_keys, sample_slice, src[SAMPLE_COMPLETION].shape[0],
            )
        ):
            logging.debug(f"{dst_fp}: already sufficient, skipping")
            return False

        if not exists_ok and os.path.exists(dst_fp):
            raise FileExistsError(
                f"Destination file {dst_fp} already exists; pass exists_ok=True "
                f"to overwrite it."
            )

        os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
        with h5py.File(dst_fp, "w") as dst:
            dst.attrs.update(src.attrs)
            for name, obj in src.items():
                if name in drop_set or not isinstance(obj, h5py.Dataset):
                    continue

                if obj.shape and name in sample_key_set:
                    start, stop, _ = sample_slice.indices(obj.shape[0])
                else:
                    start, stop = 0, (obj.shape[0] if obj.shape else 0)
                out_len = stop - start
                out_shape = (out_len, *obj.shape[1:]) if obj.shape else obj.shape

                # Only newly apply compression to fields that arrive
                # uncompressed and aren't scalars (compression needs
                # chunking, which scalar datasets can't have); anything
                # already compressed is passed through untouched.
                apply_new_compression = (
                    compress and obj.shape and obj.compression is None
                )
                if apply_new_compression:
                    chunk_shape = (
                        max(1, min(COMPRESSION_CHUNK_SAMPLES, out_shape[0])),
                        *out_shape[1:],
                    )
                    dst_dset = dst.create_dataset(
                        name,
                        shape=out_shape,
                        dtype=obj.dtype,
                        chunks=chunk_shape,
                        shuffle=DEFAULT_SHUFFLE,
                        **DEFAULT_COMPRESSION,
                    )
                else:
                    if obj.compression == "unknown":
                        raise NotImplementedError(
                            f"Field {name!r} in {src_fp} uses an "
                            f"externally-filtered (non-gzip) compression "
                            f"that copy_and_shrink_hdf5 can't forward. "
                            f"Re-source this field from an uncompressed "
                            f"or gzip-compressed copy."
                        )
                    dst_dset = dst.create_dataset(
                        name,
                        shape=out_shape,
                        dtype=obj.dtype,
                        chunks=obj.chunks,
                        compression=obj.compression,
                        compression_opts=obj.compression_opts,
                    )
                dst_dset.attrs.update(obj.attrs)

                if obj.shape == ():
                    dst_dset[()] = obj[()]
                else:
                    # Copy over batch-wise to limit memory usage
                    for batch_start in range(start, stop, copy_batch_size):
                        batch_end = min(batch_start + copy_batch_size, stop)
                        dst_dset[batch_start-start:batch_end-start] = obj[
                            batch_start:batch_end
                        ]
    return True

def plan_directory_copy(
    src_dir: str,
    dst_dir: str,
    drop_fields: Iterable[str],
    truncate_num: Optional[int] = None,
    exists_ok: bool = False,
    subset_name: str = "",
    group_label: str = "",
    compress: bool = False,
) -> list[CopyJob]:
    """Plans the per-file copy jobs needed to shrink (and optionally
    truncate) one directory of indexed .h5 files.

    When truncate_num is set, only as many files as are needed to reach
    truncate_num samples are included -- later files in the directory are
    left untouched entirely, which is the point when this is used for
    building a small dataset for testing.

    Refuses to plan into an already-existing dst_dir unless exists_ok=True --
    this is a fail-fast check (before any copying starts) independent of the
    per-file check in copy_and_shrink_hdf5.

    subset_name/group_label are attached to each CopyJob purely for
    run_copy_jobs' verbosity-level progress reporting.
    """
    if not exists_ok and os.path.exists(dst_dir):
        raise FileExistsError(
            f"Destination directory {dst_dir} already exists; pass "
            f"exists_ok=True to write into it anyway."
        )

    drop_fields = set(drop_fields)
    file_fps, local_slices = find_files_index_range(src_dir, 0, truncate_num)
    jobs = [
        CopyJob(
            src_fp=src_fp,
            dst_fp=os.path.join(dst_dir, os.path.basename(src_fp)),
            drop_fields=drop_fields,
            sample_slice=sl,
            subset_name=subset_name,
            group_label=group_label,
            compress=compress,
        )
        for src_fp, sl in zip(file_fps, local_slices)
    ]
    return jobs

def _truncate_num_for_subset(
    truncate_num: Optional[Union[int, dict]], subset_name: str
) -> Optional[int]:
    """truncate_num may be a single int (applied to every subset) or a dict
    keyed by subset_name (e.g. {"train": 200, "val": 50, "test": 50}) so
    that train/val/test can be truncated to different sizes for testing.
    """
    if truncate_num is None:
        return None
    if isinstance(truncate_num, dict):
        return truncate_num.get(subset_name)
    return truncate_num

def plan_dataset_copy(
    src_layout: RefDatasetLayout,
    dst_layout: RefDatasetLayout,
    truncate_num: Optional[Union[int, dict]] = None,
    drop_lpf: bool = True,
    drop_polar: bool = True,
    drop_scobj_from_meas: bool = True,
    drop_d_mh: bool = False,
    exists_ok: bool = False,
    compress: bool = False,
) -> list[CopyJob]:
    """Plans the full set of per-file copy jobs to shrink (and optionally
    truncate) a dataset laid out according to src_layout, writing the
    result into dst_layout.

    exists_ok is forwarded to plan_directory_copy for every scobj/meas
    directory involved; see there for details. Default False (safe).
    compress is forwarded to every copy job; see copy_and_shrink_hdf5 for
    what it does. Default False (unchanged behavior).
    """
    scobj_drop_fields = get_scobj_drop_fields(drop_lpf=drop_lpf, drop_polar=drop_polar)
    meas_drop_fields = get_meas_drop_fields(
        drop_scobj=drop_scobj_from_meas, drop_d_mh=drop_d_mh
    )

    jobs: list[CopyJob] = []
    for subset_name in src_layout.subset_list:
        subset_truncate_num = _truncate_num_for_subset(truncate_num, subset_name)
        jobs += plan_directory_copy(
            src_layout.get_scobj_dir(subset_name),
            dst_layout.get_scobj_dir(subset_name),
            scobj_drop_fields,
            truncate_num=subset_truncate_num,
            exists_ok=exists_ok,
            subset_name=subset_name,
            group_label="scobj",
            compress=compress,
        )
        for nu_str in src_layout.nu_str_list:
            jobs += plan_directory_copy(
                src_layout.get_meas_dir(subset_name, nu_str),
                dst_layout.get_meas_dir(subset_name, nu_str),
                meas_drop_fields,
                truncate_num=subset_truncate_num,
                exists_ok=exists_ok,
                subset_name=subset_name,
                group_label=f"nu={nu_str}",
                compress=compress,
            )
    return jobs

@dataclass
class CopyResult:
    src_fp: str
    dst_fp: str
    err: Optional[str]
    subset_name: str
    group_label: str
    written: bool = True  # False if skip_if_sufficient caused a skip


def _run_copy_job(
    job: CopyJob, exists_ok: bool = False, skip_if_sufficient: bool = False,
) -> CopyResult:
    err = None
    written = True
    try:
        written = copy_and_shrink_hdf5(
            job.src_fp,
            job.dst_fp,
            job.drop_fields,
            sample_slice=job.sample_slice,
            exists_ok=exists_ok,
            skip_if_sufficient=skip_if_sufficient,
            compress=job.compress,
        )
    except Exception as e:
        err = str(e)
    return CopyResult(
        src_fp=job.src_fp, dst_fp=job.dst_fp, err=err,
        subset_name=job.subset_name, group_label=job.group_label, written=written,
    )

def run_copy_jobs(
    jobs: list[CopyJob],
    n_workers: int = 4,
    exists_ok: bool = False,
    skip_if_sufficient: bool = False,
    verbosity_level: int = 1,
) -> list[CopyResult]:
    """Executes the planned copy jobs, optionally in parallel.

    The dataset lives on an NFS-mounted file system, where the bottleneck
    is typically metadata/IO contention on the file server rather than
    local CPU -- so n_workers is deliberately kept modest by default
    (a handful of worker processes) rather than scaling with os.cpu_count().
    Tune it up or down based on what you observe.

    exists_ok/skip_if_sufficient are forwarded to copy_and_shrink_hdf5 for
    every job. exists_ok is the per-file overwrite guard, independent of
    (and a fallback for) the directory-level check in plan_directory_copy.
    Both default False (safe/unchanged behavior).

    verbosity_level controls progress logging (failures always log,
    regardless of this setting):
        0: silent
        1 (default): log once a subset (all its scobj + every nu) is done
        2: also log once each group (scobj, or one nu) within a subset is done
        3: also log after every individual file
    """
    logging.info(f"Running {len(jobs)} copy jobs with n_workers={n_workers}")
    subset_totals = collections.Counter(j.subset_name for j in jobs)
    group_totals = collections.Counter((j.subset_name, j.group_label) for j in jobs)
    subset_counts = collections.Counter()
    group_counts = collections.Counter()

    def handle_result(result: CopyResult) -> None:
        subset_counts[result.subset_name] += 1
        group_counts[(result.subset_name, result.group_label)] += 1
        if verbosity_level >= 3:
            action = "wrote" if result.written else "skipped (already sufficient)"
            logging.info(f"[{result.subset_name}/{result.group_label}] {action} {result.dst_fp}")
        if verbosity_level >= 2 and group_counts[(result.subset_name, result.group_label)] == group_totals[(result.subset_name, result.group_label)]:
            logging.info(f"[{result.subset_name}] finished group {result.group_label}")
        if verbosity_level >= 1 and subset_counts[result.subset_name] == subset_totals[result.subset_name]:
            logging.info(f"finished subset {result.subset_name}")

    run_job = functools.partial(_run_copy_job, exists_ok=exists_ok, skip_if_sufficient=skip_if_sufficient)
    results = []
    if n_workers <= 1:
        for job in jobs:
            result = run_job(job)
            results.append(result)
            handle_result(result)
    else:
        with mp.Pool(n_workers) as pool:
            for result in pool.imap_unordered(run_job, jobs):
                results.append(result)
                handle_result(result)

    failures = [r for r in results if r.err is not None]
    for r in failures:
        logging.error(f"Failed to copy {r.src_fp} -> {r.dst_fp}: {r.err}")
    if failures:
        logging.error(f"{len(failures)}/{len(jobs)} copy jobs failed")
    return results


def copy_and_shrink_dataset(
    src_layout: RefDatasetLayout,
    dst_layout: RefDatasetLayout,
    truncate_num: Optional[Union[int, dict]] = None,
    n_workers: int = 4,
    drop_lpf: bool = True,
    drop_polar: bool = True,
    drop_scobj_from_meas: bool = True,
    drop_d_mh: bool = True,
    exists_ok: bool = False,
    skip_if_sufficient: bool = False,
    verbosity_level: int = 1,
    compress: bool = False,
) -> list[CopyResult]:
    """Copies a dataset laid out according to src_layout into dst_layout,
    dropping the requested fields to shrink the space usage.

    truncate_num can be used to build a small dataset for testing purposes:
    pass an int to cap every subset (train/val/test) at that many samples,
    or a dict keyed by subset name (e.g. {"train": 200, "val": 50, "test": 50})
    to cap them individually. Files beyond what's needed to reach the cap
    are skipped entirely, so this stays cheap even against a huge dataset.

    exists_ok (bool): if False (default), refuses to write into any
        destination directory or file that already exists -- protects
        against clobbering data (e.g. the original dataset) due to a
        mistaken/overlapping dst_layout. Pass True to explicitly allow
        overwriting.
    skip_if_sufficient (bool): if True, files that already have the right
        fields and enough samples are left untouched instead of being
        re-copied -- makes re-running with a bigger truncate_num only do the
        incremental work. Off by default; see copy_and_shrink_hdf5.
    verbosity_level (int): 0=silent, 1=per-subset (default), 2=also
        per-group (scobj / each nu), 3=also per-file. See run_copy_jobs.
    compress (bool): if True, apply DEFAULT_COMPRESSION (HDF5's native
        byte-shuffle + gzip filters -- both built into libhdf5 core, no
        plugin needed) to array fields that arrive uncompressed from the
        source. See copy_and_shrink_hdf5 for details. Off by default.
    """
    jobs = plan_dataset_copy(
        src_layout,
        dst_layout,
        truncate_num=truncate_num,
        drop_lpf=drop_lpf,
        drop_polar=drop_polar,
        drop_scobj_from_meas=drop_scobj_from_meas,
        drop_d_mh=drop_d_mh,
        exists_ok=exists_ok,
        compress=compress,
    )
    logging.info(f"Planned {len(jobs)} file copy jobs")
    return run_copy_jobs(
        jobs, n_workers=n_workers, exists_ok=exists_ok,
        skip_if_sufficient=skip_if_sufficient, verbosity_level=verbosity_level,
    )


def copy_dataset(
    src_layout: RefDatasetLayout,
    dst_layout: RefDatasetLayout,
    truncate_num: Optional[Union[int, dict]] = None,
    n_workers: int = 4,
    exists_ok: bool = False,
    skip_if_sufficient: bool = False,
    verbosity_level: int = 1,
) -> list[CopyResult]:
    """Plain copy of a dataset (no fields dropped) -- a thin specialization
    of copy_and_shrink_dataset with every drop-flag off. Mainly useful for
    testing: clone a dataset to a scratch destination before mutating it
    in place (e.g. with expand_dataset.py), so the original is never at risk.
    """
    return copy_and_shrink_dataset(
        src_layout,
        dst_layout,
        truncate_num=truncate_num,
        n_workers=n_workers,
        drop_lpf=False,
        drop_polar=False,
        drop_scobj_from_meas=False,
        drop_d_mh=False,
        exists_ok=exists_ok,
        skip_if_sufficient=skip_if_sufficient,
        verbosity_level=verbosity_level,
    )
