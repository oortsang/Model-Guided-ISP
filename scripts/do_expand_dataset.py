# Regenerate fields dropped by scripts/do_shrink_dataset.py
# (q_polar in scattering-object files; q_cart, q_polar, q_cart_lpf,
# q_polar_lpf, d_mh in measurement files), mutating the dataset in place.
#
# Only fills in fields that are actually missing by default; pass
# --exists-ok to force recomputing/overwriting fields that are already
# present.
#
# Example (run from the repo root):
#   python scripts/do_expand_dataset.py \
#       --base-dir /path/to/shrunk_dataset \
#       --nu 1 2 3 --subsets train val test --backend jax

import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.layout import RefDatasetLayout
from src.data.data_io import load_field_in_hdf5
from src.data.data_naming_constants import X_VALS, THETA_VALS, RHO_VALS, H_VALS, NU_SF
from src.data.expand_dataset import (
    DatasetTransforms,
    FrequencyTransforms,
    expand_scobj_hdf5,
    expand_meas_hdf5,
)
from src.utils.loading import jax_setup, logging_setup


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    bool_choices = ["true", "false"]
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory of the (shrunk) dataset to expand in place",
    )
    parser.add_argument(
        "--nu",
        type=str,
        nargs="+",
        required=True,
        help="nu (non-angular wavenumber) values as strings",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Which subsets to expand",
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "jax"],
        default="jax",
        help="Backend for the interpolation operators; see "
        "src/data/expand_dataset.py",
    )
    parser.add_argument(
        "--sparse-x",
        choices=bool_choices,
        default="true",
        help="Only used with --backend jax: makes the x-side interpolation "
        "operator sparse (BCSR); y is always dense regardless",
    )
    parser.add_argument(
        "--vram-fraction",
        type=float,
        default=0.8,
        help="Fraction of GPU VRAM for JAX to preallocate; only used with "
        "--backend jax. Passed through to src.utils.loading.jax_setup.",
    )
    parser.add_argument(
        "--jax-cpu-only",
        choices=bool_choices,
        default="false",
        help="Force JAX onto CPU regardless of GPU availability; only used "
        "with --backend jax.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device for the torch backend, e.g. 'cuda' or 'cpu' "
        "(default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--exists-ok",
        choices=bool_choices,
        default="false",
        help="Force recompute/overwrite fields that are already present. "
        "Off by default -- only fills in whatever's missing, and reuses "
        "q_cart_lpf instead of recomputing it if only q_polar_lpf needs "
        "filling in.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    a = parser.parse_args()

    bool_args = ["sparse_x", "jax_cpu_only", "exists_ok"]
    for bool_arg in bool_args:
        setattr(a, bool_arg, getattr(a, bool_arg) == "true")

    return a


def main(args: argparse.Namespace) -> None:
    layout = RefDatasetLayout(
        base_dir=args.base_dir,
        nu_str_list=args.nu,
        subset_list=args.subsets,
    )

    # Must run before jax is ever imported anywhere in the process (it just
    # sets env vars/config), so this needs to happen before the first call
    # to DatasetTransforms.from_grids(..., backend="jax", ...) below, which
    # is where expand_dataset.py lazily imports jax. Only bother when the
    # jax backend is actually requested, so a torch-backend run doesn't
    # reserve GPU memory for jax for no reason.
    if args.backend == "jax":
        jax_device = jax_setup(cpu_only=args.jax_cpu_only, vram_fraction=args.vram_fraction)
        logging.info(f"Jax device: {jax_device}")

    device = args.device
    if device is None and args.backend == "torch":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device) if device is not None else None

    # Grid metadata is never dropped by shrink_dataset.py, so it's safe to
    # read up front from the first available files. Read individual fields
    # (not the whole file) to avoid pulling in the large q_cart/d_rs arrays.
    first_subset = args.subsets[0]
    first_nu = args.nu[0]
    scobj_files_0 = layout.get_scobj_files(first_subset)
    meas_files_0 = layout.get_meas_files(first_subset, first_nu)
    if not scobj_files_0 or not meas_files_0:
        raise SystemExit(f"No files found for subset={first_subset!r}, nu={first_nu!r}")

    x_vals = load_field_in_hdf5(X_VALS, scobj_files_0[0])
    theta_vals = load_field_in_hdf5(THETA_VALS, scobj_files_0[0])
    rho_vals = load_field_in_hdf5(RHO_VALS, scobj_files_0[0])
    h_vals = load_field_in_hdf5(H_VALS, meas_files_0[0])

    logging.info(
        f"Building dataset-wide transforms (backend={args.backend}, "
        f"sparse_x={args.sparse_x}, device={device})"
    )
    dataset_transforms = DatasetTransforms.from_grids(
        x_vals, theta_vals, rho_vals, h_vals,
        backend=args.backend, sparse_x=args.sparse_x, device=device,
    )

    n_expanded = 0
    n_failed = 0

    # 1. Scattering-object files (q_polar), independent of nu
    for subset_name in args.subsets:
        scobj_files = layout.get_scobj_files(subset_name)
        logging.info(f"[{subset_name}] expanding {len(scobj_files)} scattering-object files")
        for scobj_fp in scobj_files:
            try:
                expand_scobj_hdf5(scobj_fp, dataset_transforms, exists_ok=args.exists_ok)
                n_expanded += 1
            except Exception:
                logging.exception(f"Failed to expand {scobj_fp}")
                n_failed += 1

    # 2. Measurement files (q_cart, q_polar, q_cart_lpf, q_polar_lpf, d_mh) -- one
    # FrequencyTransforms built per nu, reused across every subset for that nu
    for nu_str in args.nu:
        freq_transforms = None
        for subset_name in args.subsets:
            meas_files = layout.get_meas_files(subset_name, nu_str)
            if not meas_files:
                continue

            if freq_transforms is None:
                # nu_sf is stored as a 0-d scalar in most datasets, but as a
                # shape-(1,) array in some (e.g. the OOD contrast test
                # sets) -- np.asarray(...).reshape(-1)[0] handles either.
                nu_sf = float(np.asarray(load_field_in_hdf5(NU_SF, meas_files[0])).reshape(-1)[0])
                freq_transforms = FrequencyTransforms.from_nu(nu_sf, num_x=x_vals.shape[0])
                logging.info(f"[nu={nu_str}] nu_sf={nu_sf}")

            scobj_files = layout.get_scobj_files(subset_name)
            if len(scobj_files) != len(meas_files):
                raise ValueError(
                    f"[{subset_name}, nu={nu_str}] scobj/meas file counts "
                    f"differ: {len(scobj_files)} vs {len(meas_files)}"
                )

            logging.info(f"[{subset_name}, nu={nu_str}] expanding {len(meas_files)} measurement files")
            for scobj_fp, meas_fp in zip(scobj_files, meas_files):
                try:
                    expand_meas_hdf5(
                        meas_fp, scobj_fp, dataset_transforms, freq_transforms,
                        exists_ok=args.exists_ok,
                    )
                    n_expanded += 1
                except Exception:
                    logging.exception(f"Failed to expand {meas_fp}")
                    n_failed += 1

    logging.info(f"Finished: {n_expanded} files expanded, {n_failed} failed")
    if n_failed:
        raise SystemExit(f"{n_failed} file(s) failed to expand; see log above for details")


if __name__ == "__main__":
    a = setup_args()
    logging_setup("do_expand_dataset", level=logging.DEBUG if a.debug else logging.INFO)
    main(a)
