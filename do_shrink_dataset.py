# Shrink a dataset laid out per RefDatasetLayout by dropping fields that can
# later be regenerated with do_expand_dataset.py.
#
# Example (uniform truncation across every listed subset):
#   python do_shrink_dataset.py \
#       --src-base-dir /path/to/dataset \
#       --dst-base-dir /path/to/shrunk_dataset \
#       --nu 1 2 3 --subsets train val test \
#       --truncate-num 500 --n-workers 4
#
# Example (per-subset truncation, e.g. a bigger train set than val/test):
#   python do_shrink_dataset.py \
#       --src-base-dir /path/to/dataset \
#       --dst-base-dir /path/to/shrunk_dataset \
#       --nu 1 2 3 --subsets train val test \
#       --truncate-num train=10000 val=1000 test=1000

import argparse
import logging

from src.data.layout import RefDatasetLayout
from src.data.shrink_dataset import copy_and_shrink_dataset
from src.utils.logging_utils import FMT, TIMEFMT


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    bool_choices = ["true", "false"]
    parser.add_argument(
        "--src-base-dir",
        type=str,
        required=True,
        help="Base directory of the source (full) dataset",
    )
    parser.add_argument(
        "--dst-base-dir",
        type=str,
        required=True,
        help="Base directory to write the shrunk dataset to",
    )
    parser.add_argument(
        "--nu",
        type=str,
        nargs="+",
        required=True,
        help="nu (non-angular wavenumber) values as strings, "
        "e.g. --nu 1 2 3 or --nu 4.5",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Which subsets to shrink",
    )
    parser.add_argument(
        "--truncate-num",
        type=str,
        nargs="+",
        default=None,
        help="Cap subsets at this many samples; useful for building a small "
        "test dataset. Either a single int applied to every listed subset "
        "(e.g. --truncate-num 1000), or one or more subset=int pairs to set "
        "them individually (e.g. --truncate-num train=10000 val=1000 "
        "test=1000). Files beyond what's needed are left untouched. A value "
        "larger than what's actually available for a subset is safe -- it "
        "just includes everything for that subset, no error.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of worker processes for the copy; kept modest by "
        "default since the dataset lives on NFS",
    )

    # Drop-field settings (mirror copy_and_shrink_dataset's flags directly)
    parser.add_argument(
        "--drop-lpf",
        choices=bool_choices,
        default="true",
        help="Drop q_cart_lpf/q_polar_lpf in the scattering-object files",
    )
    parser.add_argument(
        "--drop-polar",
        choices=bool_choices,
        default="true",
        help="Drop q_polar/q_polar_lpf in the scattering-object files",
    )
    parser.add_argument(
        "--drop-scobj-fields-in-meas",
        choices=bool_choices,
        default="true",
        help="Drop q_cart/q_polar (and LPF variants) in the measurement "
        "files, even though they're already stored in the "
        "scattering-object files",
    )
    parser.add_argument(
        "--drop-d-mh",
        choices=bool_choices,
        default="true",
        help="Drop d_mh from the measurement files (off by default -- "
        "d_mh is comparatively cheap to keep and expensive to regenerate "
        "accurately)",
    )

    parser.add_argument(
        "--compress",
        choices=bool_choices,
        default="true",
        help="Apply HDF5's native byte-shuffle + gzip filters (both built "
        "into libhdf5 core -- no plugin, no HDF5_PLUGIN_PATH, ever) to "
        "array fields that arrive uncompressed from the source; fields "
        "already compressed are left with their existing filter. Off by "
        "default. Reading the result back needs no special setup at all.",
    )
    parser.add_argument(
        "--exists-ok",
        choices=bool_choices,
        default="false",
        help="Allow writing into an already-existing destination "
        "directory/file. Off by default -- refuses to touch a destination "
        "that already exists, to avoid clobbering data from a typo.",
    )
    parser.add_argument(
        "--skip-if-sufficient",
        choices=bool_choices,
        default="false",
        help="Skip files that already have the expected fields and enough "
        "samples, instead of re-copying them. Makes re-running with a "
        "bigger --truncate-num only do the incremental work. This check "
        "runs independent of --exists-ok (skipping never touches the "
        "file); any file that turns out to need an actual rewrite still "
        "goes through the normal --exists-ok check as before. Off by "
        "default.",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="0=silent, 1=log after each subset finishes (default), "
        "2=also after each nu/scobj group within a subset, 3=also after "
        "every individual file. Failures always log regardless.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    a = parser.parse_args()

    bool_args = [
        "drop_lpf",
        "drop_polar",
        "drop_scobj_fields_in_meas",
        "drop_d_mh",
        "compress",
        "exists_ok",
        "skip_if_sufficient",
    ]
    for bool_arg in bool_args:
        setattr(a, bool_arg, getattr(a, bool_arg) == "true")

    return a


def parse_truncate_num(tokens: list) -> "int | dict | None":
    """Parses --truncate-num's tokens into what copy_and_shrink_dataset
    expects: a single int (uniform across subsets) or a dict keyed by
    subset name.
    """
    if tokens is None:
        return None
    if len(tokens) == 1 and "=" not in tokens[0]:
        return int(tokens[0])
    out = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(
                f"--truncate-num token {tok!r} must be 'subset=int' when "
                f"mixing with other tokens (got: {tokens})"
            )
        subset_name, num_str = tok.split("=", 1)
        out[subset_name] = int(num_str)
    return out


def main(args: argparse.Namespace) -> None:
    src_layout = RefDatasetLayout(
        base_dir=args.src_base_dir,
        nu_str_list=args.nu,
        subset_list=args.subsets,
    )
    dst_layout = RefDatasetLayout(
        base_dir=args.dst_base_dir,
        nu_str_list=args.nu,
        subset_list=args.subsets,
    )

    truncate_num = parse_truncate_num(args.truncate_num)

    logging.info(f"Shrinking dataset: {src_layout.base_dir} -> {dst_layout.base_dir}")
    logging.info(f"nu values: {args.nu}, subsets: {args.subsets}, truncate_num: {truncate_num}")

    results = copy_and_shrink_dataset(
        src_layout,
        dst_layout,
        truncate_num=truncate_num,
        n_workers=args.n_workers,
        drop_lpf=args.drop_lpf,
        drop_polar=args.drop_polar,
        drop_scobj_from_meas=args.drop_scobj_fields_in_meas,
        drop_d_mh=args.drop_d_mh,
        compress=args.compress,
        exists_ok=args.exists_ok,
        skip_if_sufficient=args.skip_if_sufficient,
        verbosity_level=args.verbosity_level,
    )

    n_failed = sum(1 for r in results if r.err is not None)
    n_skipped = sum(1 for r in results if not r.written)
    logging.info(
        f"Finished: {len(results)} files processed, {n_skipped} skipped "
        f"(already sufficient), {n_failed} failed"
    )
    if n_failed:
        raise SystemExit(f"{n_failed} file(s) failed to shrink; see log above for details")


if __name__ == "__main__":
    a = setup_args()
    logging.basicConfig(
        format=FMT, datefmt=TIMEFMT,
        level=logging.DEBUG if a.debug else logging.INFO,
    )
    main(a)
