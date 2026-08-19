#!/bin/bash
# Run this script to populate the logs/... directories since SLURM
# will fail if the log file directories do not already exist

# Run this from the repo root that these relative paths are meant to live under
# (i.e. wherever "repo-dir" points to in system_setup.yaml).

set -euo pipefail

dirs=(
  logs/gen_data_main
  logs/gen_data_ood
  logs/rl
  logs/mmg_pipeline/mmg_solver
  logs/mmg_pipeline/train_fynet
  logs/mmg_pipeline/eval_fynet
  logs/mmg_pipeline/train_mmgu
  logs/mmg_pipeline/eval_mmgu
  logs/mmg_pipeline/train_mref
  logs/mmg_pipeline/eval_mref
  logs/mmg_pipeline/train_e2e_mref
  logs/mmg_pipeline/eval_e2e_mref
)

for d in "${dirs[@]}"; do
  mkdir -p "$d"
done

echo "Created ${#dirs[@]} logs/ and jobs/ subdirectories."
