#!/bin/bash
# Expand the OOD dataset in place using scripts/do_expand_dataset.py
# (2026-08-17_ood_dataset/2025-09-23_ood_dataset_contrast_{}/)
# for the different contrasts, across all 10 frequencies
#
# Submit to Slurm with:
#   sbatch expand_ood_dataset.sh
# or run directly on a GPU node with:
#   bash expand_ood_dataset.sh

#SBATCH --job-name=expand_ood_dataset
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --output=logs/expand_dataset/expand_ood_dataset.out
#SBATCH --error=logs/expand_dataset/expand_ood_dataset.err
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE

echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
nvidia-smi -L

source /share/data/willett-group/oortsang/miniconda/etc/profile.d/conda.sh
conda activate hpscnn-env

base_ood_dataset_dir="2026-08-17_ood_dataset"
ood_contrast_prefix="2025-09-23_ood_dataset_contrast_"

# Glob for whichever contrast directories actually exist
shopt -s nullglob
ood_contrast_dirs=("${base_ood_dataset_dir}/${ood_contrast_prefix}"*)
shopt -u nullglob

if [ ${#ood_contrast_dirs[@]} -eq 0 ]; then
    echo "No contrast directories found matching ${base_ood_dataset_dir}/${ood_contrast_prefix}*"
    exit 1
fi

# To restrict to a specific subset of contrasts instead of expanding
# whatever's found on disk, replace the glob above with an explicit list, e.g.:
# ood_contrast_dirs=(
#     "${base_ood_dataset_dir}/${ood_contrast_prefix}1.0"
#     "${base_ood_dataset_dir}/${ood_contrast_prefix}2.0"
# )

for ood_contrast_dir in "${ood_contrast_dirs[@]}"; do
    echo "Expanding ${ood_contrast_dir}"
    python scripts/do_expand_dataset.py \
           --base-dir "${ood_contrast_dir}" \
           --nu 1 2 3 4 5 6 7 8 9 10 \
           --subsets test \
           --backend jax
done
echo "`date` Finished Job"
