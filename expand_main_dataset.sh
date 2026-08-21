#!/bin/bash
# Expand the main dataset (2026-08-17_main_dataset/) in place using
# scripts/do_expand_dataset.py, across all 10 frequencies.
#
# Submit to Slurm with:
#   sbatch expand_main_dataset.sh
# or run directly on a GPU node with:
#   bash expand_main_dataset.sh

#SBATCH --job-name=expand_main_dataset
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --output=logs/expand_dataset/expand_main_dataset.out
#SBATCH --error=logs/expand_dataset/expand_main_dataset.err
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE

echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
nvidia-smi -L

source /share/data/willett-group/oortsang/miniconda/etc/profile.d/conda.sh
conda activate hpscnn-env

target_main_dataset_dir="2026-08-17_main_dataset"

python scripts/do_expand_dataset.py \
    --base-dir ${target_main_dataset_dir} \
    --nu 1 2 3 4 5 6 7 8 9 10 \
    --subsets val test train \
    --backend jax

echo "`date` Finished Job"
