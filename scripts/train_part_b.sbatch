#!/bin/bash
#SBATCH --job-name=rob599-part-b
#SBATCH --account=engin1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-part-b-%A_%a.out

# Submit from ~/ROB599-Lab1. This script never downloads or alters shared data.
# Default: one 5 cm model. --array=0-2%1 runs 2, 5 and 10 cm as separate jobs.
set -eo pipefail
module load mamba/py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ROB498_ENV:-/home/szlin/mamba_envs/rob498}"
set -u

cd "${ROB498_REPO:-/home/szlin/ROB599-Lab1}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ROB498_DATA="${ROB498_DATA:-/scratch/engin_root/engin1/szlin/ROB599-Lab1/data/scannet20_small}"
ROB498_SEED="${ROB498_SEED:-42}"
ROB498_RUN_ROOT="${ROB498_RUN_ROOT:-/scratch/engin_root/engin1/szlin/ROB599-Lab1/results/part_b_seed${ROB498_SEED}}"
ROB498_EPOCHS="${ROB498_EPOCHS:-50}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    voxel_sizes=(0.02 0.05 0.10)
    if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 2 )); then
        echo "Array task must be 0 (2 cm), 1 (5 cm) or 2 (10 cm)" >&2
        exit 1
    fi
    voxel_size="${voxel_sizes[$SLURM_ARRAY_TASK_ID]}"
else
    voxel_size="${ROB498_VOXEL_SIZE:-0.05}"
fi

# Fail promptly if the preparation command did not finish.
if [[ ! -f "$ROB498_DATA/preparation.json" ]]; then
    echo "Missing $ROB498_DATA/preparation.json; finish data preparation first" >&2
    exit 1
fi

echo "Job: $SLURM_JOB_ID | host: $(hostname) | voxel size: $voxel_size m"
echo "Data: $ROB498_DATA"
echo "Results: $ROB498_RUN_ROOT"
date
nvidia-smi

# Additional options, such as --resume or --task-classes, are forwarded unchanged.
python -u -m lab1.part_b \
    --data "$ROB498_DATA" \
    --out "$ROB498_RUN_ROOT" \
    --voxel-size "$voxel_size" \
    --epochs "$ROB498_EPOCHS" \
    --seed "$ROB498_SEED" \
    "$@"
