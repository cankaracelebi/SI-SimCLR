#!/bin/bash
#SBATCH --job-name=simclr_train
#SBATCH --output=logs/simclr_%j.out
#SBATCH --error=logs/simclr_%j.err
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --qos=job_gratis

# 1. Setup Environment
mkdir -p logs

ENV_NAME="${ENV_NAME:-si-simclr}"

if [[ -n "${CONDA_EXE}" && -f "/storage/homefs/${USER}/miniconda3/bin/activate" ]]; then
    source "/storage/homefs/${USER}/miniconda3/bin/activate"
    conda activate "${ENV_NAME}"
else
    echo "Conda not available; using current Python environment."
fi

echo "Starting training on $(hostname)"

# Resolve repository root from submit directory to avoid Slurm spool paths.
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    BASE_DIR="${SLURM_SUBMIT_DIR}"
else
    BASE_DIR="$(pwd)"
fi

if [[ -d "${BASE_DIR}/SI-SimCLR" && -d "${BASE_DIR}/configs" ]]; then
    REPO_ROOT="${BASE_DIR}"
elif [[ -d "${BASE_DIR}/../SI-SimCLR" && -d "${BASE_DIR}/../configs" ]]; then
    REPO_ROOT="$(cd "${BASE_DIR}/.." && pwd)"
else
    echo "Could not locate repository root from ${BASE_DIR}"
    exit 1
fi

TRAIN_DIR="${REPO_ROOT}/SI-SimCLR"
CONFIG_ENV="${REPO_ROOT}/configs/env.yml"
CONFIG_EXP="${REPO_ROOT}/configs/debug/simclr_acs.yml"

if [[ ! -f "${CONFIG_ENV}" ]]; then
    echo "Missing env config: ${CONFIG_ENV}"
    exit 1
fi

if [[ ! -f "${CONFIG_EXP}" ]]; then
    echo "Missing experiment config: ${CONFIG_EXP}"
    exit 1
fi

# Navigate to training directory (fixes relative imports)
cd "${TRAIN_DIR}" || exit 1

# Make parent package dirs visible (data/, utils/, and SI-SimCLR internals)
export PYTHONPATH="${REPO_ROOT}:${TRAIN_DIR}"

start=$(date +%s)

# Run training
PYTHON_EXE="${PYTHON_EXE:-python}"
"${PYTHON_EXE}" simclr.py \
    --config_env "${CONFIG_ENV}" \
    --config_exp "${CONFIG_EXP}"

end=$(date +%s)
echo "Job finished."
elapsed=$((end - start))
echo "Total time: $((elapsed/3600)) hours and $((elapsed%3600/60)) minutes"
