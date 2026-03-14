#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${BIOENCODER_CONDA_ENV:-bioencoder_dev}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BACKEND="${BACKEND:-nccl}"
CONFIG_PATH="${1:-bioencoder_configs/train_stage1.yml}"

if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Typical local Miniconda install.
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  # Fallback to whatever conda is on PATH.
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "Could not find conda initialization script." >&2
  exit 1
fi

conda activate "$ENV_NAME"

# Workaround for local NCCL transport issues seen on this machine.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m bioencoder.scripts.train \
  --config-path "$CONFIG_PATH" \
  --distributed \
  --backend "$BACKEND" \
  "$@"
