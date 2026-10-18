#!/bin/bash
set -e
CONDA_BASE="${CONDA_BASE:-$HOME/apps/miniconda3}"
[ -x "$CONDA_BASE/bin/conda" ] || CONDA_BASE="$HOME/miniconda3"
eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
conda activate ${CONDA_ENV:-b2d_zoo}

cd "$(dirname "$0")"
echo "==========================================="
echo "  Head INT8 fixed build (real calibration data)"
echo "==========================================="
nvidia-smi --query-gpu=memory.free --format=csv,noheader

python build_head_int8.py 2>&1 | tee /tmp/build_head_int8.log
