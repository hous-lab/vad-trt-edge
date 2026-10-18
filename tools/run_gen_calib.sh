#!/bin/bash
# Generate real-data calibration npz (200 frames)
set -e
CONDA_BASE="${CONDA_BASE:-$HOME/apps/miniconda3}"
[ -x "$CONDA_BASE/bin/conda" ] || CONDA_BASE="$HOME/miniconda3"
eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
conda activate ${CONDA_ENV:-b2d_zoo}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export VAD_PROJECT_ROOT="${VAD_PROJECT_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)/Bench2DriveZoo_uniad_vad}"
export VAD_DEPLOY_ROOT="${VAD_DEPLOY_ROOT:-$REPO_ROOT}"

cd "$SCRIPT_DIR"
echo "==========================================="
echo "  Generate real calibration data (200 frames)"
echo "==========================================="
nvidia-smi --query-gpu=memory.free --format=csv,noheader

python gen_calib_data.py 2>&1 | tee /tmp/gen_calib.log
