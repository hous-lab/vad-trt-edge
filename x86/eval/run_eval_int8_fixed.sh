#!/bin/bash
# Evaluate INT8 fixed build (backbone INT8 + head INT8 fixed)
set -e
CONDA_BASE="${CONDA_BASE:-$HOME/apps/miniconda3}"
[ -x "$CONDA_BASE/bin/conda" ] || CONDA_BASE="$HOME/miniconda3"
eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
conda activate ${CONDA_ENV:-b2d_zoo}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export VAD_PROJECT_ROOT="${VAD_PROJECT_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)/Bench2DriveZoo_uniad_vad}"
export VAD_DEPLOY_ROOT="${VAD_DEPLOY_ROOT:-$REPO_ROOT}"
export VAD_BACKBONE_ENGINE="${VAD_BACKBONE_ENGINE:-$VAD_DEPLOY_ROOT/x86/trt_compile/output/backbone_int8.engine}"
export VAD_HEAD_ENGINE="${VAD_HEAD_ENGINE:-$VAD_DEPLOY_ROOT/x86/trt_compile/output/head_int8_fixed.engine}"

cd "$REPO_ROOT/x86/eval"
echo "==========================================="
echo "  Eval config: INT8 Backbone + INT8 Head FIXED (real calibration + MatMul FP16)"
echo "  Backbone: $VAD_BACKBONE_ENGINE"
echo "  Head:     $VAD_HEAD_ENGINE"
echo "==========================================="
nvidia-smi --query-gpu=memory.free --format=csv,noheader

python eval_pytorch_vs_trt.py 2>&1 | tee /tmp/eval_int8_fixed.log
mkdir -p "$REPO_ROOT/x86/eval/results"
cp /tmp/trt_hybrid_results.json "$REPO_ROOT/x86/eval/results/eval_int8_fixed.json"
