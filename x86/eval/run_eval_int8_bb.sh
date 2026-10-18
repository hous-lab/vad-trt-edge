#!/bin/bash
# Evaluate INT8 backbone + FP16 head
# Usage: bash run_eval_int8_bb.sh

set -e
CONDA_BASE="${CONDA_BASE:-$HOME/apps/miniconda3}"
[ -x "$CONDA_BASE/bin/conda" ] || CONDA_BASE="$HOME/miniconda3"
eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
conda activate ${CONDA_ENV:-b2d_zoo}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export VAD_PROJECT_ROOT="${VAD_PROJECT_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)/Bench2DriveZoo_uniad_vad}"
export VAD_DEPLOY_ROOT="${VAD_DEPLOY_ROOT:-$REPO_ROOT}"
# Key: override the backbone engine with the INT8 one
export VAD_BACKBONE_ENGINE="${VAD_BACKBONE_ENGINE:-$VAD_DEPLOY_ROOT/x86/trt_compile/output/backbone_int8.engine}"
export VAD_HEAD_ENGINE="${VAD_HEAD_ENGINE:-$VAD_DEPLOY_ROOT/x86/trt_compile/output/head_withprev_fp16_fixed_v2.engine}"
export VAD_RESULT_TAG="${VAD_RESULT_TAG:-int8_bb_fp16_head}"

cd "$REPO_ROOT/x86/eval"
echo "==========================================="
echo "  Eval config: INT8 Backbone + FP16 Head"
echo "  Backbone: $VAD_BACKBONE_ENGINE"
echo "  Head:     $VAD_HEAD_ENGINE"
echo "==========================================="
nvidia-smi --query-gpu=memory.free --format=csv,noheader

python eval_pytorch_vs_trt.py 2>&1 | tee /tmp/eval_int8_bb.log
