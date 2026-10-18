#!/bin/bash
# Head mixed-precision INT8 TRT engine build script
# Usage: bash run_build_head_int8_mixed.sh [int8_random|int8_real]

MODE=${1:-int8_random}

echo "==========================================="
echo "  Head mixed INT8 TRT build"
echo "  Mode: $MODE"
echo "==========================================="

CONDA_BASE="${CONDA_BASE:-$HOME/apps/miniconda3}"
[ -x "$CONDA_BASE/bin/conda" ] || CONDA_BASE="$HOME/miniconda3"
eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
conda activate ${CONDA_ENV:-b2d_zoo}

# GPU check
nvidia-smi | head -5

# Run build
cd "$(dirname "$0")"
BUILD_MODE=$MODE python build_head_int8_mixed.py

echo ""
echo "==========================================="
echo "  Build complete"
echo "==========================================="
ls -lh ../output/head_withprev_int8_mixed.engine 2>/dev/null
ls -lh ../output/head_withprev_fp16_fixed_v2.engine 2>/dev/null
