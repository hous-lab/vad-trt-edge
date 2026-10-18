#!/bin/bash
# Backbone INT8 TRT engine build script
# Usage: bash run_build_backbone_int8.sh [int8_random|int8_real]

MODE=${1:-int8_random}

echo "==========================================="
echo "  Backbone INT8 TRT build"
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
BUILD_MODE=$MODE python build_backbone_int8.py

echo ""
echo "==========================================="
echo "  Build complete"
echo "==========================================="
ls -lh ../output/backbone_int8.engine 2>/dev/null
ls -lh ../output/backbone_fp16.engine 2>/dev/null
