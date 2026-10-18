#!/bin/bash
# ============================================================================
# Orin FP16 TRT engine build driver
# Builds Backbone FP16, then Head FP16
# Usage: bash run_build_fp16_orin.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=========================================="
echo "🏗️  Orin FP16 Engine Build (sm_87)"
echo "=========================================="
echo "PIPELINE_ROOT: $PIPELINE_ROOT"
echo "STEP_DIR:      $STEP_DIR"
echo ""

# --------------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------------
export PIPELINE_ROOT

# CUDA_HOME: prefer ORIN_CUDA_HOME, otherwise auto-detect
if [ -z "$CUDA_HOME" ]; then
    if [ -n "$ORIN_CUDA_HOME" ]; then
        export CUDA_HOME="$ORIN_CUDA_HOME"
    else
        NVCC_PATH="$(which nvcc 2>/dev/null || true)"
        if [ -n "$NVCC_PATH" ]; then
            export CUDA_HOME="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"
        else
            for candidate in /usr/local/cuda /usr/local/cuda-*; do
                if [ -d "$candidate" ] && [ -f "$candidate/bin/nvcc" ]; then
                    export CUDA_HOME="$candidate"
                    break
                fi
            done
        fi
    fi
fi

if [ -z "$CUDA_HOME" ]; then
    echo "❌ CUDA_HOME not set! Please export CUDA_HOME or ORIN_CUDA_HOME"
    exit 1
fi
echo "  CUDA_HOME:      $CUDA_HOME"

# ORIN_TRT_ROOT check
if [ -z "$ORIN_TRT_ROOT" ]; then
    echo "❌ ORIN_TRT_ROOT not set!"
    echo "   Run: bash $PIPELINE_ROOT/scripts/detect_trt_versions.sh"
    echo "   Then: export ORIN_TRT_ROOT=<path>"
    exit 1
fi
echo "  ORIN_TRT_ROOT:  $ORIN_TRT_ROOT"

# Plugin library path
PLUGIN_LIB="$PIPELINE_ROOT/arm64/plugins_build/build/libplugins_orin.so"
if [ ! -f "$PLUGIN_LIB" ]; then
    echo "❌ Plugin lib not found: $PLUGIN_LIB"
    echo "   Run: bash $PIPELINE_ROOT/arm64/plugins_build/build_orin.sh"
    exit 1
fi

export LD_LIBRARY_PATH="$(dirname "$PLUGIN_LIB"):$ORIN_TRT_ROOT/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PLUGIN_LIBPATH="$PLUGIN_LIB"
echo "  PLUGIN_LIB:     $PLUGIN_LIB"
echo "  LD_LIBRARY_PATH set"
echo ""

# ONNX model path check
ONNX_DIR="$PIPELINE_ROOT/onnx_export/output"
mkdir -p "$PIPELINE_ROOT/arm64/trt_compile/output"

# --------------------------------------------------------------------------
# Step 1: Backbone FP16
# --------------------------------------------------------------------------
echo "📦 [1/2] Building Backbone FP16 Orin engine..."
cd "$SCRIPT_DIR"

if [ ! -f "$SCRIPT_DIR/build_backbone_fp16_orin.py" ]; then
    echo "❌ build_backbone_fp16_orin.py not found in $SCRIPT_DIR"
    exit 1
fi

python3 build_backbone_fp16_orin.py 2>&1 | tee /tmp/build_backbone_fp16_orin.log

BACKBONE_ENGINE="$PIPELINE_ROOT/arm64/trt_compile/output/backbone_fp16_orin.engine"
if [ ! -f "$BACKBONE_ENGINE" ]; then
    echo "❌ Backbone engine not generated: $BACKBONE_ENGINE"
    exit 1
fi
echo "  ✅ Backbone FP16 engine: $(ls -lh "$BACKBONE_ENGINE" | awk '{print $5}')"
echo ""

# --------------------------------------------------------------------------
# Step 2: Head FP16
# --------------------------------------------------------------------------
echo "📦 [2/2] Building Head FP16 Orin engine..."

if [ ! -f "$SCRIPT_DIR/build_head_fp16_orin.py" ]; then
    echo "❌ build_head_fp16_orin.py not found in $SCRIPT_DIR"
    exit 1
fi

python3 build_head_fp16_orin.py 2>&1 | tee /tmp/build_head_fp16_orin.log

HEAD_ENGINE="$PIPELINE_ROOT/arm64/trt_compile/output/head_fp16_orin.engine"
if [ ! -f "$HEAD_ENGINE" ]; then
    echo "❌ Head engine not generated: $HEAD_ENGINE"
    exit 1
fi
echo "  ✅ Head FP16 engine: $(ls -lh "$HEAD_ENGINE" | awk '{print $5}')"
echo ""

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo "=========================================="
echo "✅ FP16 Orin Build Complete!"
echo "=========================================="
echo "  Backbone: $BACKBONE_ENGINE"
echo "  Head:     $HEAD_ENGINE"
echo ""
echo "  Next: bash $PIPELINE_ROOT/arm64/eval/run_eval_fp16_orin.sh"
echo "=========================================="
