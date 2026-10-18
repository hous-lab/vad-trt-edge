#!/bin/bash
# ============================================================================
# Orin INT8 evaluation driver
# Loads the INT8 engines and runs the Bench2Drive evaluation
# Usage: bash run_eval_int8_orin.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_ROOT="$(cd "$STEP_DIR/../.." && pwd)"

echo "=========================================="
echo "📊 Orin INT8 Evaluation"
echo "=========================================="
echo "PIPELINE_ROOT: $PIPELINE_ROOT"
echo ""

# --------------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------------
export VAD_DEPLOY_ROOT="$PIPELINE_ROOT"

# VAD_PROJECT_ROOT
if [ -z "$VAD_PROJECT_ROOT" ]; then
    export VAD_PROJECT_ROOT="$(cd "$PIPELINE_ROOT/.." && pwd)/Bench2DriveZoo_uniad_vad"
fi
echo "  VAD_PROJECT_ROOT: $VAD_PROJECT_ROOT"
echo "  VAD_DEPLOY_ROOT:  $VAD_DEPLOY_ROOT"

# Engine paths (INT8)
export VAD_BACKBONE_ENGINE="$PIPELINE_ROOT/arm64/trt_compile/output/backbone_int8_orin.engine"
export VAD_HEAD_ENGINE="$PIPELINE_ROOT/arm64/trt_compile/output/head_int8_orin.engine"

echo "  VAD_BACKBONE_ENGINE: $VAD_BACKBONE_ENGINE"
echo "  VAD_HEAD_ENGINE:     $VAD_HEAD_ENGINE"

# Check the engines exist
if [ ! -f "$VAD_BACKBONE_ENGINE" ]; then
    echo "❌ Backbone INT8 engine not found!"
    echo "   Run: bash $PIPELINE_ROOT/arm64/trt_compile/int8/run_build_int8_orin.sh"
    exit 1
fi
if [ ! -f "$VAD_HEAD_ENGINE" ]; then
    echo "❌ Head INT8 engine not found!"
    echo "   Run: bash $PIPELINE_ROOT/arm64/trt_compile/int8/run_build_int8_orin.sh"
    exit 1
fi
echo "  ✅ Both engines found"
echo ""

# CUDA / TRT / plugin environment
if [ -z "$CUDA_HOME" ]; then
    if [ -n "$ORIN_CUDA_HOME" ]; then
        export CUDA_HOME="$ORIN_CUDA_HOME"
    else
        for candidate in /usr/local/cuda /usr/local/cuda-*; do
            if [ -d "$candidate" ] && [ -f "$candidate/bin/nvcc" ]; then
                export CUDA_HOME="$candidate"
                break
            fi
        done
    fi
fi

PLUGIN_LIB="$PIPELINE_ROOT/arm64/plugins_build/build/libplugins_orin.so"

# TRT python wheel (e.g. tensorrt 10.4.0) must resolve libnvinfer from the same
# TensorRT tarball the engines were built with - otherwise deserialization fails
# with "engine plan file is not compatible" even for a locally built engine
# (JetPack system libnvinfer is 10.3.0.x while the wheel is 10.4.x).
# Auto-detect ORIN_TRT_ROOT if not exported.
if [ -z "${ORIN_TRT_ROOT:-}" ]; then
    ORIN_TRT_ROOT="$(ls -d /usr/src/TensorRT-10* 2>/dev/null | sort -V | tail -1)"
    if [ -z "$ORIN_TRT_ROOT" ]; then
        echo "ERROR: ORIN_TRT_ROOT not set and no /usr/src/TensorRT-10* found"
        exit 1
    fi
    export ORIN_TRT_ROOT
    echo "Auto-detected ORIN_TRT_ROOT: $ORIN_TRT_ROOT"
fi

export LD_LIBRARY_PATH="$(dirname "$PLUGIN_LIB"):${ORIN_TRT_ROOT:-}/lib:${CUDA_HOME:-}/lib64:${LD_LIBRARY_PATH:-}"
export PLUGIN_LIBPATH="$PLUGIN_LIB"

# --------------------------------------------------------------------------
# Run evaluation
# --------------------------------------------------------------------------
echo "📋 Running INT8 evaluation..."
cd "$SCRIPT_DIR"

if [ ! -f "$SCRIPT_DIR/eval_orin.py" ]; then
    echo "❌ eval_orin.py not found in $SCRIPT_DIR"
    exit 1
fi

# Prepare the results dir
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

python3 eval_orin.py 2>&1 | tee /tmp/eval_int8_orin.log

# Copy results
if [ -f /tmp/trt_hybrid_results.json ]; then
    cp /tmp/trt_hybrid_results.json "$RESULTS_DIR/eval_int8_orin.json"
    echo "  ✅ Results saved to: $RESULTS_DIR/eval_int8_orin.json"
elif [ -f "$PIPELINE_ROOT/arm64/trt_compile/output/eval_int8_orin.json" ]; then
    cp "$PIPELINE_ROOT/arm64/trt_compile/output/eval_int8_orin.json" "$RESULTS_DIR/eval_int8_orin.json"
    echo "  ✅ Results saved to: $RESULTS_DIR/eval_int8_orin.json"
fi

echo ""
echo "=========================================="
echo "✅ INT8 Orin Evaluation Complete!"
echo "=========================================="
echo "  Results: $RESULTS_DIR/eval_int8_orin.json"
echo "=========================================="
