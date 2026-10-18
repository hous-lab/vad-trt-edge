#!/bin/bash
# VAD-Base TensorRT deployment pipeline (x86) - one-shot end-to-end script
#
# Stages: 1 plugins -> 2 ONNX export -> 3 TRT engines -> 4 evaluation
# Usage: bash run_pipeline.sh [--from-stage N] [--to-stage N]
#
# Requires: TRT_ROOT and CUDA_HOME set; conda env `b2d_zoo` for stages 2-4.
#
# Expected eval gates (from earlier calibration runs):
#   FP16 fixed head:        TRT L2(1s) ~ 0.7275, FPS ~ 10.57
#   INT8 fixed (real calib): TRT L2(1s) ~ 0.7805

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

FROM_STAGE=1
TO_STAGE=4
while [ $# -gt 0 ]; do
    case "$1" in
        --from-stage) FROM_STAGE="$2"; shift 2 ;;
        --to-stage)   TO_STAGE="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; echo "Usage: bash run_pipeline.sh [--from-stage N] [--to-stage N]"; exit 1 ;;
    esac
done

echo "=========================================="
echo "VAD Deploy Root: $REPO_ROOT"
echo "Stages: $FROM_STAGE..$TO_STAGE"
echo "=========================================="

# Environment checks
if [ -z "$TRT_ROOT" ]; then
    echo "ERROR: TRT_ROOT not set! Please export TRT_ROOT=/path/to/TensorRT"
    exit 1
fi
if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: CUDA_HOME not set! Please export CUDA_HOME=/usr/local/cuda-11.8"
    exit 1
fi

# Conda env (stages 2-4 need python/torch/tensorrt)
CONDA_BASE="${CONDA_BASE:-$HOME/apps/miniconda3}"
[ -x "$CONDA_BASE/bin/conda" ] || CONDA_BASE="$HOME/miniconda3"
if [ -x "$CONDA_BASE/bin/conda" ]; then
    eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
    conda activate ${CONDA_ENV:-b2d_zoo}
fi

export LD_LIBRARY_PATH=$REPO_ROOT/plugins/build:$TRT_ROOT/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$REPO_ROOT/onnx_export/patches:$REPO_ROOT/onnx_export/scripts:$PYTHONPATH
export VAD_DEPLOY_ROOT="${VAD_DEPLOY_ROOT:-$REPO_ROOT}"
# Zoo repo (training code) auto-discovery: parent-of-parent, sibling, then $HOME.
# Set VAD_PROJECT_ROOT explicitly to override (docs/environment.md SS5).
if [ -z "${VAD_PROJECT_ROOT:-}" ]; then
    for _zoo_parent in "$(cd "$REPO_ROOT/../.." && pwd)" "$(cd "$REPO_ROOT/.." && pwd)" "$HOME"; do
        if [ -d "$_zoo_parent/Bench2DriveZoo_uniad_vad" ]; then
            export VAD_PROJECT_ROOT="$_zoo_parent/Bench2DriveZoo_uniad_vad"
            break
        fi
    done
    if [ -z "${VAD_PROJECT_ROOT:-}" ]; then
        echo "NOTE: Bench2DriveZoo_uniad_vad not found next to this repo or in \$HOME."
        echo "      Stages 2/4 need it: export VAD_PROJECT_ROOT=/path/to/Bench2DriveZoo_uniad_vad (docs/environment.md SS5)."
    fi
fi

# Stage 1: build plugins
if [ "$FROM_STAGE" -le 1 ] && [ "$TO_STAGE" -ge 1 ]; then
    echo ""
    echo "[Stage 1/4] Building TRT plugins..."
    if [ -f "$REPO_ROOT/plugins/build/libplugins.so" ]; then
        echo "  libplugins.so already exists, skipping"
    else
        bash "$REPO_ROOT/x86/plugins_build/build_x86.sh"
    fi
    ls -la "$REPO_ROOT/plugins/build/libplugins.so"
fi

# Stage 2: export ONNX
if [ "$FROM_STAGE" -le 2 ] && [ "$TO_STAGE" -ge 2 ]; then
    echo ""
    echo "[Stage 2/4] Exporting ONNX..."
    # demo_data is git-ignored; regenerate it from the checkpoint if missing
    if [ ! -d "$REPO_ROOT/onnx_export/demo_data/b2d_vadv1_prev.pts_bbox_head.forward" ]; then
        echo "  demo_data missing, generating via export_vad_prev.py ..."
        (cd "$REPO_ROOT/onnx_export" && python scripts/export_vad_prev.py)
    fi
    if [ -f "$REPO_ROOT/onnx_export/output/vad_head_withprev.onnx" ]; then
        echo "  vad_head_withprev.onnx already exists, skipping"
    else
        (cd "$REPO_ROOT/onnx_export" && python scripts/export_head_script.py)
    fi
    if [ -f "$REPO_ROOT/onnx_export/output/vad_backbone.onnx" ]; then
        echo "  vad_backbone.onnx already exists, skipping"
    else
        (cd "$REPO_ROOT/onnx_export" && python scripts/export_backbone_model.py)
    fi
fi

# Stage 3: compile TRT engines
if [ "$FROM_STAGE" -le 3 ] && [ "$TO_STAGE" -ge 3 ]; then
    echo ""
    echo "[Stage 3/4] Compiling TRT engines..."
    OUT="$REPO_ROOT/x86/trt_compile/output"
    mkdir -p "$OUT"
    if [ -f "$OUT/head_withprev_fp16_fixed_v2.engine" ]; then
        echo "  head_withprev_fp16_fixed_v2.engine already exists, skipping"
    else
        # BUILD_MODE=mixed -> head_withprev_fp16_fixed_v2.engine (the FP16 gate engine)
        (cd "$REPO_ROOT/x86/trt_compile/scripts" && BUILD_MODE=mixed python build_head_v2.py)
    fi
    if [ -f "$OUT/backbone_int8.engine" ]; then
        echo "  backbone_int8.engine already exists, skipping"
    else
        (cd "$REPO_ROOT/x86/trt_compile/scripts" && BUILD_MODE=int8_real python build_backbone_int8.py)
    fi
    if [ -f "$OUT/head_int8_fixed.engine" ]; then
        echo "  head_int8_fixed.engine already exists, skipping"
    else
        (cd "$REPO_ROOT/x86/trt_compile/scripts" && python build_head_int8.py)
    fi
fi

# Stage 4: evaluation
if [ "$FROM_STAGE" -le 4 ] && [ "$TO_STAGE" -ge 4 ]; then
    echo ""
    echo "[Stage 4/4] Running evaluation..."
    export VAD_BACKBONE_ENGINE="${VAD_BACKBONE_ENGINE:-$REPO_ROOT/x86/trt_compile/output/backbone_int8.engine}"
    export VAD_HEAD_ENGINE="${VAD_HEAD_ENGINE:-$REPO_ROOT/x86/trt_compile/output/head_int8_fixed.engine}"
    (cd "$REPO_ROOT/x86/eval" && python eval_pytorch_vs_trt.py)
fi

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "Expected: INT8 L2(1s) ~ 0.7805 reference-build / ~0.763 fresh rebuild (+/-0.02); FP16 0.727 (reference engines) - see docs/pipeline_guide.md SS5"
echo "=========================================="
