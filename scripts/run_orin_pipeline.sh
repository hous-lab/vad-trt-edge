#!/bin/bash
# ============================================================================
# VAD Orin one-shot deployment pipeline
# Flow: TRT detection -> plugin build -> FP16 engines -> FP16 eval -> INT8 engines -> INT8 eval
# Usage: bash run_orin_pipeline.sh [--skip-fp16|--skip-int8|--auto]
#   --auto       run all steps without prompting
#   --skip-fp16  skip FP16 build and evaluation
#   --skip-int8  skip INT8 build and evaluation
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
AUTO_MODE=false
SKIP_FP16=false
SKIP_INT8=false

for arg in "$@"; do
    case "$arg" in
        --auto)       AUTO_MODE=true ;;
        --skip-fp16)  SKIP_FP16=true ;;
        --skip-int8)  SKIP_INT8=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "╔══════════════════════════════════════════╗"
echo "║   VAD Orin Deploy Pipeline (one-shot)    ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  PIPELINE_ROOT: $PIPELINE_ROOT"
echo "  Mode:          $([ "$AUTO_MODE" = true ] && echo 'Auto (no prompts)' || echo 'Interactive')"
echo "  FP16:          $([ "$SKIP_FP16" = true ] && echo '⏭️  Skip' || echo '✅ Run')"
echo "  INT8:          $([ "$SKIP_INT8" = true ] && echo '⏭️  Skip' || echo '✅ Run')"
echo ""

# Helper: wait for user confirmation
confirm_step() {
    local step_name="$1"
    if [ "$AUTO_MODE" = true ]; then
        echo "  🤖 Auto mode: proceeding with $step_name"
        return 0
    fi
    read -p "  ▶️  Proceed with $step_name? [Y/n] " response
    case "$response" in
        [nN]*) 
            echo "  ⏭️  Skipped: $step_name"
            return 1 
            ;;
        *)
            return 0
            ;;
    esac
}

# Track step results
declare -a STEP_STATUS

# ==========================================================================
# Step 1: Detect TensorRT versions
# ==========================================================================
echo "┌──────────────────────────────────────────┐"
echo "│  [Step 1/6] 🔍 Detect TensorRT Versions  │"
echo "└──────────────────────────────────────────┘"

if [ -z "$ORIN_TRT_ROOT" ]; then
    echo "  ORIN_TRT_ROOT not set, running detection..."
    bash "$SCRIPT_DIR/detect_trt_versions.sh"
    echo ""
    
    if [ "$AUTO_MODE" = false ]; then
        echo "  ⚠️  Please set ORIN_TRT_ROOT based on the detection above."
        read -p "  Enter ORIN_TRT_ROOT path (or press Enter to skip): " user_trt_root
        if [ -n "$user_trt_root" ]; then
            export ORIN_TRT_ROOT="$user_trt_root"
        fi
    fi
    
    if [ -z "$ORIN_TRT_ROOT" ]; then
        echo "  ❌ ORIN_TRT_ROOT is required. Aborting."
        STEP_STATUS+=("Step1:FAIL")
        exit 1
    fi
else
    echo "  ✅ ORIN_TRT_ROOT already set: $ORIN_TRT_ROOT"
fi

echo "  ✅ Step 1 complete: ORIN_TRT_ROOT=$ORIN_TRT_ROOT"
STEP_STATUS+=("Step1:OK")
echo ""

# ==========================================================================
# Step 2: Build Orin plugins
# ==========================================================================
echo "┌──────────────────────────────────────────┐"
echo "│  [Step 2/6] 🔧 Build Orin Plugins        │"
echo "└──────────────────────────────────────────┘"

PLUGIN_SO="$PIPELINE_ROOT/arm64/plugins_build/build/libplugins_orin.so"
if [ -f "$PLUGIN_SO" ]; then
    echo "  ✅ Plugin already exists: $PLUGIN_SO"
    echo "  Size: $(ls -lh "$PLUGIN_SO" | awk '{print $5}')"
    STEP_STATUS+=("Step2:SKIP(exists)")
else
    if confirm_step "Build Orin Plugins"; then
        bash "$PIPELINE_ROOT/arm64/plugins_build/build_orin.sh"
        echo "  ✅ Step 2 complete: Plugin built"
        STEP_STATUS+=("Step2:OK")
    else
        echo "  ⚠️  Plugin build skipped. Subsequent steps may fail."
        STEP_STATUS+=("Step2:SKIP(user)")
    fi
fi
echo ""

# ==========================================================================
# Step 3: FP16 engine build + eval
# ==========================================================================
if [ "$SKIP_FP16" = false ]; then
    echo "┌──────────────────────────────────────────┐"
    echo "│  [Step 3/6] 🏗️  Build FP16 Engines       │"
    echo "└──────────────────────────────────────────┘"

    FP16_BACKBONE="$PIPELINE_ROOT/arm64/trt_compile/output/backbone_fp16_orin.engine"
    FP16_HEAD="$PIPELINE_ROOT/arm64/trt_compile/output/head_fp16_orin.engine"

    if [ -f "$FP16_BACKBONE" ] && [ -f "$FP16_HEAD" ]; then
        echo "  ✅ FP16 engines already exist"
        STEP_STATUS+=("Step3:SKIP(exists)")
    else
        if confirm_step "Build FP16 Engines"; then
            bash "$PIPELINE_ROOT/arm64/trt_compile/fp16/run_build_fp16_orin.sh"
            echo "  ✅ Step 3 complete: FP16 engines built"
            STEP_STATUS+=("Step3:OK")
        else
            STEP_STATUS+=("Step3:SKIP(user)")
        fi
    fi
    echo ""

    # ==========================================================================
    # Step 4: FP16 evaluation
    # ==========================================================================
    echo "┌──────────────────────────────────────────┐"
    echo "│  [Step 4/6] 📊 Evaluate FP16             │"
    echo "└──────────────────────────────────────────┘"

    if [ -f "$FP16_BACKBONE" ] && [ -f "$FP16_HEAD" ]; then
        if confirm_step "Evaluate FP16"; then
            bash "$PIPELINE_ROOT/arm64/eval/run_eval_fp16_orin.sh"
            echo "  ✅ Step 4 complete: FP16 evaluation done"
            STEP_STATUS+=("Step4:OK")
        else
            STEP_STATUS+=("Step4:SKIP(user)")
        fi
    else
        echo "  ⏭️  Skipped: FP16 engines not available"
        STEP_STATUS+=("Step4:SKIP(no-engine)")
    fi
    echo ""
fi

# ==========================================================================
# Step 5: INT8 engine build
# ==========================================================================
if [ "$SKIP_INT8" = false ]; then
    echo "┌──────────────────────────────────────────┐"
    echo "│  [Step 5/6] 🏗️  Build INT8 Engines       │"
    echo "└──────────────────────────────────────────┘"

    INT8_BACKBONE="$PIPELINE_ROOT/arm64/trt_compile/output/backbone_int8_orin.engine"
    INT8_HEAD="$PIPELINE_ROOT/arm64/trt_compile/output/head_int8_orin.engine"

    if [ -f "$INT8_BACKBONE" ] && [ -f "$INT8_HEAD" ]; then
        echo "  ✅ INT8 engines already exist"
        STEP_STATUS+=("Step5:SKIP(exists)")
    else
        if confirm_step "Build INT8 Engines"; then
            bash "$PIPELINE_ROOT/arm64/trt_compile/int8/run_build_int8_orin.sh"
            echo "  ✅ Step 5 complete: INT8 engines built"
            STEP_STATUS+=("Step5:OK")
        else
            STEP_STATUS+=("Step5:SKIP(user)")
        fi
    fi
    echo ""

    # ==========================================================================
    # Step 6: INT8 evaluation
    # ==========================================================================
    echo "┌──────────────────────────────────────────┐"
    echo "│  [Step 6/6] 📊 Evaluate INT8             │"
    echo "└──────────────────────────────────────────┘"

    if [ -f "$INT8_BACKBONE" ] && [ -f "$INT8_HEAD" ]; then
        if confirm_step "Evaluate INT8"; then
            bash "$PIPELINE_ROOT/arm64/eval/run_eval_int8_orin.sh"
            echo "  ✅ Step 6 complete: INT8 evaluation done"
            STEP_STATUS+=("Step6:OK")
        else
            STEP_STATUS+=("Step6:SKIP(user)")
        fi
    else
        echo "  ⏭️  Skipped: INT8 engines not available"
        STEP_STATUS+=("Step6:SKIP(no-engine)")
    fi
    echo ""
fi

# ==========================================================================
# Final summary
# ==========================================================================
echo "╔══════════════════════════════════════════╗"
echo "║        🏁 Pipeline Summary               ║"
echo "╚══════════════════════════════════════════╝"
echo ""
for status in "${STEP_STATUS[@]}"; do
    step_name="${status%%:*}"
    step_result="${status##*:}"
    if [[ "$step_result" == "OK" ]]; then
        echo "  ✅ $step_name"
    elif [[ "$step_result" == SKIP* ]]; then
        echo "  ⏭️  $step_name ($step_result)"
    else
        echo "  ❌ $step_name ($step_result)"
    fi
done
echo ""

# Check result files
echo "📂 Results:"
for result_file in \
    "$PIPELINE_ROOT/arm64/eval/results/eval_fp16_orin.json" \
    "$PIPELINE_ROOT/arm64/eval/results/eval_int8_orin.json"; do
    if [ -f "$result_file" ]; then
        echo "  📄 $(basename "$result_file"): $(ls -lh "$result_file" | awk '{print $5}')"
    fi
done
echo ""

echo "╔══════════════════════════════════════════╗"
echo "║  ✅ Orin Pipeline Finished!              ║"
echo "╚══════════════════════════════════════════╝"
