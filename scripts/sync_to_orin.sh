#!/bin/bash
# ============================================================================
# rsync sync: x86_64 host (RTX 4090) -> Orin (ARM64, AGX Orin)
# Syncs everything needed to build engines and run evaluation on the Orin
# device. Never uses --delete.
#
# NOTE: ckpts/ (~668MB) is NOT synced by default (slow USB link); push it
# manually once if pytorch-side eval on Orin is needed.
#
# Usage: bash sync_to_orin.sh [ORIN_HOST] [ORIN_REMOTE_DIR]
#   ORIN_HOST        ssh alias/IP of the Orin (default: orin, or $ORIN_HOST)
#   ORIN_REMOTE_DIR  target dir on the Orin (default: vad_deploy_pipeline
#                    relative to the remote home, or $ORIN_REMOTE_DIR)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ORIN_HOST="${1:-${ORIN_HOST:-orin}}"
ORIN_REMOTE_DIR="${2:-${ORIN_REMOTE_DIR:-vad_deploy_pipeline}}"

REMOTE="${ORIN_HOST}:${ORIN_REMOTE_DIR}"

echo "=========================================="
echo "Sync to Orin (rsync)"
echo "=========================================="
echo "REPO_ROOT:       $REPO_ROOT"
echo "ORIN_HOST:       $ORIN_HOST"
echo "ORIN_REMOTE_DIR: $ORIN_REMOTE_DIR"
echo "Remote Target:   $REMOTE"
echo "=========================================="
echo ""

if ! command -v rsync &>/dev/null; then
    echo "ERROR: rsync not found! Please install: sudo apt install rsync"
    exit 1
fi

echo "Ensuring remote directory exists..."
ssh "$ORIN_HOST" "mkdir -p '$ORIN_REMOTE_DIR'"
echo "  Remote directory ready"
echo ""

SYNCED_COUNT=0
SKIPPED_COUNT=0

# Sync one item; never passes --delete
sync_item() {
    local desc="$1"
    local src="$2"
    local dst_subdir="$3"

    if [ ! -e "$src" ]; then
        echo "  Skip (not found): $desc -- $src"
        SKIPPED_COUNT=$((SKIPPED_COUNT+1))
        return 0
    fi

    echo "  Syncing: $desc"
    echo "     $src -> $REMOTE/$dst_subdir"
    # pre-create the remote parent dir (rsync's implied-dir creation can fail
    # when dst_subdir is nested, e.g. onnx_export/output/)
    ssh "$ORIN_HOST" "mkdir -p '$ORIN_REMOTE_DIR/$(dirname "$dst_subdir")'"
    rsync -avz --progress "$src" "$REMOTE/$dst_subdir"
    echo "  Done: $desc"
    echo ""
    SYNCED_COUNT=$((SYNCED_COUNT+1))
}

echo "[1/6] ONNX Models"
sync_item "ONNX models" \
    "$REPO_ROOT/onnx_export/output/" \
    "onnx_export/output/"

echo "[2/6] Calibration Data"
# data/calib_data holds the *.calib caches (small, required for engine
# builds) and the large *.npz sets (~37GB total) used to regenerate them.
sync_item "Calibration data (calib caches + npz)" \
    "$REPO_ROOT/data/calib_data/" \
    "data/calib_data/"

echo "[3/6] Bench2Drive Eval Data"
sync_item "Bench2Drive data (pkls + mini set)" \
    "$REPO_ROOT/data/" \
    "data/"

echo "[4/6] Plugin Source (shared with Orin build)"
sync_item "Plugin src/" \
    "$REPO_ROOT/plugins/src/" \
    "plugins/src/"
sync_item "Plugin CMakeLists.txt" \
    "$REPO_ROOT/plugins/CMakeLists.txt" \
    "plugins/"
sync_item "Plugin logger_def.cpp" \
    "$REPO_ROOT/plugins/logger_def.cpp" \
    "plugins/"

echo "[5/6] Orin-side Directories (arm64/)"
for arm_dir in arm64/plugins_build arm64/trt_compile arm64/eval; do
    sync_item "$arm_dir/" \
        "$REPO_ROOT/$arm_dir/" \
        "$arm_dir/"
done

echo "[6/6] Shared Scripts, Configs, Tools"
sync_item "scripts/" \
    "$REPO_ROOT/scripts/" \
    "scripts/"
sync_item "configs/" \
    "$REPO_ROOT/configs/" \
    "configs/"
sync_item "tools/" \
    "$REPO_ROOT/tools/" \
    "tools/"

echo "=========================================="
echo "Sync Summary"
echo "=========================================="
echo "  Synced items: $SYNCED_COUNT"
echo "  Skipped (not found): $SKIPPED_COUNT"
echo ""
echo "  Remote: $REMOTE"
echo ""
echo "  Next steps on Orin:"
echo "    cd $ORIN_REMOTE_DIR"
echo "    bash scripts/detect_trt_versions.sh"
echo "    export ORIN_TRT_ROOT=<detected_path>"
echo "    bash arm64/plugins_build/build_orin.sh"
echo "=========================================="
