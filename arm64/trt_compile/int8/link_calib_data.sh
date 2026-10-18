#!/bin/bash
# ============================================================================
# Symlink shared calibration data for the Orin INT8 build
# Source: $REPO_ROOT/data/calib_data/*.npz
# Target: $SCRIPT_DIR/calib_data/*.npz
# Does not overwrite existing symlinks
# Usage: bash link_calib_data.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Shared calibration data source dir
SHARED_CALIB_DIR="$PIPELINE_ROOT/data/calib_data"
# Local calib_data dir
LOCAL_CALIB_DIR="$STEP_DIR/calib_data"

echo "=========================================="
echo "🔗 Link Calibration Data"
echo "=========================================="
echo "  Source (shared): $SHARED_CALIB_DIR"
echo "  Target (local):  $LOCAL_CALIB_DIR"
echo ""

# Create the local calib_data dir
mkdir -p "$LOCAL_CALIB_DIR"

# Calibration files to link
CALIB_FILES=(
    "head_calib.npz"
    "backbone_calib.npz"
)

LINKED_COUNT=0
SKIP_COUNT=0

for calib_file in "${CALIB_FILES[@]}"; do
    src_file="$SHARED_CALIB_DIR/$calib_file"
    dst_link="$LOCAL_CALIB_DIR/$calib_file"

    # Check the source file exists
    if [ ! -f "$src_file" ]; then
        echo "  ⚠️  Source not found: $src_file"
        echo "     Run sync first: bash $PIPELINE_ROOT/scripts/sync_to_orin.sh"
        SKIP_COUNT=$((SKIP_COUNT+1))
        continue
    fi

    # Check if the target already exists (file or symlink)
    if [ -e "$dst_link" ] || [ -L "$dst_link" ]; then
        if [ -L "$dst_link" ]; then
            existing_target="$(readlink -f "$dst_link")"
            if [ "$existing_target" = "$(readlink -f "$src_file")" ]; then
                echo "  ⏭️  Already linked: $calib_file → $(readlink "$dst_link")"
                SKIP_COUNT=$((SKIP_COUNT+1))
                continue
            fi
        fi
        echo "  ⚠️  Already exists (not a matching symlink): $dst_link"
        echo "     Remove it manually if you want to re-link."
        SKIP_COUNT=$((SKIP_COUNT+1))
        continue
    fi

    # Create a relative symlink (portable across directories)
    ln -s "../../../data/calib_data/$calib_file" "$dst_link"
    echo "  ✅ Linked: $calib_file → ../../../data/calib_data/$calib_file"
    LINKED_COUNT=$((LINKED_COUNT+1))
done

echo ""
echo "=========================================="
echo "📊 Link Summary"
echo "=========================================="
echo "  ✅ Linked: $LINKED_COUNT"
echo "  ⏭️  Skipped: $SKIP_COUNT"
echo ""

# Verify links
echo "  Current calib_data contents:"
ls -la "$LOCAL_CALIB_DIR/" | grep -v "^total" | sed 's/^/    /'
echo "=========================================="
