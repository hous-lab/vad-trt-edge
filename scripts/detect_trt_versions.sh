#!/bin/bash
# ============================================================================
# Detect available TensorRT versions on the Orin device
# Searches libnvinfer.so, /opt/TensorRT-*, Python tensorrt, etc.
# Usage: bash detect_trt_versions.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "🔍 TensorRT Version Detection (Orin)"
echo "=========================================="
echo ""

# Collect all detected TRT versions
declare -a FOUND_VERSIONS
declare -a FOUND_PATHS
declare -a FOUND_TYPES

FOUND_COUNT=0

# --------------------------------------------------------------------------
# 1. Search libnvinfer.so (system-installed TRT)
# --------------------------------------------------------------------------
echo "🔎 [1/3] Searching system libnvinfer.so..."

LIB_SEARCH_PATHS=(
    "/usr/lib/aarch64-linux-gnu/libnvinfer.so*"
    "/usr/lib/aarch64-linux-gnu/libnvinfer.so"
)

# Also search TRT installs under /opt and /usr/local
for pattern in /opt/TensorRT-* /usr/local/TensorRT* /usr/src/TensorRT*; do
    if [ -d "$pattern" ]; then
        LIB_SEARCH_PATHS+=("$pattern/lib/libnvinfer.so*")
        LIB_SEARCH_PATHS+=("$pattern/lib/libnvinfer.so")
    fi
done

declare -A SEEN_LIBS

for search_pattern in "${LIB_SEARCH_PATHS[@]}"; do
    for lib_path in $search_pattern; do
        # Skip non-existent files and self-referential symlinks
        [ -f "$lib_path" ] || continue
        # Skip .so.X dev symlinks; check main or versioned libs only
        real_path="$(readlink -f "$lib_path")"
        [ -n "${SEEN_LIBS[$real_path]}" ] && continue
        SEEN_LIBS[$real_path]=1

        version=""
        # Extract the version via strings
        version=$(strings "$lib_path" 2>/dev/null | grep -oP 'TensorRT v?\K[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)

        if [ -z "$version" ]; then
            # Fallback: extract from filename
            version=$(echo "$lib_path" | grep -oP '\d+\.\d+\.\d+' | head -1 || true)
        fi

        if [ -n "$version" ]; then
            FOUND_VERSIONS+=("$version")
            FOUND_PATHS+=("$real_path")
            FOUND_TYPES+=("system-lib")
            ((FOUND_COUNT++))
            echo "  ✅ Found: TensorRT v$version"
            echo "     Path: $real_path"
        fi
    done
done

if [ "${#SEEN_LIBS[@]}" -eq 0 ]; then
    echo "  ⚠️  No libnvinfer.so found in system paths"
fi
echo ""

# --------------------------------------------------------------------------
# 2. Search TRT install directories under /opt and /usr/local
# --------------------------------------------------------------------------
echo "🔎 [2/3] Searching TRT installation directories..."

for trt_root in /opt/TensorRT-* /usr/local/TensorRT* /usr/src/TensorRT*; do
    [ -d "$trt_root" ] || continue
    [ -d "$trt_root/include" ] || continue

    # Extract version from directory name
    dir_version=$(basename "$trt_root" | grep -oP '\d+\.\d+[\.\d]*' | head -1 || true)

    # Also verify the version via libnvinfer
    lib_check="$trt_root/lib/libnvinfer.so"
    lib_version=""
    if [ -f "$lib_check" ]; then
        lib_version=$(strings "$lib_check" 2>/dev/null | grep -oP 'TensorRT v?\K[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
    fi

    use_version="${lib_version:-$dir_version}"

    if [ -n "$use_version" ]; then
        # Dedup check
        already_found=false
        for i in "${!FOUND_PATHS[@]}"; do
            if [ "${FOUND_PATHS[$i]}" = "$(readlink -f "$trt_root/lib/libnvinfer.so" 2>/dev/null || echo "")" ]; then
                already_found=true
                break
            fi
        done

        if [ "$already_found" = false ]; then
            FOUND_VERSIONS+=("$use_version")
            FOUND_PATHS+=("$trt_root")
            FOUND_TYPES+=("install-dir")
            ((FOUND_COUNT++))
            echo "  ✅ Found: TensorRT v$use_version"
            echo "     Root: $trt_root"
        fi
    fi
done
echo ""

# --------------------------------------------------------------------------
# 3. Check Python tensorrt (conda/venv)
# --------------------------------------------------------------------------
echo "🔎 [3/3] Searching Python tensorrt installations..."

# System python
PY_VERSION=$(python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || true)
if [ -n "$PY_VERSION" ]; then
    PY_PATH=$(python3 -c "import tensorrt; print(tensorrt.__file__)" 2>/dev/null || echo "unknown")
    FOUND_VERSIONS+=("$PY_VERSION")
    FOUND_PATHS+=("$PY_PATH")
    FOUND_TYPES+=("python-system")
    ((FOUND_COUNT++))
    echo "  ✅ Found: TensorRT v$PY_VERSION (system python3)"
    echo "     Module: $PY_PATH"
fi

# Conda environments
CONDA_BASE="${CONDA_BASE:-$HOME/apps/miniconda3}"
[ -x "$CONDA_BASE/bin/conda" ] || CONDA_BASE="$HOME/miniconda3"
if [ -d "$CONDA_BASE/envs" ]; then
    for env_dir in "$CONDA_BASE"/envs/*/; do
        env_name=$(basename "$env_dir")
        env_py="$env_dir/bin/python"
        if [ -x "$env_py" ]; then
            env_version=$("$env_py" -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || true)
            if [ -n "$env_version" ]; then
                FOUND_VERSIONS+=("$env_version")
                FOUND_PATHS+=("$env_dir")
                FOUND_TYPES+=("python-conda:$env_name")
                ((FOUND_COUNT++))
                echo "  ✅ Found: TensorRT v$env_version (conda: $env_name)"
                echo "     Env: $env_dir"
            fi
        fi
    done
fi

# Common venv locations
for venv_dir in ~/venv* /opt/venv*; do
    [ -d "$venv_dir/bin" ] || continue
    venv_py="$venv_dir/bin/python"
    if [ -x "$venv_py" ]; then
        venv_version=$("$venv_py" -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || true)
        if [ -n "$venv_version" ]; then
            FOUND_VERSIONS+=("$venv_version")
            FOUND_PATHS+=("$venv_dir")
            FOUND_TYPES+=("python-venv:$(basename "$venv_dir")")
            ((FOUND_COUNT++))
            echo "  ✅ Found: TensorRT v$venv_version (venv: $(basename "$venv_dir"))"
        fi
    fi
done
echo ""

# --------------------------------------------------------------------------
# Print a formatted table
# --------------------------------------------------------------------------
echo "=========================================="
echo "📊 Detected TensorRT Versions"
echo "=========================================="
printf "%-4s | %-15s | %-15s | %s\n" "#" "Version" "Type" "Path"
printf "%s\n" "---------------------------------------------------------------"

for i in "${!FOUND_VERSIONS[@]}"; do
    idx=$((i + 1))
    printf "%-4s | %-15s | %-15s | %s\n" "$idx" "${FOUND_VERSIONS[$i]}" "${FOUND_TYPES[$i]}" "${FOUND_PATHS[$i]}"
done

if [ "$FOUND_COUNT" -eq 0 ]; then
    echo "❌ No TensorRT installations found!"
    echo "   Please install TensorRT on this Orin device."
    exit 1
fi

echo ""

# --------------------------------------------------------------------------
# Recommend the newest version
# --------------------------------------------------------------------------
echo "=========================================="
echo "💡 Recommendation"
echo "=========================================="

# Newest install-dir entry
BEST_ROOT=""
BEST_VERSION="0.0.0"
for i in "${!FOUND_TYPES[@]}"; do
    if [[ "${FOUND_TYPES[$i]}" == "install-dir" ]]; then
        v="${FOUND_VERSIONS[$i]}"
        # Simple version compare
        if [[ "$(printf '%s\n%s\n' "$v" "$BEST_VERSION" | sort -V | tail -1)" == "$v" ]]; then
            BEST_VERSION="$v"
            BEST_ROOT="${FOUND_PATHS[$i]}"
        fi
    fi
done

# If no install-dir found, infer the root from a system-lib hit
if [ -z "$BEST_ROOT" ]; then
    for i in "${!FOUND_TYPES[@]}"; do
        if [[ "${FOUND_TYPES[$i]}" == "system-lib" ]]; then
            lib_path="${FOUND_PATHS[$i]}"
            # Try to infer the root: ../../ relative to the lib
            guessed_root="$(cd "$(dirname "$lib_path")/../.." && pwd 2>/dev/null || echo "")"
            if [ -n "$guessed_root" ] && [ -d "$guessed_root/include" ]; then
                BEST_ROOT="$guessed_root"
                BEST_VERSION="${FOUND_VERSIONS[$i]}"
            fi
            break
        fi
    done
fi

if [ -n "$BEST_ROOT" ]; then
    echo "  Recommended (newest install-dir):"
    echo "    export ORIN_TRT_ROOT=$BEST_ROOT"
    echo "    # Version: $BEST_VERSION"
else
    echo "  ⚠️  No dedicated TRT install-dir found."
    echo "  For system-installed TRT, set ORIN_TRT_ROOT to a directory containing include/ and lib/:"
    echo "    export ORIN_TRT_ROOT=/usr  # if headers in /usr/include, libs in /usr/lib/..."
fi
echo ""
echo "=========================================="
