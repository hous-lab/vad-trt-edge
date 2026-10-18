#!/bin/bash
# ============================================================================
# Orin (AGX Orin, ARM64, sm_87) TensorRT plugin build script
# Uses the shared plugin sources: plugins/src/ + CMakeLists.txt + logger_def.cpp
# Usage: bash build_orin.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_PLUGINS_DIR="$PIPELINE_ROOT/plugins"
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_SO="$BUILD_DIR/libplugins_orin.so"

echo "=========================================="
echo "🔧 Orin Plugin Build (aarch64, sm_87)"
echo "=========================================="
echo "SCRIPT_DIR:        $SCRIPT_DIR"
echo "PIPELINE_ROOT:     $PIPELINE_ROOT"
echo "SHARED_PLUGINS_DIR: $SHARED_PLUGINS_DIR"
echo ""

# --------------------------------------------------------------------------
# 1. Check the shared sources exist
# --------------------------------------------------------------------------
echo "📋 [1/5] Checking shared plugin source..."
if [ ! -f "$SHARED_PLUGINS_DIR/CMakeLists.txt" ]; then
    echo "❌ Shared CMakeLists.txt not found: $SHARED_PLUGINS_DIR/CMakeLists.txt"
    exit 1
fi
if [ ! -f "$SHARED_PLUGINS_DIR/logger_def.cpp" ]; then
    echo "❌ Shared logger_def.cpp not found: $SHARED_PLUGINS_DIR/logger_def.cpp"
    exit 1
fi
if [ ! -d "$SHARED_PLUGINS_DIR/src" ]; then
    echo "❌ Shared src/ not found: $SHARED_PLUGINS_DIR/src/"
    exit 1
fi
echo "  ✅ Shared source verified:"
echo "     - CMakeLists.txt: $SHARED_PLUGINS_DIR/CMakeLists.txt"
echo "     - logger_def.cpp: $SHARED_PLUGINS_DIR/logger_def.cpp"
echo "     - src/: $(ls "$SHARED_PLUGINS_DIR/src/")"
echo ""

# --------------------------------------------------------------------------
# 2. Auto-detect CUDA_HOME (JetPack)
# --------------------------------------------------------------------------
echo "📋 [2/5] Detecting CUDA_HOME..."

if [ -n "$CUDA_HOME" ]; then
    echo "  ℹ️  CUDA_HOME already set: $CUDA_HOME"
elif [ -n "$ORIN_CUDA_HOME" ]; then
    CUDA_HOME="$ORIN_CUDA_HOME"
    echo "  ℹ️  Using ORIN_CUDA_HOME: $CUDA_HOME"
else
    # Infer from the nvcc path
    NVCC_PATH="$(which nvcc 2>/dev/null || true)"
    if [ -n "$NVCC_PATH" ]; then
        CUDA_HOME="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"
        echo "  ℹ️  Detected CUDA_HOME from nvcc: $CUDA_HOME"
    else
        # Search /usr/local/cuda* (common JetPack locations)
        for candidate in /usr/local/cuda /usr/local/cuda-*; do
            if [ -d "$candidate" ] && [ -f "$candidate/bin/nvcc" ]; then
                CUDA_HOME="$candidate"
                break
            fi
        done
        if [ -n "$CUDA_HOME" ]; then
            echo "  ℹ️  Detected CUDA_HOME from /usr/local/cuda*: $CUDA_HOME"
        fi
    fi
fi

if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    echo "❌ CUDA_HOME not found! Please set it:"
    echo "   export CUDA_HOME=/usr/local/cuda"
    exit 1
fi
echo "  ✅ CUDA_HOME = $CUDA_HOME"
echo ""

# --------------------------------------------------------------------------
# 3. Auto-detect ORIN_TRT_ROOT (JetPack)
# --------------------------------------------------------------------------
echo "📋 [3/5] Detecting ORIN_TRT_ROOT..."

if [ -n "$ORIN_TRT_ROOT" ]; then
    echo "  ℹ️  ORIN_TRT_ROOT already set: $ORIN_TRT_ROOT"
else
    # Search common JetPack TensorRT locations
    TRT_SEARCH_PATHS=(
        /usr/src/TensorRT
        /opt/TensorRT
    )
    # Expand wildcard paths
    for pattern in /usr/src/TensorRT-* /opt/TensorRT-* /usr/local/TensorRT*; do
        if [ -d "$pattern" ]; then
            TRT_SEARCH_PATHS+=("$pattern")
        fi
    done

    for candidate in "${TRT_SEARCH_PATHS[@]}"; do
        if [ -d "$candidate/include" ] && [ -d "$candidate/lib" ]; then
            ORIN_TRT_ROOT="$candidate"
            break
        fi
    done

    if [ -n "$ORIN_TRT_ROOT" ]; then
        echo "  ℹ️  Detected ORIN_TRT_ROOT: $ORIN_TRT_ROOT"
    fi
fi

if [ -z "$ORIN_TRT_ROOT" ] || [ ! -d "$ORIN_TRT_ROOT" ]; then
    echo "❌ ORIN_TRT_ROOT not found! Please set it:"
    echo "   export ORIN_TRT_ROOT=/usr/src/TensorRT-10.x"
    echo "   Run scripts/detect_trt_versions.sh to find available versions."
    exit 1
fi

# Verify TRT headers and libs exist
if [ ! -f "$ORIN_TRT_ROOT/include/NvInfer.h" ]; then
    echo "❌ NvInfer.h not found in $ORIN_TRT_ROOT/include/"
    exit 1
fi
echo "  ✅ ORIN_TRT_ROOT = $ORIN_TRT_ROOT"
echo ""

# --------------------------------------------------------------------------
# 4. CMake configure + build
# --------------------------------------------------------------------------
echo "📋 [4/5] Building with CMake (TARGET=aarch64, ARCH=87)..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "  → cmake configure..."
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DTARGET=aarch64 \
    -DARCH=87 \
    -DTRT_ROOT="$ORIN_TRT_ROOT" \
    -DCUDA_ROOT="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$ORIN_TRT_ROOT/lib -L$CUDA_HOME/lib64" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$ORIN_TRT_ROOT/lib -L$CUDA_HOME/lib64" \
    "$SHARED_PLUGINS_DIR"

echo ""
echo "  → make -j$(nproc)..."
make -j"$(nproc)"

# CMakeLists.txt produces libplugins.so; rename to libplugins_orin.so to tell platforms apart
if [ -f "$BUILD_DIR/libplugins.so" ]; then
    cp "$BUILD_DIR/libplugins.so" "$OUTPUT_SO"
    echo "  ✅ Renamed libplugins.so → libplugins_orin.so"
else
    echo "❌ Build failed: libplugins.so not generated"
    exit 1
fi
echo ""

# --------------------------------------------------------------------------
# 5. Verify plugin symbols
# --------------------------------------------------------------------------
echo "📋 [5/5] Verifying plugin symbols..."
echo "  → Checking for plugin creator symbols in $OUTPUT_SO..."

SYMBOL_COUNT=$(nm -D "$OUTPUT_SO" 2>/dev/null | grep -c -i "plugin\|Plugin\|nvinfer1" || true)
if [ "$SYMBOL_COUNT" -gt 0 ]; then
    echo "  ✅ Found $SYMBOL_COUNT plugin-related symbols"
    echo "  → Sample symbols:"
    nm -D "$OUTPUT_SO" 2>/dev/null | grep -i "plugin\|Plugin\|nvinfer1" | head -10 | sed 's/^/    /'
else
    echo "  ⚠️  No plugin symbols found (may use RTTI/demangled names). Trying strings..."
    STRING_COUNT=$(strings "$OUTPUT_SO" | grep -c -i "plugin\|getPluginCreator" || true)
    if [ "$STRING_COUNT" -gt 0 ]; then
        echo "  ✅ Found $STRING_COUNT plugin-related strings"
        strings "$OUTPUT_SO" | grep -i "getPluginCreator\|PluginImpl\|registerPlugin" | head -10 | sed 's/^/    /'
    else
        echo "  ⚠️  Could not verify plugin symbols. Manual check recommended:"
        echo "      nm -D $OUTPUT_SO | grep -i plugin"
    fi
fi
echo ""

# --------------------------------------------------------------------------
# Final output
# --------------------------------------------------------------------------
echo "=========================================="
echo "✅ Orin Plugin Build Complete!"
echo "=========================================="
echo "Output: $OUTPUT_SO"
ls -lh "$OUTPUT_SO"
echo ""
echo "LD_LIBRARY_PATH setup:"
echo "   export LD_LIBRARY_PATH=$BUILD_DIR:\$LD_LIBRARY_PATH"
echo "=========================================="
