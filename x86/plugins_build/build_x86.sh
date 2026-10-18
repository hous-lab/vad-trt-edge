#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$DEPLOY_ROOT/plugins/build"

echo "=========================================="
echo "VAD TRT Plugin Build (x86_64)"
echo "=========================================="
echo "DEPLOY_ROOT: $DEPLOY_ROOT"
echo "BUILD_DIR: $BUILD_DIR"
echo ""

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ -z "${TRT_ROOT:-}" ]; then
    echo "ERROR: TRT_ROOT not set (TRT 10.x tarball, e.g. /usr/src/TensorRT-10.4.0.26)"
    exit 1
fi
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"

echo "[1/3] Configuring CMake..."
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DARCH=${X86_ARCH:-89} \
    -DTRT_ROOT="$TRT_ROOT" \
    -DCUDA_ROOT="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$TRT_ROOT/lib -L$CUDA_HOME/lib64" \
    "$DEPLOY_ROOT/plugins"

echo ""
echo "[2/3] Building plugins..."
make -j$(nproc)

echo ""
echo "[3/3] Verifying build result..."
if [ -f "$BUILD_DIR/libplugins.so" ]; then
    echo "✅ Built: $BUILD_DIR/libplugins.so"
    ls -lh "$BUILD_DIR/libplugins.so"
else
    echo "❌ Build failed: libplugins.so not generated"
    exit 1
fi

echo ""
echo "=========================================="
echo "Plugin build complete!"
echo "=========================================="
