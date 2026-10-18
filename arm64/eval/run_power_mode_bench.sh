#!/usr/bin/env bash
# Power-mode latency benchmark driver (runs ON the Orin).
#
# For each requested nvpmodel mode: switch mode, VERIFY the switch actually
# took effect (some modes need a reboot; a failed switch silently keeps the
# old clocks and produces meaningless numbers), run the benchmark, collect
# per-mode JSON + tegrastats logs. Restores MAXN at the end.
#
# Usage:  bash run_power_mode_bench.sh [mode_ids...]   (default: 0 3)
# Modes:  0=MAXN  1=MODE_15W  2=MODE_30W  3=MODE_50W
# SUFFIX: optional output-name suffix (e.g. SUFFIX=_base_int8 writes
#         power_mode_MAXN_base_int8.json) so different engine pairs do not
#         overwrite each other's result files.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

TRT_ROOT="${ORIN_TRT_ROOT:-/usr/src/TensorRT-10.4.0.26}"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:${LD_LIBRARY_PATH:-}"

BB_ENGINE="${BB_ENGINE:-$REPO_ROOT/arm64/trt_compile/output/backbone_fp16_orin.engine}"
HEAD_ENGINE="${HEAD_ENGINE:-$REPO_ROOT/arm64/trt_compile/output/head_fp16_orin.engine}"
PLUGIN_LIB="${PLUGIN_LIB:-$REPO_ROOT/arm64/plugins_build/build/libplugins_orin.so}"
CALIB_NPZ="${CALIB_NPZ:-$REPO_ROOT/data/calib_data/head_calib.npz}"
REPS="${REPS:-10}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/arm64/eval/results/power_modes}"
MODE_LABELS=([0]="MAXN" [1]="15w" [2]="30w" [3]="50w")

mode_name() {
    nvpmodel -q | head -1 | sed 's/NV Power Mode: //'
}

SUFFIX="${SUFFIX:-}"
for id in "${@:-0 3}"; do
    label="${MODE_LABELS[$id]:-mode$id}${SUFFIX}"
    echo "=== [$id] $label ==="

    sudo -n nvpmodel -m "$id" || { echo "nvpmodel -m $id failed"; exit 1; }
    sleep 2
    got="$(mode_name)"
    echo "nvpmodel reports: '$got'"
    case "$got" in
        MAXN)     want=0 ;;
        MODE_15W) want=1 ;;
        MODE_30W) want=2 ;;
        MODE_50W) want=3 ;;
        *) echo "FATAL: unknown mode name '$got' - aborting."; exit 1 ;;
    esac
    if [ "$want" != "$id" ]; then
        echo "FATAL: requested mode $id but device reports '$got' (id $want)."
        echo "This mode requires a reboot to take effect - aborting (nothing was benchmarked)."
        exit 2
    fi

    python3 "$HERE/power_mode_bench.py" \
        --backbone-engine "$BB_ENGINE" \
        --head-engine "$HEAD_ENGINE" \
        --plugin-lib "$PLUGIN_LIB" \
        --calib-npz "$CALIB_NPZ" \
        --reps "$REPS" \
        --mode-id "$id" \
        --mode-label "$label" \
        --output "$OUT_DIR/power_mode_${label}.json"
    status=$?
    echo "bench exit: $status"
    if [ $status -ne 0 ]; then
        echo "benchmark failed for mode $id - restoring MAXN and aborting."
        sudo -n nvpmodel -m 0; exit 3
    fi
done

# Restore MAXN so the device is left in its default state. If the current mode
# needs a reboot to leave, nvpmodel refuses the live switch (interactive prompt);
# report that instead of silently failing - never auto-reboot from a benchmark.
if ! restore_out="$(sudo -n nvpmodel -m 0 2>&1)"; then
    echo "NOTE: could not restore MAXN live -$(echo "$restore_out" | grep -o 'Reboot required.*' || echo ' nvpmodel refused')".
    echo "      Reboot the device (or run: echo yes | sudo nvpmodel -m 0) to return to MAXN."
else
    echo "=== done, restored MAXN ==="
fi
