#!/usr/bin/env bash
# Stage-1 production benchmark: throughput vs batch size.
#
# Sweeps batch sizes 1, 2, 4, 8 across two methods (h2o vs streaming, the
# cheap baseline) at two compression ratios (0.0 baseline + 0.5 working
# point) on two datasets (mmmu where H2O looked best, docvqa where it
# regressed). 32 cells total. Designed to test Finding 2 from the README:
# does H2O's instrumentation overhead get amortized across the batch?
#
# Usage:
#   bash scripts/run_batched_sweep.sh                    # default: 32 samples per cell
#   MAX_SAMPLES=64 bash scripts/run_batched_sweep.sh
#   BATCH_SIZES="1 2 4" bash scripts/run_batched_sweep.sh
#
# Env vars:
#   DATASETS      default: "mmmu docvqa"
#   METHODS       default: "h2o streaming"
#   RATIOS        default: "0.0 0.5"
#   BATCH_SIZES   default: "1 2 4 8"
#   MAX_SAMPLES   default: 32

set -euo pipefail

DATASETS="${DATASETS:-mmmu docvqa}"
METHODS="${METHODS:-h2o streaming}"
RATIOS="${RATIOS:-0.0 0.5}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"

echo "Batched sweep config"
echo "  datasets    : ${DATASETS}"
echo "  methods     : ${METHODS}"
echo "  ratios      : ${RATIOS}"
echo "  batch_sizes : ${BATCH_SIZES}"
echo "  max_samples : ${MAX_SAMPLES}"
echo

for dataset in ${DATASETS}; do
    for method in ${METHODS}; do
        for ratio in ${RATIOS}; do
            for bs in ${BATCH_SIZES}; do
                echo ">>> ${method} cr=${ratio} bs=${bs} on ${dataset}"
                python -m eval.eval_kv_batched \
                    --method "${method}" \
                    --dataset "${dataset}" \
                    --compression_ratio "${ratio}" \
                    --batch_size "${bs}" \
                    --max_samples "${MAX_SAMPLES}" \
                    || echo "    (failed, continuing)"
            done
        done
    done
done

echo
echo "Sweep complete. Results in results/kv_batched/"
echo "Aggregate with: python eval/aggregate_batched.py"
