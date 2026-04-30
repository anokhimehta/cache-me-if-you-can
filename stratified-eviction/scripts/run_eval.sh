#!/bin/bash
# Run a single benchmark evaluation job.
# Usage:
#   sbatch scripts/run_eval.sh <benchmark> <method> <compression_ratio> [vision_weight]
#
# Examples:
#   sbatch scripts/run_eval.sh realworldqa h2o 0.5
#   sbatch scripts/run_eval.sh docvqa stratified_eviction 0.5 0.2
#
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

BENCHMARK=$1
METHOD=$2
CR=$3
VW=${4:-0.2}   # vision_weight only used if method is stratified_eviction

#SBATCH --job-name=${METHOD}_${BENCHMARK}_cr${CR}
#SBATCH --output=logs/${METHOD}_${BENCHMARK}_cr${CR}_vw${VW}_%j.out
#SBATCH --error=logs/${METHOD}_${BENCHMARK}_cr${CR}_vw${VW}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=am16455@nyu.edu

mkdir -p logs

echo "Benchmark      : $BENCHMARK"
echo "Method         : $METHOD"
echo "Compression    : $CR"
echo "Vision weight  : $VW"
echo "Job ID         : $SLURM_JOB_ID"
echo "Start          : $(date)"

export HF_HOME=/scratch/am16455/hf_cache

singularity exec --nv --overlay /scratch/am16455/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        cd /scratch/am16455/qwen3-efficiency-optim
        export PYTHONPATH=/scratch/am16455/qwen3-efficiency-optim

        python3 benchmarks/evaluate_${BENCHMARK}.py \
            --method $METHOD \
            --compression_ratio $CR \
            --vision_weight $VW
    "

echo "Done: $(date)"