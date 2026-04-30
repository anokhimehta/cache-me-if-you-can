#!/bin/bash
# Slurm array job: sweep H2O compression ratios across all benchmarks.
# Array index 0-4 → CR in (0.1 0.2 0.3 0.5 0.7)
# Usage: sbatch scripts/run_h2o.sh
#SBATCH --job-name=h2o_sweep
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=h2o_%A_%a.out
#SBATCH --error=h2o_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=am16455@nyu.edu

export HF_HOME=/scratch/am16455/hf_cache

CRS=(0.1 0.2 0.3 0.5 0.7)
CR=${CRS[$SLURM_ARRAY_TASK_ID]}

singularity exec --nv --overlay /scratch/am16455/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        cd /scratch/am16455/qwen3-efficiency-optim
        PYTHONPATH=/scratch/am16455/qwen3-efficiency-optim
        export PYTHONPATH

        python3 benchmarks/evaluate_realworldqa.py --method h2o --compression_ratio $CR
        python3 benchmarks/evaluate_docvqa.py      --method h2o --compression_ratio $CR
        python3 benchmarks/evaluate_mathvista.py   --method h2o --compression_ratio $CR
        python3 benchmarks/evaluate_mmmu.py        --method h2o --compression_ratio $CR
    "
