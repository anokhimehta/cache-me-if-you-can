#!/bin/bash
#SBATCH --job-name=kv_sweep
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/kv_sweep_%j.out
#SBATCH --error=logs/kv_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=22navyakumar@gmail.com

mkdir -p logs
mkdir -p $HOME/qwen3-vl-efficiency/results/kv_compression

export HF_HOME=$SCRATCH/hf_cache

singularity exec --nv --overlay $SCRATCH/overlay.ext3:ro \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        export HF_HOME=$SCRATCH/hf_cache
        export PYTHONNOUSERSITE=1
        source /ext3/miniconda3/etc/profile.d/conda.sh
        conda activate qwen310
        cd \$HOME/qwen3-vl-efficiency
        export PYTHONPATH=\$PWD:\$PYTHONPATH
        echo '=== node ==='; hostname; nvidia-smi | head -10
        MAX_SAMPLES=\${MAX_SAMPLES:-50} bash scripts/run_kv_sweep.sh
    "
