#!/bin/bash
#SBATCH --job-name=realworldqa_baseline
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=realworldqa_%j.out
#SBATCH --error=realworldqa_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=al9581@nyu.edu


set -euo pipefail


# -------------------------------
# Environment setup
# -------------------------------
export HF_HOME=/scratch/al9581/hf_cache
export TRANSFORMERS_CACHE=/scratch/al9581/hf_cache
export HF_DATASETS_CACHE=/scratch/al9581/hf_datasets
export PYTHONUNBUFFERED=1

mkdir -p /scratch/al9581/hf_cache
mkdir -p /scratch/al9581/hf_datasets
mkdir -p /scratch/al9581/project/results_manual

cd /scratch/al9581/project_work/vision-zip

# -------------------------------
# Run inside Singularity
# -------------------------------
bash /scratch/work/public/singularity/run-cuda-12.2.2.bash <<'EOF'
set -euo pipefail



export HF_HOME=/scratch/al9581/hf_cache
export TRANSFORMERS_CACHE=/scratch/al9581/hf_cache
export HF_DATASETS_CACHE=/scratch/al9581/hf_datasets
export PATH=$HOME/.local/bin:${PATH:-}
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1


JOB_ID=${SLURM_JOB_ID:-manual}
python3 eval_realworldqa.py --split test --limit 500 --visionzip_keep_ratio 0.8
EOF