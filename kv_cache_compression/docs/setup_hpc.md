# NYU HPC Setup — Qwen3-VL KV Cache Compression

End-to-end setup for running this repo on the NYU Greene HPC cluster. Follow
these in order from a fresh login. Tested on April 2026 with the
`cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif` Singularity image and
both L4 (interactive dev) and A100 (sweeps).

---

## 0. Prerequisites

You should have:

- An NYU HPC account (`netid@greene.hpc.nyu.edu`)
- Membership in your course allocation (e.g. `ece_gy_9143-2026sp`)
- The repo cloned to `$HOME/qwen3-vl-efficiency`

```bash
ssh netid@greene.hpc.nyu.edu
cd $HOME
git clone <repo-url> qwen3-vl-efficiency
cd qwen3-vl-efficiency
```

---

## 1. Get an interactive GPU node

Do NOT run installs from a login node. Grab an L4 (or any GPU) for
interactive development:

```bash
srun --account=ece_gy_9143-2026sp \
     --partition=g2-standard-12 \
     --gres=gpu:1 --cpus-per-task=4 --mem=40GB --time=4:00:00 \
     --pty /bin/bash
```

Once you land on a compute node (e.g. `b-31-129`), confirm GPU:

```bash
nvidia-smi | head -10
```

You should see `NVIDIA L4` (or A100 if you used a different partition).

> **Important:** If you have an OOD (Open OnDemand) Jupyter job running, it
> will lock the Singularity overlay file and block your interactive shell
> from mounting it `:rw`. Cancel any OOD job first:
> `squeue -u $USER` then `scancel <JOBID>`.

---

## 2. Create the Singularity overlay (one time)

The overlay is a writable filesystem image where you'll install miniconda
and your Python env. Create it once in `$SCRATCH` (persists across sessions):

```bash
ls $SCRATCH/overlay.ext3 2>/dev/null && echo "overlay exists" || {
    cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz $SCRATCH/overlay.ext3.gz
    gunzip $SCRATCH/overlay.ext3.gz
}
ls -la $SCRATCH/overlay.ext3
```

You should see a ~15 GB `.ext3` file.

---

## 3. Enter the Singularity container (read-write)

```bash
singularity exec --nv \
    --overlay $SCRATCH/overlay.ext3:rw \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash
```

Your prompt becomes `Singularity>`.

> If you get `can't open ... overlay.ext3 for writing, currently in use by
> another process`, another process is holding it. Outside the container:
> `squeue -u $USER` and cancel any other jobs (especially OOD Jupyter), then
> retry.

---

## 4. Install miniconda inside the overlay (one time)

```bash
ls /ext3/miniconda3/bin/conda 2>/dev/null && echo "conda installed" || {
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh
    bash Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -b -p /ext3/miniconda3
    rm Miniconda3-py310_24.7.1-0-Linux-x86_64.sh
}
```

Activate base:

```bash
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate base
python --version    # may report 3.13 — that's fine, we don't use base
```

---

## 5. Create the project conda env (one time)

The base conda env ships with Python 3.13, but `torch==2.4.1` only has
wheels for Python ≤ 3.12. Create a Python 3.10 env from `conda-forge`
(no Terms-of-Service prompt, unlike the default channel):

```bash
conda create -y -n qwen310 -c conda-forge --override-channels python=3.10
conda activate qwen310
python --version    # Python 3.10.x
which pip           # /ext3/miniconda3/envs/qwen310/bin/pip
```

---

## 6. Block stale user-site packages from leaking in

If you ever ran `pip install --user ...` outside the container, those packages
live in `~/.local/lib/python3.10/site-packages` and can shadow your conda env.
Block them:

```bash
export PYTHONNOUSERSITE=1
```

If you suspect leakage from a previous run, archive the directory:

```bash
[ -d /home/$USER/.local/lib/python3.10 ] && \
    mv /home/$USER/.local/lib/python3.10 /home/$USER/.local/lib/python3.10.bak
```

---

## 7. Install pinned requirements

The repo's `requirements.txt` already pins compatible versions. The full
working set:

```
torch==2.4.1
torchvision==0.19.1
transformers>=4.57,<4.60
accelerate>=1.0.1
datasets>=2.21,<4
huggingface-hub>=0.30
tokenizers
pillow
tqdm
numpy>=2.0,<3.0
pandas
matplotlib
kvpress==0.5.3
```

Why these pins:

- `torch==2.4.1` + `torchvision==0.19.1` — known-good combo on CUDA 12.1.
- `transformers>=4.57` — needed for `Qwen3VLForConditionalGeneration`.
- `kvpress==0.5.3` — supports modern transformers and exposes
  `ExpectedAttentionPress`, `SnapKVPress`, `PyramidKVPress`,
  `StreamingLLMPress`, and the `ScorerPress` base.
- `numpy>=2.0` — kvpress 0.5.x requires numpy 2.x.
- `datasets<4` — keep a stable dataset API.

Install:

```bash
cd $HOME/qwen3-vl-efficiency
pip install --no-cache-dir -r requirements.txt
```

If torch downloads slowly or wheels can't be found, install torch
explicitly first:

```bash
pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -r requirements.txt
```

---

## 8. Verify the env

```bash
python -c "
import torch, transformers, kvpress, datasets, numpy
import importlib.metadata as m
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('kvpress', m.version('kvpress'))
print('datasets', datasets.__version__)
print('numpy', numpy.__version__)
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
from transformers import Qwen3VLForConditionalGeneration
print('Qwen3VL class OK')
"
```

Expected output:

```
torch 2.4.1+cu121 cuda True
transformers 4.57.x
kvpress 0.5.3
datasets 2.21.x
numpy 2.x
gpu NVIDIA L4
Qwen3VL class OK
```

---

## 9. Set HuggingFace cache to scratch

The Qwen3-VL-4B weights are ~8 GB; don't download them into `$HOME` (small
quota). Point HF at `$SCRATCH`:

```bash
export HF_HOME=$SCRATCH/hf_cache
mkdir -p $HF_HOME
```

> Add this `export` to your sbatch script too (Step 12).

---

## 10. First model load (downloads weights)

```bash
cd $HOME/qwen3-vl-efficiency
export PYTHONPATH=$PWD:$PYTHONPATH
python -c "from src.load_model import load_model_and_processor; m, p = load_model_and_processor(); print('OK', m.config.model_type)"
```

First run downloads ~8 GB (5–10 min). Subsequent runs reuse the cache.
You should see `OK qwen3_vl` (or similar `model_type`).

---

## 11. Smoke tests

Run 5-sample evals to confirm the full pipeline works on the L4:

```bash
# H2O at 50% KV eviction
python -m eval.eval_kv_methods --method h2o \
    --dataset realworldqa --compression_ratio 0.5 --max_samples 5

# Modality-aware: aggressive on image, light on text
python -m eval.eval_kv_methods --method modality \
    --dataset realworldqa \
    --image_compression_ratio 0.7 --text_compression_ratio 0.2 \
    --inner h2o --max_samples 5
```

Both should finish without traceback and write JSONL results to
`results/kv_compression/`.

If either one errors with `unexpected keyword argument` or similar, the
kvpress 0.5.x API differs slightly from older versions — patch the
matching wrapper in `src/kv_compression/`.

---

## 12. Submit the full sweep on A100

Exit the container (`exit`) and return to the regular shell. From your
repo root:

```bash
cd $HOME/qwen3-vl-efficiency
mkdir -p logs

cat > scripts/sbatch_kv_sweep.sh <<'EOF'
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

singularity exec --nv \
    --overlay $SCRATCH/overlay.ext3:ro \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        export HF_HOME=$SCRATCH/hf_cache
        export PYTHONNOUSERSITE=1
        source /ext3/miniconda3/etc/profile.d/conda.sh
        conda activate qwen310
        cd \$HOME/qwen3-vl-efficiency
        export PYTHONPATH=\$PWD:\$PYTHONPATH
        MAX_SAMPLES=50 bash scripts/run_kv_sweep.sh
    "
EOF

chmod +x scripts/sbatch_kv_sweep.sh
sbatch scripts/sbatch_kv_sweep.sh
```

Note the overlay is mounted `:ro` (read-only) for batch jobs — they don't
install anything, just run. Read-only mounts also let multiple jobs share
the same overlay simultaneously.

Monitor:

```bash
squeue -u $USER
tail -f logs/kv_sweep_*.out
```

---

## 13. Returning to work later

After your first-time setup, returning to the project from a fresh login is
short:

```bash
# 1. login + grab a GPU node
srun --account=ece_gy_9143-2026sp \
     --partition=g2-standard-12 \
     --gres=gpu:1 --cpus-per-task=4 --mem=40GB --time=4:00:00 \
     --pty /bin/bash

# 2. enter the container
singularity exec --nv \
    --overlay $SCRATCH/overlay.ext3:rw \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash

# 3. activate env + set caches
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate qwen310
export HF_HOME=$SCRATCH/hf_cache
export PYTHONNOUSERSITE=1
cd $HOME/qwen3-vl-efficiency
export PYTHONPATH=$PWD:$PYTHONPATH

# 4. run experiments
python -m eval.eval_kv_methods --method h2o \
    --dataset realworldqa --compression_ratio 0.5 --max_samples 50
```

---

## Common pitfalls

**Overlay locked:** "can't open overlay.ext3 for writing, currently in use".
Another job (often OOD Jupyter) holds it. `squeue -u $USER` then `scancel
<JOBID>`. Wait 5 seconds, retry.

**`No matching distribution found for torch==2.4.1`:** You're in conda
`base` (Python 3.13). Activate `qwen310` (Python 3.10).

**`Terms of Service have not been accepted`:** Use `conda-forge` for env
creation: `conda create -y -n qwen310 -c conda-forge --override-channels
python=3.10`.

**`Cannot install ... conflicting dependencies`:** Re-derive the pins. The
chain is: `kvpress` constrains `transformers`, `transformers` constrains
`tokenizers` and `huggingface-hub`, `kvpress` also constrains `numpy` and
`datasets`. The `requirements.txt` in this repo is already the resolved
set — don't downgrade individual pins without re-checking.

**`Requirement already satisfied: ... in /home/.../.local/...`:** Stale
user-site packages leaking in. `export PYTHONNOUSERSITE=1` and (optionally)
`mv /home/$USER/.local/lib/python3.10 /home/$USER/.local/lib/python3.10.bak`.

**`ImportError: cannot import name 'Qwen3VLForConditionalGeneration'`:**
transformers too old. Bump to >=4.57 and use `kvpress==0.5.3` (which allows
that range).

**`module 'kvpress' has no attribute '__version__'`:** kvpress doesn't
expose `__version__`. Use `importlib.metadata.version('kvpress')`.

**OOM on L4 (23 GB) for 4B in bf16:** Shouldn't happen for short prompts,
but on long DocVQA contexts it might. Move that experiment to A100 via
sbatch.

**Slow HuggingFace download:** First model load is ~8 GB. Use
`HF_HOME=$SCRATCH/hf_cache` so it persists across sessions and you only pay
the download once.

---

## File map referenced

```
qwen3-vl-efficiency/
├── requirements.txt          # pinned env
├── src/
│   ├── load_model.py         # Qwen3-VL loader
│   └── kv_compression/       # H2O / Streaming / SnapKV / PyramidKV / Modality
├── eval/
│   └── eval_kv_methods.py    # --method dispatcher
├── scripts/
│   ├── run_kv_compression.sh # single cell
│   ├── run_kv_sweep.sh       # full grid (called by sbatch)
│   └── sbatch_kv_sweep.sh    # cluster job submission
└── docs/
    ├── kv_compression_plan.md  # method runbook
    └── setup_hpc.md            # this file
```
