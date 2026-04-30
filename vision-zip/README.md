# VisionZip Evaluation Framework for Qwen3-VL

This folder contains the VisionZip-style evaluation code for testing visual token reduction on `Qwen/Qwen3-VL-4B-Instruct`. The goal is to evaluate whether image-token reduction improves inference efficiency while preserving accuracy on multimodal benchmarks.

The implementation focuses on a **pre-KV-cache visual-token reduction** setup. Image tokens are scored before generation, reduced according to a keep ratio, and then passed through Qwen3-VL for answer generation and metric collection.

---

## Repository Structure

```text
visionzip/
├── eval_docvqa.py             # DocVQA evaluation script
├── eval_mathvista.py          # MathVista evaluation script
├── eval_mmmu.py               # MMMU evaluation script
├── eval_realworldqa.py        # RealWorldQA evaluation script
├── eval_realworldqa_dup.py    # Duplicate RealWorldQA script, kept for reference
├── modeling_qwen3_vl.py       # Modified Qwen3-VL model file with VisionZip logic
├── qwen_utils.py              # Shared loading, generation, metric, and answer utilities
├── run_realworldqa.slurm      # SLURM job script for RealWorldQA on NYU HPC
├── run_vision_zip.sh          # Bash sweep over keep ratios
└── results/                   # JSON outputs are saved here
```

---

## Project Objective

Vision-language models generate thousands of visual tokens from images. These tokens increase prefill cost, memory usage, and end-to-end latency. This project tests whether reducing image tokens can make inference cheaper while maintaining task performance.

The main research question is:

> Can we reduce redundant image tokens in Qwen3-VL and still preserve acceptable accuracy across RealWorldQA, DocVQA, MathVista, and MMMU?

---

## Supported Benchmarks

| Script | Dataset | Task Type | Default Split |
|---|---|---|---|
| `eval_realworldqa.py` | `xai-org/RealworldQA` | Real-world image QA | `test` |
| `eval_docvqa.py` | `lmms-lab/DocVQA` | Document visual QA | `validation` |
| `eval_mathvista.py` | `AI4Math/MathVista` | Visual math reasoning | `testmini` |
| `eval_mmmu.py` | `MMMU/MMMU` | Multidiscipline multimodal reasoning | `test` |

---

## Main Approach

The VisionZip implementation uses the `visionzip_keep_ratio` argument to control how many visual tokens are retained.



The code then computes an importance score for each visual token using the L2 norm and keeps the top-`K` tokens. In `qwen_utils.py`, the pixel tensor can be masked before generation. In the modified `modeling_qwen3_vl.py`, image embeddings are also selected using top-`K` L2-norm scoring and DeepStack image features are updated consistently.

Typical keep ratios used in experiments:

```text
0, 0.25, 0.5, 0.75, 0.8, 1
```

Interpretation:

| Keep Ratio | Meaning |
|---|---|
| `0` | Extreme compression; code keeps at least one token |
| `0.25` | Keep 25% of image tokens |
| `0.5` | Keep 50% of image tokens |
| `0.75` | Keep 75% of image tokens |
| `0.8` | Keep 80% of image tokens |
| `1` | Keep all image tokens; near-baseline setting |

---

## Setup

Create a Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch transformers datasets accelerate tqdm pillow
```

Optional Hugging Face authentication:

```bash
export HF_TOKEN=<your_huggingface_token>
```

For HPC/Singularity runs, make sure GPU-enabled PyTorch and Hugging Face libraries are available inside the container.

---

## Running VisionZip Experiments

### RealWorldQA

```bash
python3 eval_realworldqa.py \
  --split test \
  --limit 500 \
  --visionzip_keep_ratio 0.5 \
  --max_new_tokens 16
```

### DocVQA

```bash
python3 eval_docvqa.py \
  --split validation \
  --limit 100 \
  --visionzip_keep_ratio 0.5 \
  --max_new_tokens 4
```

### MathVista

```bash
python3 eval_mathvista.py \
  --split testmini \
  --limit 500 \
  --visionzip_keep_ratio 0.5 \
  --max_new_tokens 4
```

### MMMU

Single subject:

```bash
python3 eval_mmmu.py \
  --split validation \
  --limit 50 \
  --visionzip_keep_ratio 0.5 \
  --subjects Accounting
```

Multiple subjects:

```bash
python3 eval_mmmu.py \
  --split validation \
  --limit 50 \
  --visionzip_keep_ratio 0.5 \
  --subjects Accounting Biology Computer_Science
```

All subjects:

```bash
python3 eval_mmmu.py \
  --split validation \
  --limit 50 \
  --visionzip_keep_ratio 0.5 \
  --subjects all
```

---

## Running a Keep-Ratio Sweep

Use the provided script:

```bash
bash run_vision_zip.sh
```

The script sweeps over:

```bash
ratios=(0 0.25 0.5 0.75 0.8 1)
```

By default, it runs DocVQA. To run RealWorldQA, MathVista, or MMMU, uncomment the corresponding command block and comment out the current one.

---

## Running on SLURM

The included `run_realworldqa.slurm` script is configured for an NYU HPC-style GPU job. Submit it using:

```bash
sbatch run_realworldqa.slurm
```

The script sets Hugging Face cache paths, enters the project directory, starts the CUDA Singularity environment, and runs:

```bash
python3 eval_realworldqa.py --split test --limit 500 --visionzip_keep_ratio 0.8
```

Update the partition, account, project path, and email before running on a different cluster.

---

## Output Format

Each evaluation script saves a JSON file under `results/`. The output contains:

```json
{
  "config": {...},
  "summary": {...},
  "results": [...]
}
```

For MMMU, the output also includes:

```json
"per_subject_summary": {...}
```

---

## Metrics

| Metric | Meaning |
|---|---|
| `accuracy` | Fraction of correctly answered examples |
| `avg_latency_sec` | Average end-to-end generation latency |
| `avg_prefill_sec` | Average time for the model forward/prefill pass |
| `avg_decode_sec` | Average decode/generation time |
| `avg_peak_memory_gb` | Peak GPU memory allocated during inference |
| `avg_throughput_tok_per_sec` | Generated tokens per second |
| `avg_image_tokens_before` | Number of visual tokens before reduction |
| `avg_image_tokens_after` | Number of visual tokens after VisionZip reduction |

These metrics help compare the trade-off between task accuracy and inference efficiency.


## Results
The results are present in this [link](https://docs.google.com/spreadsheets/d/12PNi3hzRkLylr0DWR3TUmmn1R_9DeLRX2lvOyDYmHUY/edit?gid=2040556413#gid=2040556413 )




