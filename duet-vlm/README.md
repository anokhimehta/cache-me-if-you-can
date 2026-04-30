# DUET-VLM Evaluation Framework for Qwen3-VL

This repository contains an evaluation framework for testing **DUET-VLM-style visual token reduction** on **Qwen/Qwen3-VL-4B-Instruct** across multiple multimodal benchmarks. The main goal is to measure whether visual tokens can be pruned before or during the language model forward pass while preserving task accuracy and improving inference efficiency.

The code supports two visual-token reduction modes:

1. **VisionZip-style image token dropping** using `visionzip_keep_ratio`.
2. **Text-to-Vision (T2V) pruning** using attention from text tokens to visual tokens at selected decoder layers.

The evaluation scripts report accuracy, latency, prefill time, decode time, GPU memory, throughput, and image-token counts before and after pruning.

---

## Repository Structure

```text
.
├── eval_docvqa.py          # Evaluation script for DocVQA
├── eval_mathvista.py       # Evaluation script for MathVista
├── eval_mmmu.py            # Evaluation script for MMMU
├── eval_realworldqa.py     # Evaluation script for RealWorldQA
├── modeling_qwen3_vl.py    # Modified Qwen3-VL model with token dropping and T2V pruning
├── qwen_utils.py           # Shared model loading, generation, metric collection, answer parsing
├── run_duetvlm.sh          # Bash script for sweeping keep ratios
└── results/                # Output JSON files are saved here
```

---

## Project Overview

Large vision-language models convert each image into many visual tokens. These tokens increase prefill cost, memory usage, and end-to-end latency. This project experiments with reducing the number of visual tokens while keeping answer quality as close as possible to the baseline.

The experiments are designed around the following question:

> Can we remove redundant image tokens before or during inference and still maintain acceptable multimodal task accuracy?

The implementation focuses on **pre-KV-cache optimization**, meaning the visual token reduction happens before or during the initial forward pass rather than after the KV cache has already been built.

---

## Supported Benchmarks

| Script | Dataset | Task Type | Default Split |
|---|---|---|---|
| `eval_docvqa.py` | `lmms-lab/DocVQA` | Document visual question answering | `validation` |
| `eval_mathvista.py` | `AI4Math/MathVista` | Math and visual reasoning | `testmini` |
| `eval_mmmu.py` | `MMMU/MMMU` | Multidiscipline multimodal reasoning | `test` |
| `eval_realworldqa.py` | `xai-org/RealworldQA` | Real-world image question answering | `test` |

---

## Main Components

### 1. `qwen_utils.py`

This file contains the shared utilities used by all benchmark scripts.

Main responsibilities:

- Load the Qwen3-VL model and processor.
- Format image-text prompts using the processor chat template.
- Run generation with `model.generate()`.
- Collect performance metrics.
- Normalize and extract final answers.
- Save results to JSON.

The default model is:

```python
Qwen/Qwen3-VL-4B-Instruct
```

The model is loaded using the custom `Qwen3VLForConditionalGeneration` class from `modeling_qwen3_vl.py`, not directly from Hugging Face Transformers.

---

### 2. `modeling_qwen3_vl.py`

This is a modified Qwen3-VL model file. It contains the actual visual-token pruning logic.

The key modifications are:

#### VisionZip-style token dropping

When `model.config.visionzip_keep_ratio` is set, the model:

1. Computes image embeddings.
2. Scores visual tokens using L2 norm.
3. Keeps the top `K = int(N * keep_ratio)` tokens.
4. Drops the remaining visual tokens.
5. Updates corresponding image masks and DeepStack visual features.
6. Passes the shortened visual sequence into the language model.

This is a true token-dropping approach rather than simply masking token values.

#### Text-to-Vision pruning

When T2V pruning is enabled, the language model uses attention weights from text tokens to vision tokens at selected decoder layers. The implementation keeps the most attended visual tokens and removes the rest from the hidden-state sequence.

The current T2V configuration uses pruning at layers such as:

```python
{14: 0.5, 21: 0.5}
```

This means that at those layers, only a fraction of the visual tokens are retained based on text-to-vision attention importance.

---

### 3. Evaluation scripts

Each evaluation script follows the same general pipeline:

1. Load benchmark dataset.
2. Extract image, question, and gold answer.
3. Build a short task-specific prompt.
4. Run Qwen3-VL generation with optional pruning.
5. Normalize the prediction.
6. Compare prediction with the ground truth.
7. Record accuracy and efficiency metrics.
8. Save detailed results as a JSON file.

---

## Installation

Create and activate a Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch transformers datasets accelerate tqdm pillow
```

If running on an HPC cluster or Singularity container, make sure the environment has GPU-enabled PyTorch installed.

Optional, but recommended for Hugging Face downloads:

```bash
export HF_TOKEN=<your_huggingface_token>
```

---

## Running Experiments

### RealWorldQA

Baseline without token dropping:

```bash
python3 eval_realworldqa.py \
  --split test \
  --limit 500 \
  --max_new_tokens 16
```

VisionZip-style token dropping:

```bash
python3 eval_realworldqa.py \
  --split test \
  --limit 500 \
  --visionzip_keep_ratio 0.5 \
  --max_new_tokens 16
```

VisionZip + T2V pruning:

```bash
python3 eval_realworldqa.py \
  --split test \
  --limit 500 \
  --visionzip_keep_ratio 0.5 \
  --t2v \
  --t2v_layer1_ratio 0.5 \
  --t2v_layer2_ratio 0.25 \
  --max_new_tokens 16
```

---

### DocVQA

```bash
python3 eval_docvqa.py \
  --split validation \
  --limit 500 \
  --visionzip_keep_ratio 0.5 \
  --t2v \
  --max_new_tokens 4
```

For DocVQA, the script includes an optional image resizing line:

```python
# image = image.resize((512, 512))
```

This can be enabled to reduce very large document images before evaluation.

---

### MathVista

```bash
python3 eval_mathvista.py \
  --split testmini \
  --limit 500 \
  --visionzip_keep_ratio 0.5 \
  --t2v \
  --max_new_tokens 4
```

The MathVista script skips samples with more than 2000 image tokens to avoid extremely expensive cases:

```python
if num_tokens > 2000:
    print(f"Skipping sample {i} (tokens={num_tokens})")
    continue
```

---

### MMMU

Run one subject:

```bash
python3 eval_mmmu.py \
  --split validation \
  --limit 50 \
  --visionzip_keep_ratio 0.5 \
  --t2v \
  --subjects Accounting
```

Run multiple subjects:

```bash
python3 eval_mmmu.py \
  --split validation \
  --limit 50 \
  --visionzip_keep_ratio 0.5 \
  --t2v \
  --subjects Accounting Biology Computer_Science
```

Run all subjects:

```bash
python3 eval_mmmu.py \
  --split validation \
  --limit 50 \
  --visionzip_keep_ratio 0.5 \
  --t2v \
  --subjects all
```

The MMMU script also saves a `per_subject_summary` section in the output JSON.

---

## Sweeping Keep Ratios

Use `run_duetvlm.sh` to run experiments across multiple keep ratios:

```bash
bash run_duetvlm.sh
```

The current sweep is:

```bash
ratios=(0 0.25 0.5 0.75 0.8 1)
```

Only one benchmark block should be active at a time. The current script runs DocVQA by default, while the RealWorldQA, MMMU, and MathVista commands are commented out.

To switch datasets, comment out the current command block and uncomment the dataset you want to run.

---

## Important Arguments

| Argument | Description |
|---|---|
| `--split` | Dataset split to evaluate. |
| `--limit` | Number of samples to evaluate. Useful for quick tests. |
| `--output` | Directory where result JSON files are saved. Default: `results/`. |
| `--max_new_tokens` | Maximum number of generated tokens. |
| `--visionzip_keep_ratio` | Fraction of visual tokens to keep. Example: `0.5` keeps 50%. |
| `--t2v` | Enables text-to-vision pruning. |
| `--t2v_layer1_ratio` | First T2V keep ratio, used in RealWorldQA. |
| `--t2v_layer2_ratio` | Second T2V keep ratio, used in RealWorldQA. |
| `--subjects` | MMMU subjects to evaluate. Use `all` for all subjects. |

---

## Output Format

Each run saves a timestamped JSON file in the `results/` directory.



Each output file contains:

```json
{
  "config": {},
  "summary": {},
  "results": []
}
```

For MMMU, the output also contains:

```json
"per_subject_summary": {}
```

---

## Metrics

| Metric | Meaning |
|---|---|
| `accuracy` | Fraction of correctly answered samples. |
| `avg_latency_sec` | Average end-to-end generation time. |
| `avg_prefill_sec` | Average time for the initial forward/prefill pass. |
| `avg_decode_sec` | Decode/generation time. In this implementation, this is recorded as total generation latency. |
| `avg_peak_memory_gb` | Peak GPU memory allocated during generation. |
| `avg_throughput_tok_per_sec` | Generated tokens per second. |
| `avg_image_tokens_before` | Number of visual tokens before pruning. |
| `avg_image_tokens_after` | Number of visual tokens retained after pruning. |


## Results
Results are present in this [link](https://docs.google.com/spreadsheets/d/12PNi3hzRkLylr0DWR3TUmmn1R_9DeLRX2lvOyDYmHUY/edit?gid=2040556413#gid=2040556413) 

