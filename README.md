# Cache Me If You Can 


## Introduction
This project focuses on improving the inference efficiency of Qwen3-VL-4B-Instruct, a vision-language model used for multimodal question answering tasks. Vision-language models process both text and image inputs, which can create very long input sequences, especially when high-resolution images are converted into many visual tokens.
During decoding, every newly generated token attends to the existing key-value cache, also called the KV cache. As the sequence length increases, the KV cache becomes larger, leading to higher memory usage, more memory traffic, and increased latency. The main research question of this project is:
Can we reduce the visual token or KV cache cost to make inference faster and more memory-efficient without significantly reducing model accuracy?

To study this, the project evaluates multiple training-free compression techniques on Qwen3-VL-4B-Instruct across visual question-answering benchmarks. The main approaches explored are:

### [KV Cache Compression] (https://github.com/anokhimehta/cache-me-if-you-can/blob/main/kv_cache_compression/README.md)
This reduces the size of the KV cache during decoding. Token key-value pairs are ranked using an importance score, such as expected attention, and low-importance tokens are evicted based on a compression ratio. This helps reduce memory usage and decoding overhead while preserving the most useful context for generation.

### [VisionZIP-style visual token pruning](https://github.com/anokhimehta/cache-me-if-you-can/blob/main/vision-zip/README.md)
This reduces the number of image tokens before they are passed into the model. Image embeddings are ranked using an importance score, such as L2 norm, and only the most important tokens are retained.

### [DuetVLM-style token pruning](https://github.com/anokhimehta/cache-me-if-you-can/blob/main/duet-vlm/README_DUET_VLM.md)
This performs token pruning in a more model-aware way. It combines vision-to-vision token compression with text-guided pruning, where text queries help identify the most relevant visual tokens.

### [Modality-Stratified KV Cache ](https://github.com/anokhimehta/cache-me-if-you-can/blob/main/stratified-eviction/README.md)
This improves upon uniform KV cache eviction by maintaining separate budgets for image and text tokens. The motivation is that visual tokens are often more redundant than text tokens, so treating both modalities equally may be inefficient.

### [DivPrune visual token pruning](https://github.com/anokhimehta/cache-me-if-you-can/blob/main/divprune_eval/README.md)
This approach selects a diverse subset of visual tokens using a greedy max-min diversity algorithm on normalized token embeddings, then removes redundant image tokens before LLM computation.

The experiments were run on an NVIDIA A100 40GB GPU.


## Project Milestones and Completion Status
| Area | Status |
|---|---|
| Baseline benchmarking | Completed |
| KV cache compression | Completed |
| Modality-aware eviction | Completed  |
| Image-token masking and dropping | Completed |
| KV-cache image-token pruning | Completed |

### Links for results and running commands are present in the experiment's respective README files


