# Curated Production LLM Inference Curriculum & Primary Source Blueprint

This document compiles the foundational papers, architectural guides, interactive visualizers, and benchmarks for mastering production LLM serving from first principles.

---

## 1. Foundations of LLM Serving

Understanding the asymmetry between prompt evaluation and autoregressive token generation:

- **Attention Is All You Need** (Vaswani et al., 2017)  
  *The foundational paper introducing multi-head self-attention mechanisms.*  
  [Paper Link](https://arxiv.org/abs/1706.03762)

- **Let's Build GPT: from Scratch, in Code, Spelled Out** (Andrej Karpathy)  
  *The gold standard walk-through of tokenization, forward passes, and autoregressive generation loops.*  
  [Video & Code](https://github.com/karpathy/ng-video-lecture)

- **The Illustrated Transformer** (Jay Alammar)  
  *Visual breakdown of matrix transformations, embeddings, and attention projections.*  
  [Visual Guide](https://jalammar.github.io/illustrated-transformer/)

---

## 2. Transformer Computation & Attention Mechanics

- **Brendan Bycroft's 3D Interactive LLM Visualizer**  
  *An interactive 3D visualization showing every tensor transformation, QKV projection, and residual stream in real time.*  
  [Interactive Tool](https://bbycroft.net/llm)

- **Fast Transformer Decoding: One Write-Head is All You Need (MQA)** (Noam Shazeer, 2019)  
  *The seminal paper introducing Multi-Query Attention to eliminate memory bandwidth bottlenecks in autoregressive decoding.*  
  [Paper Link](https://arxiv.org/abs/1911.02150)

- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Ainslie et al., 2023)  
  *The architecture used by Llama-2/3 to balance memory efficiency and model capacity.*  
  [Paper Link](https://arxiv.org/abs/2305.13245)

- **DeepSeek-V2 / DeepSeek-V3 Multi-Head Latent Attention (MLA)**  
  *Compressing the KV cache by projecting key-value states into low-dimensional latent vectors.*  
  [Paper Link](https://arxiv.org/abs/2405.04434)

---

## 3. GPU Hardware & The Roofline Model

- **Making Deep Learning Go Brrr from First Principles** (Horace He)  
  *The definitive guide on GPU architecture, Streaming Multiprocessors, memory hierarchies, SRAM vs HBM, and kernel launch overheads.*  
  [Technical Guide](https://horace.io/brrr_intro.html)

- **Roofline: An Insightful Visual Performance Model for Multicore Architectures** (Williams et al., CACM 2009)  
  *Understanding when a kernel is bound by arithmetic intensity (FLOPS) vs memory bandwidth (GB/s).*  
  [CACM Paper](https://dl.acm.org/doi/10.1145/1498765.1498785)

- **NVIDIA CUDA Programming Guide & Hardware Architecture**  
  *Official documentation on warp scheduling, shared memory banks, and Tensor Core utilization.*  
  [NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

## 4. Optimization Techniques

- **PagedAttention: Efficient Memory Management for Large Language Model Serving with vLLM** (Kwon et al., SOSP 2023)  
  *Eliminating internal and external fragmentation in KV cache using virtual memory paging.*  
  [Paper Link](https://arxiv.org/abs/2309.06180)

- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (Tri Dao, 2023)  
  *IO-aware exact attention that computes attention without materializing the intermediate $N \times N$ matrix in HBM.*  
  [Paper Link](https://arxiv.org/abs/2307.08691)

- **Orca: A Distributed Serving System for Transformer-Based Generative Models** (OSDI 2022)  
  *The introduction of iteration-level continuous batching for high-throughput serving.*  
  [Paper Link](https://www.usenix.org/conference/osdi22/presentation/yu)

- **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2023)  
  *Using smaller draft models to generate candidate tokens verified in parallel by the target model.*  
  [Paper Link](https://arxiv.org/abs/2211.17192)

- **Aleksa Gordic's vLLM & PagedAttention Deep Dive**  
  *Engineering walk-through of block managers, physical memory allocations, and continuous batching schedulers.*  
  [Blog Post](https://aleksagordic.com/blog/vllm)

---

## 5. Production Inference Engines

- **vLLM Project**  
  *High-throughput serving engine with PagedAttention, chunked prefill, and multi-GPU tensor parallelism.*  
  [vLLM GitHub](https://github.com/vllm-project/vllm)

- **SGLang Project**  
  *High-performance engine featuring RadixAttention for automatic KV cache reuse across complex prompt templates, multi-turn chats, and agent loops.*  
  [SGLang GitHub](https://github.com/sgl-project/sglang)

- **TensorRT-LLM (NVIDIA)**  
  *NVIDIA's specialized runtime leveraging deep kernel fusion, FP8 GEMM, and In-Flight Batching.*  
  [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)

- **llama.cpp**  
  *Pure C/C++ inference optimized for Apple Silicon, CPU execution, and GGUF quantization formats.*  
  [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
