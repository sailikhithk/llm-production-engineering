# Sources

All topics in this repo are public knowledge grounded in primary sources.
No content is adapted from any proprietary handbook. This file lists the
primary sources used per section, for verification and attribution.

## Section 01 - Cost Tracking

- OpenTelemetry GenAI semantic conventions (OTel spec)
- "How to Track Token Usage, Prompt Costs, and Model Latency with
  OpenTelemetry" - OneUptime blog (2026)
- "LLM Observability & Monitoring" - CalibreOS
- "LLM API Observability: Metrics, Traces, Logs, and Cost" - flatkey.ai
- "What You Cannot See Will Break Your LLM App" - DevOps.com
- Production experience: cost attribution across multi-model gateways

## Section 02 - Eval-Driven Deployment

- OpenInference semantic conventions (Arize AI)
- LangSmith / Phoenix eval session patterns
- Production experience: golden-set regression at Airbnb

## Section 03 - Capacity Planning

- vLLM production-stack release blog (vLLM, Jan 2025)
- AIBrix release blog (vLLM, Feb 2025)
- "Implementing High-Performance LLM Serving on GKE" - Google Cloud Blog
  (Jul 2025)
- Production experience: sizing for 100x traffic variance

## Section 04 - Observability

- "Observing vLLM with OpenTelemetry and Dash0" - Dash0 blog
- vLLM Prometheus metrics documentation
- "LLM API Observability" - flatkey.ai
- CalibreOS GenAI observability guide
- Production experience: TTFT/ITL SLO design

## Section 05 - Incident Playbooks

- vLLM preemption documentation
- vLLM production-stack fault tolerance documentation
- AIBrix GPU hardware failure detection documentation
- Production experience: on-call for LLM endpoints

## Section 06 - Decision Framework

- "AWQ vs GPTQ vs FP8 Compared" - packet.ai
- "The serving economics of quantization" - Tomoda Hinata
- "Systematic Characterization of LLM Quantization" - arXiv 2508.16712
- "Exploring the Trade-Offs: Quantization Methods, Task Difficulty, and
  Model Size" - IJCAI 2025
- Production experience: engine and quant selection across workloads

## Section 07 - Engine Tradeoffs

- vLLM documentation and blog (SOSP 2023 paper)
- SGLang documentation and paper
- TensorRT-LLM documentation (NVIDIA)
- NVIDIA Dynamo developer blog (2025)
- llama.cpp documentation
- Production experience: running multiple engines across workloads

## Foundational papers (referenced across sections)

- Kwon et al. "Efficient Memory Management for Large Language Model Serving
  with PagedAttention" - SOSP 2023
- Williams et al. "The Roofline Model" - CACM 2009
- Shazeer "Fast Transformer Decoding" - 2019
- Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models"
  - 2023
- Dao et al. "FlashAttention-2" - 2023
- Leviathan et al. "Fast Inference from Transformers via Speculative
  Decoding" - 2023
- Zhong et al. "DistServe: Disaggregating Prefill and Decoding for
  Goodput-optimized LLM Serving" - OSDI 2024

## KV cache compression papers (referenced in observability and incident
playbacks)

- KVzip - NeurIPS 2025
- RocketKV - arXiv 2502.14051 (2025)
- Compactor - arXiv 2507.08143 (2025)
- EVICPRESS - arXiv 2512.14946 (2025)
- "KVCache Cache in the Wild" - arXiv 2506.02634 (2025)

## Kubernetes and serving infrastructure

- "Introducing Gateway API Inference Extension" - Kubernetes blog (Jun 2025)
- vLLM production-stack documentation
- AIBrix documentation
