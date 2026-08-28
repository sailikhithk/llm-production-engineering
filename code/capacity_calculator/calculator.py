"""
Capacity Planning & KV Cache Sizing Engine for Production LLM Serving.
Author: Sai Likhith Kanuparthi
Repo: github.com/sailikhithk/llm-production-engineering
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import argparse


@dataclass(frozen=True)
class ModelArchitecture:
    name: str
    params_b: float
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    default_precision_bytes: int = 2  # 2 for FP16/BF16, 1 for FP8, 0.5 for 4-bit AWQ

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads < self.num_heads

    @property
    def gqa_ratio(self) -> float:
        return self.num_heads / self.num_kv_heads


@dataclass(frozen=True)
class GPUHardware:
    name: str
    vram_gb: float
    memory_bandwidth_tb_s: float
    fp16_tflops: float


# Pre-configured production models
MODEL_PRESETS: Dict[str, ModelArchitecture] = {
    "llama-3-8b": ModelArchitecture(
        name="Meta Llama-3-8B",
        params_b=8.03,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        default_precision_bytes=2,
    ),
    "llama-3-70b": ModelArchitecture(
        name="Meta Llama-3-70B",
        params_b=70.6,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        default_precision_bytes=2,
    ),
    "mistral-7b": ModelArchitecture(
        name="Mistral-7B-v0.3",
        params_b=7.24,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        default_precision_bytes=2,
    ),
    "qwen-2.5-7b": ModelArchitecture(
        name="Qwen2.5-7B",
        params_b=7.61,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
        head_dim=128,
        default_precision_bytes=2,
    ),
    "deepseek-v3-small": ModelArchitecture(
        name="DeepSeek-V3-Lite",
        params_b=16.0,
        num_layers=28,
        num_heads=16,
        num_kv_heads=16,
        head_dim=128,
        default_precision_bytes=1,  # FP8 Native
    ),
}

# Pre-configured enterprise GPUs
GPU_PRESETS: Dict[str, GPUHardware] = {
    "a100-80gb": GPUHardware(
        name="NVIDIA A100 SXM4 80GB HBM2e",
        vram_gb=80.0,
        memory_bandwidth_tb_s=2.039,
        fp16_tflops=312.0,
    ),
    "h100-80gb": GPUHardware(
        name="NVIDIA H100 SXM5 80GB HBM3",
        vram_gb=80.0,
        memory_bandwidth_tb_s=3.35,
        fp16_tflops=989.0,
    ),
    "l40s-48gb": GPUHardware(
        name="NVIDIA L40S 48GB GDDR6",
        vram_gb=48.0,
        memory_bandwidth_tb_s=0.864,
        fp16_tflops=366.0,
    ),
    "rtx-4090-24gb": GPUHardware(
        name="NVIDIA RTX 4090 24GB GDDR6X",
        vram_gb=24.0,
        memory_bandwidth_tb_s=1.008,
        fp16_tflops=165.0,
    ),
}


class CapacityCalculator:
    """
    Core engine calculating KV cache requirements, memory pools, concurrency,
    and cluster sizing for production LLM deployments.
    """

    @staticmethod
    def calculate_kv_bytes_per_token(
        model: ModelArchitecture, precision_bytes: Optional[int] = None
    ) -> int:
        """
        Formula: 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
        (Factor of 2 accounts for Key and Value matrices).
        """
        bytes_per_elem = precision_bytes or model.default_precision_bytes
        return 2 * model.num_layers * model.num_kv_heads * model.head_dim * bytes_per_elem

    @staticmethod
    def calculate_request_kv_bytes(
        model: ModelArchitecture,
        prompt_tokens: int,
        output_tokens: int,
        precision_bytes: Optional[int] = None,
        prefix_cache_hit_rate: float = 0.0,
    ) -> int:
        """
        Calculates maximum dedicated KV memory for a single request.
        Adjusts for prefix caching savings on prompt tokens.
        """
        kv_bytes_per_token = CapacityCalculator.calculate_kv_bytes_per_token(
            model, precision_bytes
        )
        effective_prompt_tokens = prompt_tokens * (1.0 - prefix_cache_hit_rate)
        total_tokens = effective_prompt_tokens + output_tokens
        return int(total_tokens * kv_bytes_per_token)

    @staticmethod
    def calculate_usable_kv_pool_gb(
        gpu: GPUHardware,
        model: ModelArchitecture,
        gpu_memory_utilization: float = 0.90,
        overhead_gb: float = 4.0,
        precision_bytes: Optional[int] = None,
        tensor_parallel_size: int = 1,
    ) -> float:
        """
        Calculates usable KV cache memory pool (GB) per GPU.
        Pool = (Total VRAM * Util) - (Model Weights / TP) - PyTorch Overhead.
        """
        bytes_per_param = precision_bytes or model.default_precision_bytes
        weights_gb = (model.params_b * bytes_per_param) / tensor_parallel_size
        max_allocated = gpu.vram_gb * gpu_memory_utilization
        usable_kv_gb = max_allocated - weights_gb - overhead_gb
        return max(0.0, usable_kv_gb)

    @staticmethod
    def calculate_max_concurrency(
        gpu: GPUHardware,
        model: ModelArchitecture,
        prompt_tokens: int = 1500,
        output_tokens: int = 500,
        gpu_memory_utilization: float = 0.90,
        overhead_gb: float = 4.0,
        precision_bytes: Optional[int] = None,
        prefix_cache_hit_rate: float = 0.0,
        tensor_parallel_size: int = 1,
    ) -> int:
        """
        Calculates hard concurrency threshold before vLLM/SGLang starts preemption.
        Concurrency = Usable KV Cache / Request Footprint.
        """
        usable_kv_bytes = (
            CapacityCalculator.calculate_usable_kv_pool_gb(
                gpu=gpu,
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
                overhead_gb=overhead_gb,
                precision_bytes=precision_bytes,
                tensor_parallel_size=tensor_parallel_size,
            )
            * (1024**3)
        )

        req_kv_bytes = CapacityCalculator.calculate_request_kv_bytes(
            model=model,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            precision_bytes=precision_bytes,
            prefix_cache_hit_rate=prefix_cache_hit_rate,
        )

        if req_kv_bytes == 0:
            return 0

        return int(usable_kv_bytes // req_kv_bytes)

    @staticmethod
    def calculate_sustainable_qps(
        max_concurrency: int, avg_latency_seconds: float = 4.2
    ) -> float:
        """
        Little's Law: Throughput (QPS) = Concurrency / Latency (sec)
        """
        if avg_latency_seconds <= 0:
            return 0.0
        return round(max_concurrency / avg_latency_seconds, 2)

    @staticmethod
    def calculate_cluster_sizing(
        target_peak_qps: float,
        qps_per_gpu: float,
        headroom_multiplier: float = 1.25,
        tensor_parallel_size: int = 1,
    ) -> Dict[str, float]:
        """
        Computes minimum GPUs and physical server nodes required to satisfy SLA.
        """
        if qps_per_gpu <= 0:
            return {"gpus_required": 0, "safety_qps": 0}

        raw_gpus = (target_peak_qps / qps_per_gpu) * headroom_multiplier
        # Must be multiple of tensor_parallel_size
        units = math.ceil(raw_gpus / tensor_parallel_size)
        gpus_required = units * tensor_parallel_size

        return {
            "gpus_required": gpus_required,
            "target_peak_qps": target_peak_qps,
            "cluster_max_qps": round(gpus_required * qps_per_gpu, 2),
            "headroom_percent": round((headroom_multiplier - 1.0) * 100, 1),
        }


def format_sizing_report(
    model_key: str,
    gpu_key: str,
    prompt_tokens: int,
    output_tokens: int,
    target_qps: float,
    avg_latency_s: float = 4.2,
    prefix_cache_hit_rate: float = 0.0,
    tensor_parallel_size: int = 1,
) -> str:
    model = MODEL_PRESETS[model_key]
    gpu = GPU_PRESETS[gpu_key]

    kv_per_token_bytes = CapacityCalculator.calculate_kv_bytes_per_token(model)
    req_kv_bytes = CapacityCalculator.calculate_request_kv_bytes(
        model, prompt_tokens, output_tokens, prefix_cache_hit_rate=prefix_cache_hit_rate
    )
    usable_kv_gb = CapacityCalculator.calculate_usable_kv_pool_gb(
        gpu, model, tensor_parallel_size=tensor_parallel_size
    )
    max_concurrency = CapacityCalculator.calculate_max_concurrency(
        gpu,
        model,
        prompt_tokens,
        output_tokens,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
        tensor_parallel_size=tensor_parallel_size,
    )
    qps_per_gpu = CapacityCalculator.calculate_sustainable_qps(
        max_concurrency, avg_latency_s
    )
    sizing = CapacityCalculator.calculate_cluster_sizing(
        target_qps, qps_per_gpu, tensor_parallel_size=tensor_parallel_size
    )

    return f"""
================================================================================
🚀 LLM SERVING CAPACITY & SIZING REPORT
================================================================================
Model:                {model.name} ({model.params_b}B Params | GQA Ratio: {model.gqa_ratio:.1f}x)
GPU Hardware:         {gpu.name} ({gpu.vram_gb} GB VRAM)
Tensor Parallel:      TP={tensor_parallel_size}

1. MEMORY FOOTPRINT & KV CACHE
--------------------------------------------------------------------------------
KV Cache per Token:   {kv_per_token_bytes:,} Bytes (~{kv_per_token_bytes / 1024:.1f} KB/token)
Sequence Length:      {prompt_tokens} Prompt + {output_tokens} Output = {prompt_tokens + output_tokens} Tokens
Prefix Cache Hit:     {prefix_cache_hit_rate * 100:.1f}%
Request KV Footprint: {req_kv_bytes / (1024**2):.2f} MB / Concurrent Request
Weights Memory:       {(model.params_b * model.default_precision_bytes) / tensor_parallel_size:.1f} GB per GPU
Usable KV Cache Pool: {usable_kv_gb:.2f} GB per GPU

2. CONCURRENCY & THROUGHPUT PER GPU
--------------------------------------------------------------------------------
Max Safe Concurrency: {max_concurrency:,} Concurrent Streams (Zero Preemption Ceiling)
Avg Request Latency:  {avg_latency_s:.2f} seconds
Sustainable QPS/GPU:  {qps_per_gpu:.2f} QPS

3. CLUSTER SIZING RECOMMENDATION
--------------------------------------------------------------------------------
Target Peak Load:     {target_qps:.1f} QPS
GPUs Required:        {int(sizing['gpus_required'])}x {gpu_key.upper()} (with {sizing['headroom_percent']}% headroom)
Cluster Max Capacity: {sizing['cluster_max_qps']:.1f} QPS
================================================================================
"""


def main():
    parser = argparse.ArgumentParser(
        description="LLM Production Serving Capacity Calculator"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_PRESETS.keys()),
        default="llama-3-8b",
        help="Model architecture preset",
    )
    parser.add_argument(
        "--gpu",
        choices=list(GPU_PRESETS.keys()),
        default="a100-80gb",
        help="GPU hardware preset",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=1500,
        help="Average input prompt token count",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=500,
        help="Max generated output token count",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=200.0,
        help="Target peak QPS for the serving cluster",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=4.2,
        help="Average request end-to-end latency in seconds",
    )
    parser.add_argument(
        "--prefix-cache",
        type=float,
        default=0.0,
        help="Prefix cache hit rate fraction (0.0 to 1.0)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism size (e.g. 1, 2, 4, 8)",
    )

    args = parser.parse_args()
    report = format_sizing_report(
        model_key=args.model,
        gpu_key=args.gpu,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        target_qps=args.qps,
        avg_latency_s=args.latency,
        prefix_cache_hit_rate=args.prefix_cache,
        tensor_parallel_size=args.tp,
    )
    print(report)


if __name__ == "__main__":
    main()
