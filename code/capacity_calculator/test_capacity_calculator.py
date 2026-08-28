"""
Unit tests for the Capacity Planning & KV Cache Sizing Engine.
"""

import pytest
from .calculator import (
    CapacityCalculator,
    ModelArchitecture,
    GPUHardware,
    MODEL_PRESETS,
    GPU_PRESETS,
    format_sizing_report,
)


def test_kv_bytes_per_token_llama3_8b():
    # Llama-3-8B: 32 layers, 8 kv heads, 128 head_dim, 2 bytes (FP16)
    # Formula: 2 * 32 * 8 * 128 * 2 = 131,072 bytes (128 KB)
    model = MODEL_PRESETS["llama-3-8b"]
    kv_bytes = CapacityCalculator.calculate_kv_bytes_per_token(model)
    assert kv_bytes == 131072


def test_kv_bytes_per_token_fp8():
    model = MODEL_PRESETS["llama-3-8b"]
    kv_bytes_fp8 = CapacityCalculator.calculate_kv_bytes_per_token(model, precision_bytes=1)
    assert kv_bytes_fp8 == 65536  # Half of FP16


def test_request_kv_bytes():
    model = MODEL_PRESETS["llama-3-8b"]
    # 1500 prompt + 500 output = 2000 tokens
    # 2000 * 131,072 bytes = 262,144,000 bytes = 250 MB
    req_bytes = CapacityCalculator.calculate_request_kv_bytes(model, prompt_tokens=1500, output_tokens=500)
    assert req_bytes == 2000 * 131072


def test_usable_kv_pool_a100():
    gpu = GPU_PRESETS["a100-80gb"]
    model = MODEL_PRESETS["llama-3-8b"]
    # Total VRAM = 80GB, Util = 0.90 -> 72GB
    # Weights = 8.03 * 2 = 16.06 GB
    # Overhead = 4.0 GB
    # Usable = 72 - 16.06 - 4.0 = 51.94 GB
    usable_gb = CapacityCalculator.calculate_usable_kv_pool_gb(
        gpu=gpu, model=model, gpu_memory_utilization=0.90, overhead_gb=4.0
    )
    assert round(usable_gb, 2) == 51.94


def test_max_concurrency_calculation():
    gpu = GPU_PRESETS["a100-80gb"]
    model = MODEL_PRESETS["llama-3-8b"]
    concurrency = CapacityCalculator.calculate_max_concurrency(
        gpu=gpu,
        model=model,
        prompt_tokens=1500,
        output_tokens=500,
        gpu_memory_utilization=0.90,
        overhead_gb=4.0,
    )
    # 51.94 GB = 55,769,720,012 bytes
    # Request bytes = 262,144,000 bytes
    # 55,769,720,012 // 262,144,000 = 212
    assert concurrency >= 200
    assert concurrency <= 220


def test_prefix_caching_concurrency_boost():
    gpu = GPU_PRESETS["a100-80gb"]
    model = MODEL_PRESETS["llama-3-8b"]
    base_concurrency = CapacityCalculator.calculate_max_concurrency(
        gpu=gpu, model=model, prompt_tokens=1500, output_tokens=500, prefix_cache_hit_rate=0.0
    )
    cached_concurrency = CapacityCalculator.calculate_max_concurrency(
        gpu=gpu, model=model, prompt_tokens=1500, output_tokens=500, prefix_cache_hit_rate=0.50
    )
    # 50% cache hit on 1500 prompt tokens reduces sequence length from 2000 to 1250 tokens
    # Concurrency should increase by ~1.6x
    assert cached_concurrency > base_concurrency
    assert abs(cached_concurrency - int(base_concurrency * (2000 / 1250))) <= 1


def test_cluster_sizing():
    # Target 200 QPS with 48 QPS per GPU -> 200 / 48 = 4.16 GPUs
    # With 1.25 headroom -> 4.16 * 1.25 = 5.2 -> 6 GPUs (or 5 depending on TP)
    sizing = CapacityCalculator.calculate_cluster_sizing(
        target_peak_qps=200.0, qps_per_gpu=48.3, headroom_multiplier=1.25, tensor_parallel_size=1
    )
    assert sizing["gpus_required"] >= 5


def test_format_sizing_report_runs():
    report = format_sizing_report(
        model_key="llama-3-8b",
        gpu_key="a100-80gb",
        prompt_tokens=1500,
        output_tokens=500,
        target_qps=200.0,
    )
    assert "LLM SERVING CAPACITY & SIZING REPORT" in report
    assert "Meta Llama-3-8B" in report
    assert "Max Safe Concurrency" in report
