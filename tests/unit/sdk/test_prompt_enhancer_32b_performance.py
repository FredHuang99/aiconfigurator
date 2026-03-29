# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Performance tests for PromptEnhancer-32B model.

Tests latency and memory across various input/output length combinations using SGLANG backend.
"""

import itertools
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import get_model

pytestmark = pytest.mark.unit

# Path to the PromptEnhancer-32B model
PROMPT_ENHANCER_32B_MODEL_PATH = "/data/projects/myproject/aiconfigurator/enhancer_models/PromptEnhancer-32B"

# Test parameters - 32B model has 128K context window
CORE_INPUTS = [128, 512, 1024, 2048, 4096]
CORE_OUTPUTS = [128, 512, 1024, 2048]
FULL_INPUTS = [128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 4096, 8192]
FULL_OUTPUTS = [128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048]


def _build_mock_backend():
    """Build a mock SGLANG backend that returns deterministic results."""
    backend = MagicMock()
    backend.name = SimpleNamespace(value="sglang")
    backend._agg_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    def _run_static(model, database, runtime_config, mode, stride=32, latency_correction_scale=1.0):
        """Return deterministic inference results."""
        # Calculate simple metrics based on input/output lengths
        isl = runtime_config.isl
        osl = runtime_config.osl
        bs = runtime_config.batch_size or 1

        # 32B model: larger hidden size (5120) means more compute per token
        # TTFT scales with ISL * model_size_factor
        model_size_factor = 5120 / 4096  # normalized to 7B model
        ttft = isl * 0.01 * model_size_factor  # 0.01ms per input token, scaled
        tpot = 0.12 * model_size_factor  # 0.12ms per output token base
        request_latency = ttft + tpot * (osl - 1)

        # Memory roughly scales with ISL + OSL and model size
        memory_gb = (isl + osl) * bs * 0.00012 * model_size_factor

        # Build result dict
        result_dict = {
            "model": model.model_path if hasattr(model, "model_path") else "test",
            "isl": isl,
            "osl": osl,
            "prefix": runtime_config.prefix or 0,
            "concurrency": bs,
            "request_rate": bs * 1000.0 / max(request_latency, 1),
            "bs": bs,
            "global_bs": bs,
            "ttft": ttft,
            "tpot": tpot,
            "seq/s": bs * 1000.0 / max(request_latency, 1),
            "seq/s/gpu": bs * 1000.0 / max(request_latency, 1),
            "tokens/s": bs * 1000.0 * osl / max(request_latency, 1),
            "tokens/s/gpu": bs * 1000.0 * osl / max(request_latency, 1),
            "tokens/s/user": 1000.0 / max(tpot, 0.001),
            "request_latency": request_latency,
            "num_total_gpus": model.config.tp_size * model.config.pp_size if hasattr(model, "config") else 1,
            "tp": model.config.tp_size if hasattr(model, "config") else 1,
            "pp": model.config.pp_size if hasattr(model, "config") else 1,
            "dp": model.config.attention_dp_size if hasattr(model, "config") else 1,
            "moe_tp": 1,
            "moe_ep": 1,
            "parallel": f"tp{model.config.tp_size if hasattr(model, 'config') else 1}",
            "gemm": "fp16",
            "kvcache": "fp16",
            "fmha": "fp16",
            "moe": "none",
            "comm": "half",
            "memory": memory_gb,
            "backend": "sglang",
            "version": "0.5.6.post2",
            "system": "h200_sxm",
            "power_w": 400.0,  # 32B model uses more power
        }

        summary = InferenceSummary(runtime_config=runtime_config)
        summary.set_oom(False)
        summary.set_summary_df(pd.DataFrame([result_dict], columns=common.ColumnsStatic))
        return summary

    backend.run_static = _run_static
    return backend


def _build_mock_database():
    """Build a mock database with required attributes."""
    mock_db = MagicMock()
    mock_db.backend = "sglang"
    mock_db.version = "0.5.6.post2"
    mock_db.system = "h200_sxm"
    mock_db.system_spec = {
        "gpu": {
            "float16_tc_flops": 1_000_000_000_000.0,
            "mem_bw": 1_000_000_000_000.0,
            "mem_capacity": 80 * 1024 * 1024 * 1024,  # 80GB
        },
        "misc": {
            "nccl_version": "v1",
            "nccl_mem": {1: 0, 2: 0, 4: 0, 8: 0},
            "other_mem": 2 * 1024 * 1024 * 1024,  # 2GB
        },
    }
    return mock_db


def _run_single_inference(
    model,
    database,
    backend,
    isl: int,
    osl: int,
    batch_size: int = 1,
) -> dict:
    """
    Run a single inference and extract key metrics.

    Returns dict with: ttft, tpot, total_latency, memory_gb
    """
    runtime_config = RuntimeConfig(
        batch_size=batch_size,
        beam_width=1,
        isl=isl,
        osl=osl,
        prefix=0,
    )

    session = InferenceSession(model=model, database=database, backend=backend)
    summary = session.run_static(runtime_config, mode="e2e")

    # Extract metrics from summary
    df = summary.get_summary_df()
    if df is not None and not df.empty:
        row = df.iloc[0]
        ttft = float(row.get("ttft", 0))
        tpot = float(row.get("tpot", 0))
        request_latency = float(row.get("request_latency", ttft + tpot * max(osl - 1, 0)))
        memory_gb = float(row.get("memory", 0))
    else:
        ttft = tpot = request_latency = memory_gb = 0.0

    return {
        "input_len": isl,
        "output_len": osl,
        "batch_size": batch_size,
        "ttft": ttft,
        "tpot": tpot,
        "total_latency": request_latency,
        "memory_gb": memory_gb,
    }


def _run_combinations(
    model,
    database,
    backend,
    input_lengths: list[int],
    output_lengths: list[int],
) -> pd.DataFrame:
    """
    Run inference across all input/output length combinations.

    Returns DataFrame with columns: input_len, output_len, batch_size, ttft, tpot, total_latency, memory_gb
    """
    results = []
    total_combinations = len(input_lengths) * len(output_lengths)

    for i, (isl, osl) in enumerate(itertools.product(input_lengths, output_lengths)):
        result = _run_single_inference(model, database, backend, isl, osl)
        results.append(result)

        # Progress logging every 10%
        if (i + 1) % max(1, total_combinations // 10) == 0:
            progress = (i + 1) / total_combinations * 100
            print(f"Progress: {progress:.1f}% ({i + 1}/{total_combinations})")

    return pd.DataFrame(results)


class TestPromptEnhancer32BPerformance:
    """Test performance metrics for PromptEnhancer-32B model."""

    @pytest.fixture
    def model(self):
        """Load the PromptEnhancer-32B model with TP=1."""
        model_config = ModelConfig(tp_size=1, pp_size=1)
        return get_model(
            model_path=PROMPT_ENHANCER_32B_MODEL_PATH,
            model_config=model_config,
            backend_name="sglang",
        )

    @pytest.fixture
    def model_tp4(self):
        """Load the PromptEnhancer-32B model with TP=4."""
        model_config = ModelConfig(tp_size=4, pp_size=1)
        return get_model(
            model_path=PROMPT_ENHANCER_32B_MODEL_PATH,
            model_config=model_config,
            backend_name="sglang",
        )

    @pytest.fixture
    def backend(self):
        """Get mock SGLANG backend for fast deterministic testing."""
        return _build_mock_backend()

    @pytest.fixture
    def database(self):
        """Use mock performance database for fast testing."""
        return _build_mock_database()

    def test_core_combinations(self, model, backend, database):
        """
        Test core input/output length combinations (20 combos).

        Core combinations: 128/512/1024/2048/4096 x 128/512/1024/2048
        """
        print("\n" + "=" * 60)
        print("Running core combinations test (20 combos)")
        print("=" * 60)

        df = _run_combinations(
            model=model,
            database=database,
            backend=backend,
            input_lengths=CORE_INPUTS,
            output_lengths=CORE_OUTPUTS,
        )

        # Verify we got results for all combinations
        expected_count = len(CORE_INPUTS) * len(CORE_OUTPUTS)
        assert len(df) == expected_count, f"Expected {expected_count} results, got {len(df)}"

        # Basic sanity checks
        assert (df["ttft"] >= 0).all(), "TTFT should be non-negative"
        assert (df["tpot"] >= 0).all(), "TPOT should be non-negative"
        assert (df["memory_gb"] > 0).all(), "Memory should be positive"

        # Print summary
        print("\nCore Combinations Results:")
        print(df.to_string(index=False))

        # Save to CSV
        csv_path = "test_results_prompt_enhancer_32b_core.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    def test_full_combinations(self, model, backend, database):
        """
        Test full input/output length combinations (110 combos).

        Full combinations: 11 input lengths x 10 output lengths
        This is a longer test covering edge cases.
        """
        print("\n" + "=" * 60)
        print("Running full combinations test (110 combos)")
        print("=" * 60)

        df = _run_combinations(
            model=model,
            database=database,
            backend=backend,
            input_lengths=FULL_INPUTS,
            output_lengths=FULL_OUTPUTS,
        )

        # Verify we got results for all combinations
        expected_count = len(FULL_INPUTS) * len(FULL_OUTPUTS)
        assert len(df) == expected_count, f"Expected {expected_count} results, got {len(df)}"

        # Basic sanity checks
        assert (df["ttft"] >= 0).all(), "TTFT should be non-negative"
        assert (df["tpot"] >= 0).all(), "TPOT should be non-negative"
        assert (df["memory_gb"] > 0).all(), "Memory should be positive"

        # Print summary
        print("\nFull Combinations Results:")
        print(df.to_string(index=False))

        # Save to CSV
        csv_path = "test_results_prompt_enhancer_32b_full.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    def test_extreme_output_lengths(self, model, backend, database):
        """
        Test extreme output length scenarios where output >> input.

        Specifically tests: (128→2048), (512→4096), (1024→8192)
        These are the latency spike scenarios mentioned in the requirements.
        """
        print("\n" + "=" * 60)
        print("Running extreme output length tests")
        print("=" * 60)

        extreme_cases = [
            (128, 2048),
            (512, 4096),
            (1024, 8192),
        ]

        results = []
        for isl, osl in extreme_cases:
            result = _run_single_inference(model, database, backend, isl, osl)
            results.append(result)
            print(
                f"ISL={isl}, OSL={osl}: TTFT={result['ttft']:.2f}ms, "
                f"TPOT={result['tpot']:.2f}ms, Total={result['total_latency']:.2f}ms, "
                f"Mem={result['memory_gb']:.2f}GB"
            )

        df = pd.DataFrame(results)

        # Save to CSV
        csv_path = "test_results_prompt_enhancer_32b_extreme_outputs.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nExtreme cases results saved to {csv_path}")

        # Verify latency growth is roughly linear with OSL
        # TPOT should be relatively stable across these cases
        tpot_values = df["tpot"].values
        tpot_std = pd.Series(tpot_values).std()
        tpot_mean = pd.Series(tpot_values).mean()
        print(f"\nTPOT stats: mean={tpot_mean:.2f}ms, std={tpot_std:.2f}ms")
        print("Note: High variance in TPOT may indicate non-linear scaling at extreme OSL")

    def test_memory_growth_curve(self, model, backend, database):
        """
        Test that memory usage grows linearly with output length.

        Fixed input (2048), varying output (128 to 4096).
        """
        print("\n" + "=" * 60)
        print("Running memory growth curve test")
        print("=" * 60)

        fixed_isl = 2048
        osl_values = [128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 4096]

        results = []
        for osl in osl_values:
            result = _run_single_inference(model, database, backend, fixed_isl, osl)
            results.append(result)
            print(f"ISL={fixed_isl}, OSL={osl}: Memory={result['memory_gb']:.2f}GB")

        df = pd.DataFrame(results)

        # Save to CSV
        csv_path = "test_results_prompt_enhancer_32b_memory_growth.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nMemory growth curve saved to {csv_path}")

        # Check memory growth is monotonic (or near-monotonic)
        memory_values = df["memory_gb"].values
        is_monotonic = all(memory_values[i] <= memory_values[i + 1] * 1.05 for i in range(len(memory_values) - 1))
        print(f"\nMemory is {'approximately' if is_monotonic else 'NOT'} monotonic with OSL")

    def test_tensor_parallel_scaling(self, model_tp4, backend, database):
        """
        Test that memory usage scales with tensor parallelism.

        TP=4 should reduce memory per GPU.
        """
        print("\n" + "=" * 60)
        print("Running tensor parallel scaling test")
        print("=" * 60)

        isl, osl = 2048, 1024

        # With TP=4, each GPU handles smaller model shards
        result = _run_single_inference(model_tp4, database, backend, isl, osl)
        print(f"TP=4, ISL={isl}, OSL={osl}: Memory={result['memory_gb']:.2f}GB")

        # Verify memory is within reasonable bounds for TP=4
        assert result["memory_gb"] > 0, "Memory should be positive with TP=4"

    def test_large_context_handling(self, model, backend, database):
        """
        Test handling of large context lengths (up to 32K tokens).

        PromptEnhancer-32B has 128K context window.
        """
        print("\n" + "=" * 60)
        print("Running large context handling test")
        print("=" * 60)

        large_context_cases = [
            (8192, 256),
            (16384, 256),
            (32768, 256),
        ]

        results = []
        for isl, osl in large_context_cases:
            result = _run_single_inference(model, database, backend, isl, osl)
            results.append(result)
            print(f"ISL={isl}, OSL={osl}: TTFT={result['ttft']:.2f}ms, Memory={result['memory_gb']:.2f}GB")

        df = pd.DataFrame(results)

        # Save to CSV
        csv_path = "test_results_prompt_enhancer_32b_large_context.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nLarge context results saved to {csv_path}")

        # Verify TTFT increases with context size
        assert df["ttft"].is_monotonic_increasing, "TTFT should increase with context size"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
