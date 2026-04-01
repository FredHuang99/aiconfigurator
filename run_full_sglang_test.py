#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HunYuanDense 7B SGLang 全硬件完整性能测试

测试所有 SGLang 支持的硬件在 1/2/4/8 GPU 配置下的完整 ISL×OSL 矩阵。

使用方法:
    python run_full_sglang_test.py

输出文件:
    test_results/{system}_{gpu}gpu_{timestamp}.csv
"""

import itertools
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from aiconfigurator.cli.api import cli_default

# 模型路径
ENHANCER_MODEL_PATH = "/data/projects/myproject/aiconfigurator/enhancer_models/tencent--HunyuanImage-2.1--reprompt"

# ISL 范围 (16 个值: 128 到 2048, 步长 128)
ISL_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]

# OSL 范围 (16 个值: 128 到 2048, 步长 128)
OSL_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]

# SGLang 支持的系统
SGLANG_SYSTEMS = ["h100_sxm", "h200_sxm", "a100_sxm", "b200_sxm", "gb200", "gb300", "l40s"]

# GPU 数量
GPU_COUNTS = [1, 2, 4, 8]


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def test_single_combination(isl: int, osl: int, gpu: int, system: str) -> dict:
    """
    测试单个 ISL/OSL 组合 (使用 AGG 聚合模式，不使用 PD 分离)
    """
    try:
        result = cli_default(
            model_path=ENHANCER_MODEL_PATH,
            total_gpus=gpu,
            system=system,
            backend="sglang",
            isl=isl,
            osl=osl,
            database_mode="HYBRID",
        )

        agg_df = result.best_configs.get("agg")
        if agg_df is not None and not agg_df.empty:
            row = agg_df.iloc[0]
            return {
                "model": ENHANCER_MODEL_PATH,
                "system": system,
                "backend": "sglang",
                "gpu": gpu,
                "isl": isl,
                "osl": osl,
                "ttft": row.get("ttft", 0),
                "tpot": row.get("tpot", 0),
                "tokens_per_s": row.get("tokens/s", 0),
                "tokens_per_s_gpu": row.get("tokens/s/gpu", 0),
                "request_latency": row.get("request_latency", 0),
                "memory_gb": row.get("memory", 0),
                "concurrency": row.get("concurrency", 0),
            }
        return None

    except Exception as e:
        print(f"    失败: {e}")
        return None


def test_full_matrix(gpu: int, system: str) -> pd.DataFrame:
    """
    测试完整的 ISL x OSL 矩阵
    """
    print(f"\n{'=' * 70}")
    print(f"开始测试: {system}, {gpu} GPU")
    print(f"ISL 范围: {ISL_VALUES[0]} ~ {ISL_VALUES[-1]} (共 {len(ISL_VALUES)} 个)")
    print(f"OSL 范围: {OSL_VALUES[0]} ~ {OSL_VALUES[-1]} (共 {len(OSL_VALUES)} 个)")
    print(f"{'=' * 70}\n")

    results = []
    total = len(ISL_VALUES) * len(OSL_VALUES)
    start_time = time.time()

    for i, (isl, osl) in enumerate(itertools.product(ISL_VALUES, OSL_VALUES)):
        print(f"  ISL={isl}, OSL={osl}, GPU={gpu}, System={system}...", end=" ", flush=True)

        metrics = test_single_combination(isl, osl, gpu, system)
        if metrics:
            results.append(metrics)
            print(f"TTFT={metrics['ttft']:.2f}ms, TPOT={metrics['tpot']:.4f}ms, "
                  f"Tokens/s={metrics['tokens_per_s']:.1f}, Mem={metrics['memory_gb']:.2f}GB")
        else:
            print("无结果")

        # 进度报告
        if (i + 1) % 20 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            progress = (i + 1) / total * 100
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"\n  进度: {progress:.1f}% ({i + 1}/{total}) "
                  f"已用: {elapsed:.1f}s 预计剩余: {eta:.1f}s\n")

    elapsed_total = time.time() - start_time
    print(f"\n完成! 总用时: {elapsed_total:.1f}s, 成功: {len(results)}/{total}")

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, system: str, gpu: int) -> str:
    """保存结果到 CSV"""
    os.makedirs("test_results", exist_ok=True)
    timestamp = get_timestamp()
    filename = f"test_results/hunyuan_sglang_{system}_{gpu}gpu_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"结果已保存到: {filename}")
    return filename


def main():
    timestamp = get_timestamp()
    print(f"\n{'=' * 70}")
    print(f"HunYuanDense 7B SGLang 全硬件完整性能测试")
    print(f"开始时间: {timestamp}")
    print(f"系统: {SGLANG_SYSTEMS}")
    print(f"GPU 配置: {GPU_COUNTS}")
    print(f"ISL×OSL 组合: {len(ISL_VALUES)}×{len(OSL_VALUES)} = {len(ISL_VALUES) * len(OSL_VALUES)}")
    print(f"总计测试: {len(SGLANG_SYSTEMS)} × {len(GPU_COUNTS)} × {len(ISL_VALUES) * len(OSL_VALUES)} = "
          f"{len(SGLANG_SYSTEMS) * len(GPU_COUNTS) * len(ISL_VALUES) * len(OSL_VALUES)}")
    print(f"{'=' * 70}\n")

    os.makedirs("test_results", exist_ok=True)

    all_results = []
    total_tests = len(SGLANG_SYSTEMS) * len(GPU_COUNTS)
    test_count = 0
    total_start_time = time.time()

    for system, gpu in itertools.product(SGLANG_SYSTEMS, GPU_COUNTS):
        test_count += 1
        test_start_time = time.time()

        print(f"\n{'=' * 70}")
        print(f"[{test_count}/{total_tests}] 测试 {system} @ {gpu} GPU")
        print(f"{'=' * 70}")

        df = test_full_matrix(gpu, system)

        if not df.empty:
            save_results(df, system, gpu)
            all_results.append(df)

        test_elapsed = time.time() - test_start_time
        total_elapsed = time.time() - total_start_time
        avg_time_per_test = total_elapsed / test_count
        remaining_tests = total_tests - test_count
        eta = avg_time_per_test * remaining_tests

        print(f"\n本轮用时: {test_elapsed:.1f}s")
        print(f"总已用时: {total_elapsed:.1f}s")
        print(f"预计剩余: {eta:.1f}s ({eta/3600:.1f} 小时)")
        print(f"总体进度: {test_count}/{total_tests} ({(test_count/total_tests*100):.1f}%)")

    # 生成汇总报告
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        summary_path = f"test_results/hunyuan_sglang_all_systems_summary_{timestamp}.csv"
        combined.to_csv(summary_path, index=False)
        print(f"\n{'=' * 70}")
        print(f"所有测试完成!")
        print(f"总用时: {(time.time() - total_start_time)/3600:.2f} 小时")
        print(f"汇总报告已保存到: {summary_path}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
