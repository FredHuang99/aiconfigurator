#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HunYuanDense 7B SGLang 性能测试脚本

使用方法:
    # 测试单个 ISL/OSL 组合
    python run_hunyuan_sglang_test.py --isl 1024 --osl 512 --gpu 1 --system h100_sxm

    # 测试完整矩阵 (16x16 ISL/OSL 组合)
    python run_hunyuan_sglang_test.py --mode full --gpu 1 --system h100_sxm

    # 测试所有系统
    python run_hunyuan_sglang_test.py --mode full --gpu 1

输出文件:
    test_results/{system}_{gpu}gpu_{timestamp}.csv
"""

import argparse
import itertools
import os
import time
from datetime import datetime

import pandas as pd

# CLI API
from aiconfigurator.cli.api import cli_default

# 模型路径
ENHANCER_MODEL_PATH = "/data/projects/myproject/aiconfigurator/enhancer_models/tencent--HunyuanImage-2.1--reprompt"

# ISL 范围 (16 个值: 128 到 2048, 步长 128)
ISL_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]

# OSL 范围 (16 个值: 128 到 2048, 步长 128)
OSL_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]

# SGLang 支持的系统
SGLANG_SYSTEMS = ["h100_sxm", "h200_sxm", "a100_sxm", "b200_sxm", "gb200", "gb300", "l40s"]


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def test_single_combination(isl: int, osl: int, gpu: int, system: str) -> dict:
    """
    测试单个 ISL/OSL 组合 (使用 AGG 聚合模式，不使用 PD 分离)
    """
    print(f"  ISL={isl}, OSL={osl}, GPU={gpu}, System={system}...", end=" ", flush=True)

    try:
        # 使用 cli_default API
        result = cli_default(
            model_path=ENHANCER_MODEL_PATH,
            total_gpus=gpu,
            system=system,
            backend="sglang",
            isl=isl,
            osl=osl,
            database_mode="HYBRID",  # 使用 HYBRID 模式因为 HunYuanDense 没有 SILICON 数据
        )

        # 提取 agg (聚合模式) 最佳配置的结果
        agg_df = result.best_configs.get("agg")
        if agg_df is not None and not agg_df.empty:
            row = agg_df.iloc[0]
            metrics = {
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
            print(f"TTFT={metrics['ttft']:.2f}ms, TPOT={metrics['tpot']:.4f}ms, "
                  f"Tokens/s={metrics['tokens_per_s']:.1f}, Mem={metrics['memory_gb']:.2f}GB")
            return metrics
        else:
            print("无结果")
            return None

    except Exception as e:
        print(f"失败: {e}")
        return None


def test_full_matrix(gpu: int, system: str) -> pd.DataFrame:
    """
    测试完整的 ISL x OSL 矩阵
    """
    print(f"\n{'=' * 70}")
    print(f"完整矩阵测试: {system}, {gpu} GPU")
    print(f"ISL 范围: {ISL_VALUES[0]} ~ {ISL_VALUES[-1]} (共 {len(ISL_VALUES)} 个)")
    print(f"OSL 范围: {OSL_VALUES[0]} ~ {OSL_VALUES[-1]} (共 {len(OSL_VALUES)} 个)")
    print(f"{'=' * 70}\n")

    results = []
    total = len(ISL_VALUES) * len(OSL_VALUES)
    start_time = time.time()

    for i, (isl, osl) in enumerate(itertools.product(ISL_VALUES, OSL_VALUES)):
        metrics = test_single_combination(isl, osl, gpu, system)
        if metrics:
            results.append(metrics)

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
    parser = argparse.ArgumentParser(description="HunYuanDense SGLang 性能测试")
    parser.add_argument("--mode", choices=["single", "full"], default="single",
                        help="测试模式: single=单个组合, full=完整矩阵")
    parser.add_argument("--isl", type=int, default=1024, help="输入序列长度")
    parser.add_argument("--osl", type=int, default=512, help="输出序列长度")
    parser.add_argument("--gpu", type=int, default=1, choices=[1, 2, 4, 8], help="GPU 数量")
    parser.add_argument("--system", default="h100_sxm", choices=SGLANG_SYSTEMS, help="系统类型")
    parser.add_argument("--all-systems", action="store_true", help="测试所有系统")
    parser.add_argument("--all-gpus", action="store_true", help="测试所有 GPU 配置")

    args = parser.parse_args()

    if args.mode == "single":
        # 单个组合测试
        result = test_single_combination(args.isl, args.osl, args.gpu, args.system)
        if result:
            df = pd.DataFrame([result])
            save_results(df, args.system, args.gpu)

    elif args.mode == "full":
        # 完整矩阵测试
        systems = SGLANG_SYSTEMS if args.all_systems else [args.system]
        gpus = [1, 2, 4, 8] if args.all_gpus else [args.gpu]

        all_results = []
        for system, gpu in itertools.product(systems, gpus):
            df = test_full_matrix(gpu, system)
            if not df.empty:
                save_results(df, system, gpu)
                all_results.append(df)

        # 生成汇总报告
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            summary_path = f"test_results/hunyuan_sglang_summary_{get_timestamp()}.csv"
            combined.to_csv(summary_path, index=False)
            print(f"\n汇总报告已保存到: {summary_path}")


if __name__ == "__main__":
    main()
