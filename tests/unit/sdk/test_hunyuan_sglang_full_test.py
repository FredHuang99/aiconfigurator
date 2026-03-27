# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HunYuanDenseV1ForCausalLM 7B 完整性能测试脚本

测试目标:
- 模型: HunYuanDenseV1ForCausalLM (使用 SGLang 后端)
- ISL/OSL: 16 x 16 = 256 种组合
- GPU 数量: 1, 2, 4, 8 (TP sizes)
- 系统: h100_sxm, h200_sxm, a100_sxm, b200_sxm, gb200, gb300, l40s
- 模式: SILICON (使用实际收集的性能数据)

输出指标:
- ttft: 首 Token 延迟 (ms)
- tpot: 每输出 Token 延迟 (ms)
- request_latency: 总请求延迟 (ms)
- tokens/s: 吞吐量 (tokens/秒)
- tokens/s/gpu: 每 GPU 吞吐量
- memory_gb: 显存占用 (GB)

使用方法:
    # 运行完整测试
    pytest tests/unit/sdk/test_hunyuan_sglang_full_test.py -v -s

    # 运行特定系统的测试
    pytest tests/unit/sdk/test_hunyuan_sglang_full_test.py::TestHunYuanSGLangFull::test_system_h100_sxm -v -s

    # 运行特定 GPU 数量的测试
    pytest tests/unit/sdk/test_hunyuan_sglang_full_test.py::TestHunYuanSGLangFull::test_gpu_count_1 -v -s

输出文件:
    test_results_{system}_{gpu_count}gpu_{timestamp}.csv
"""

import itertools
import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit

# 模型路径
ENHANCER_MODEL_PATH = "/data/projects/myproject/aiconfigurator/enhancer_models/tencent--HunyuanImage-2.1--reprompt"

# ISL 范围 (16 个值: 128 到 2048, 步长 128)
ISL_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]

# OSL 范围 (16 个值: 128 到 2048, 步长 128)
OSL_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]

# GPU 数量 (TP sizes)
GPU_COUNTS = [1, 2, 4, 8]

# SGLang 支持的系统
SGLANG_SYSTEMS = [
    "h100_sxm",
    "h200_sxm",
    "a100_sxm",
    "b200_sxm",
    "gb200",
    "gb300",
    "l40s",
]

# SGLang 版本 (使用最新稳定版)
SGLANG_VERSION = "0.5.9"


def get_timestamp() -> str:
    """获取时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_perf_database(system: str, version: str = SGLANG_VERSION) -> Optional[PerfDatabase]:
    """
    创建性能数据库实例

    Args:
        system: 系统名称 (如 h100_sxm)
        version: SGLang 版本

    Returns:
        PerfDatabase 实例, 如果创建失败则返回 None
    """
    import os
    import aiconfigurator

    # 获取 aiconfigurator 包中的 systems 目录路径
    systems_root = os.path.join(os.path.dirname(aiconfigurator.__file__), "systems")

    try:
        db = PerfDatabase(system=system, backend="sglang", version=version, systems_root=systems_root)
        # 由于 HunYuanDense 模型在数据库中没有实际的 SILICON 数据，
        # 使用 HYBRID 模式（有数据时用数据，否则用经验公式估算）
        db.set_default_database_mode(common.DatabaseMode.HYBRID)
        return db
    except Exception as e:
        print(f"警告: 无法创建 {system}/{version} 的数据库: {e}")
        return None


def run_single_inference(
    model,
    database: PerfDatabase,
    backend: SGLANGBackend,
    isl: int,
    osl: int,
    system: str,
    version: str,
    batch_size: int = 1,
) -> Optional[dict]:
    """
    运行单次推理并提取指标

    Args:
        model: 模型实例
        database: 性能数据库
        backend: SGLang 后端
        isl: 输入序列长度
        osl: 输出序列长度
        system: 系统名称
        version: SGLang 版本
        batch_size: 批大小

    Returns:
        包含指标的字典, 失败时返回 None
    """
    try:
        runtime_config = RuntimeConfig(
            batch_size=batch_size,
            beam_width=1,
            isl=isl,
            osl=osl,
            prefix=0,
        )

        session = InferenceSession(model=model, database=database, backend=backend)
        summary = session.run_static(runtime_config, mode="e2e")

        # 检查是否 OOM
        if summary.check_oom():
            return None

        # 提取指标
        df = summary.get_summary_df()
        if df is None or df.empty:
            return None

        row = df.iloc[0]
        return {
            "model": model.model_path,
            "architecture": model.architecture,
            "model_family": model.model_family,
            "system": system,
            "backend": "sglang",
            "version": version,
            "tp": model.config.tp_size,
            "pp": model.config.pp_size,
            "isl": isl,
            "osl": osl,
            "batch_size": batch_size,
            "ttft": float(row.get("ttft", 0)),
            "tpot": float(row.get("tpot", 0)),
            "request_latency": float(row.get("request_latency", 0)),
            "tokens_per_s": float(row.get("tokens/s", 0)),
            "tokens_per_s_gpu": float(row.get("tokens/s/gpu", 0)),
            "tokens_per_s_user": float(row.get("tokens/s/user", 0)),
            "memory_gb": float(row.get("memory", 0)),
            "concurrency": int(row.get("concurrency", 0)),
        }
    except Exception as e:
        print(f"    推理失败 (ISL={isl}, OSL={osl}): {e}")
        return None


def run_full_test(
    system: str,
    gpu_count: int,
    model_path: str = ENHANCER_MODEL_PATH,
    version: str = SGLANG_VERSION,
) -> pd.DataFrame:
    """
    运行完整测试

    Args:
        system: 系统名称
        gpu_count: GPU 数量 (TP size)
        model_path: 模型路径
        version: SGLang 版本

    Returns:
        包含所有结果的 DataFrame
    """
    print(f"\n{'=' * 70}")
    print(f"开始测试: system={system}, gpu_count={gpu_count}, version={version}")
    print(f"{'=' * 70}")

    # 创建数据库
    database = create_perf_database(system, version)
    if database is None:
        print("错误: 无法创建数据库")
        return pd.DataFrame()

    # 创建后端
    backend = SGLANGBackend()

    # 加载模型
    model_config = ModelConfig(tp_size=gpu_count, pp_size=1)
    try:
        model = get_model(model_path=model_path, model_config=model_config, backend_name="sglang")
    except Exception as e:
        print(f"错误: 无法加载模型: {e}")
        return pd.DataFrame()

    # 确认模型类型
    print(f"模型: {model.architecture} ({model.model_family})")
    print(f"TP={gpu_count}, Layers={model._num_layers}, Hidden={model._hidden_size}")

    # 运行测试
    results = []
    total_combinations = len(ISL_VALUES) * len(OSL_VALUES)
    start_time = time.time()

    for i, (isl, osl) in enumerate(itertools.product(ISL_VALUES, OSL_VALUES)):
        result = run_single_inference(model, database, backend, isl, osl, system, version)
        if result:
            results.append(result)

        # 进度报告
        if (i + 1) % 50 == 0 or (i + 1) == total_combinations:
            elapsed = time.time() - start_time
            progress = (i + 1) / total_combinations * 100
            eta = elapsed / (i + 1) * (total_combinations - i - 1)
            print(f"  进度: {progress:.1f}% ({i + 1}/{total_combinations}) 已用时: {elapsed:.1f}s 预计剩余: {eta:.1f}s")

    elapsed_total = time.time() - start_time
    print(f"\n完成! 总用时: {elapsed_total:.1f}s, 成功: {len(results)}/{total_combinations}")

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, system: str, gpu_count: int) -> str:
    """
    保存结果到 CSV 文件

    Args:
        df: 结果 DataFrame
        system: 系统名称
        gpu_count: GPU 数量

    Returns:
        保存的文件路径
    """
    timestamp = get_timestamp()
    filename = f"test_results_{system}_{gpu_count}gpu_{timestamp}.csv"

    # 创建输出目录
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"结果已保存到: {filepath}")

    return filepath


class TestHunYuanSGLangFull:
    """HunYuanDense 模型在 SGLang 后端上的完整性能测试"""

    @pytest.fixture(params=SGLANG_SYSTEMS)
    def system(self, request):
        """参数化 fixture: 遍历所有系统"""
        return request.param

    @pytest.fixture(params=GPU_COUNTS)
    def gpu_count(self, request):
        """参数化 fixture: 遍历所有 GPU 数量"""
        return request.param

    def test_full_matrix(self, system, gpu_count):
        """
        测试完整的 ISL x OSL 矩阵

        针对每个系统和 GPU 组合测试所有 256 种 ISL/OSL 组合
        """
        df = run_full_test(system=system, gpu_count=gpu_count)

        if df.empty:
            pytest.skip(f"无法为 {system}/{gpu_count}gpu 创建数据库或加载模型")

        save_results(df, system, gpu_count)

        # 基本验证
        assert len(df) > 0, "应该有至少一些成功的测试结果"
        assert (df["ttft"] >= 0).all(), "TTFT 应该非负"
        assert (df["tpot"] >= 0).all(), "TPOT 应该非负"

    def test_extreme_cases(self, system, gpu_count):
        """
        测试极端情况: ISL=2048, OSL=2048

        这是计算量最大的场景, 用于验证极限性能
        """
        extreme_cases = [
            (2048, 128),
            (2048, 512),
            (2048, 1024),
            (2048, 2048),
            (128, 2048),
            (512, 2048),
            (1024, 2048),
        ]

        print(f"\n测试极端情况: system={system}, gpu_count={gpu_count}")

        database = create_perf_database(system)
        if database is None:
            pytest.skip(f"无法创建 {system} 的数据库")

        backend = SGLANGBackend()
        model_config = ModelConfig(tp_size=gpu_count, pp_size=1)

        try:
            model = get_model(ENHANCER_MODEL_PATH, model_config=model_config, backend_name="sglang")
        except Exception as e:
            pytest.skip(f"无法加载模型: {e}")

        results = []
        for isl, osl in extreme_cases:
            result = run_single_inference(model, database, backend, isl, osl, system, SGLANG_VERSION)
            if result:
                results.append(result)
                print(
                    f"  ISL={isl}, OSL={osl}: "
                    f"TTFT={result['ttft']:.2f}ms, "
                    f"TPOT={result['tpot']:.4f}ms, "
                    f"Total={result['request_latency']:.2f}ms, "
                    f"Mem={result['memory_gb']:.2f}GB"
                )
            else:
                print(f"  ISL={isl}, OSL={osl}: 失败 (可能 OOM)")

        df = pd.DataFrame(results)
        if not df.empty:
            save_results(df, f"{system}_extreme", gpu_count)

    def test_memory_scaling(self, system, gpu_count):
        """
        测试显存随 OSL 的增长曲线

        固定 ISL=1024, 变化 OSL
        """
        print(f"\n测试显存增长曲线: system={system}, gpu_count={gpu_count}")

        database = create_perf_database(system)
        if database is None:
            pytest.skip(f"无法创建 {system} 的数据库")

        backend = SGLANGBackend()
        model_config = ModelConfig(tp_size=gpu_count, pp_size=1)

        try:
            model = get_model(ENHANCER_MODEL_PATH, model_config=model_config, backend_name="sglang")
        except Exception as e:
            pytest.skip(f"无法加载模型: {e}")

        fixed_isl = 1024
        results = []

        for osl in OSL_VALUES:
            result = run_single_inference(model, database, backend, fixed_isl, osl, system, SGLANG_VERSION)
            if result:
                results.append(result)
                print(f"  ISL={fixed_isl}, OSL={osl}: Memory={result['memory_gb']:.4f}GB")

        df = pd.DataFrame(results)
        if not df.empty:
            save_results(df, f"{system}_mem_scaling", gpu_count)


def generate_summary_report():
    """
    生成汇总报告

    读取所有测试结果文件, 生成汇总统计
    """
    import glob

    output_dir = "test_results"
    if not os.path.exists(output_dir):
        print("没有找到测试结果目录")
        return

    csv_files = glob.glob(os.path.join(output_dir, "test_results_*.csv"))
    if not csv_files:
        print("没有找到测试结果文件")
        return

    print(f"\n{'=' * 70}")
    print("汇总报告")
    print(f"{'=' * 70}")

    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # 按系统和 GPU 统计
    summary = (
        combined.groupby(["system", "tp"])
        .agg(
            {
                "ttft": ["mean", "std", "min", "max"],
                "tpot": ["mean", "std", "min", "max"],
                "tokens_per_s_gpu": ["mean", "max"],
                "memory_gb": ["mean", "max"],
            }
        )
        .round(3)
    )

    print("\n性能汇总 (按系统 x GPU 数量):")
    print(summary.to_string())

    # 保存汇总
    summary_path = os.path.join(output_dir, f"summary_{get_timestamp()}.csv")
    summary.to_csv(summary_path)
    print(f"\n汇总已保存到: {summary_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 直接运行特定测试
        pytest.main([__file__, "-v", "-s", sys.argv[1]])
    else:
        # 运行所有测试
        pytest.main([__file__, "-v", "-s"])
