#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# HunYuanDense 7B SGLang Full Hardware Performance Test
# Usage: bash run_all_sglang_tests.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="test_results/test_all_${TIMESTAMP}.log"
PYTHON_SCRIPT="/tmp/run_sglang_test_${TIMESTAMP}.py"

mkdir -p test_results

SYSTEMS="h100_sxm h200_sxm a100_sxm b200_sxm gb200 gb300 l40s"
GPUS="1 2 4 8"

TOTAL_SYSTEMS=7
TOTAL_GPUS=4
TOTAL_COMBINATIONS=$((TOTAL_SYSTEMS * TOTAL_GPUS))

# Create Python test script
cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3
import sys
import itertools
from aiconfigurator.cli.api import cli_default
import pandas as pd

MODEL_PATH = "/data/projects/myproject/aiconfigurator/enhancer_models/tencent--HunyuanImage-2.1--reprompt"
MODEL_NAME = "tencent--HunyuanImage-2.1--reprompt"
ISL_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
OSL_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]

system = sys.argv[1]
gpu = int(sys.argv[2])
output_file = sys.argv[3]

results = []
total = len(ISL_VALUES) * len(OSL_VALUES)

for i, (isl, osl) in enumerate(itertools.product(ISL_VALUES, OSL_VALUES)):
    try:
        result = cli_default(
            model_path=MODEL_PATH,
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
            results.append({
                "model": MODEL_NAME,
                "isl": isl,
                "osl": osl,
                "prefix": row.get("prefix", 0),
                "concurrency": row.get("concurrency", 0),
                "request_rate": row.get("request_rate", 0),
                "bs": row.get("bs", 0),
                "global_bs": row.get("global_bs", 0),
                "ttft": row.get("ttft", 0),
                "tpot": row.get("tpot", 0),
                "request_latency": row.get("request_latency", 0),
                "seq/s": row.get("seq/s", 0),
                "seq/s/gpu": row.get("seq/s/gpu", 0),
                "tokens/s": row.get("tokens/s", 0),
                "tokens/s/gpu": row.get("tokens/s/gpu", 0),
                "tokens/s/user": row.get("tokens/s/user", 0),
                "num_total_gpus": row.get("num_total_gpus", 0),
                "tp": row.get("tp", 0),
                "pp": row.get("pp", 0),
                "dp": row.get("dp", 0),
                "moe_tp": row.get("moe_tp", 0),
                "moe_ep": row.get("moe_ep", 0),
                "parallel": row.get("parallel", ""),
                "gemm": row.get("gemm", ""),
                "kvcache": row.get("kvcache", ""),
                "fmha": row.get("fmha", ""),
                "moe": row.get("moe", ""),
                "comm": row.get("comm", ""),
                "memory": row.get("memory", 0),
                "balance_score": row.get("balance_score", 0),
                "num_ctx_reqs": row.get("num_ctx_reqs", 0),
                "num_gen_reqs": row.get("num_gen_reqs", 0),
                "num_tokens": row.get("num_tokens", 0),
                "ctx_tokens": row.get("ctx_tokens", 0),
                "gen_tokens": row.get("gen_tokens", 0),
                "backend": "sglang",
                "version": row.get("version", ""),
                "system": system,
                "power_w": row.get("power_w", 0),
                "tokens/s/gpu_cluster": row.get("tokens/s/gpu_cluster", 0),
                "gpu": gpu,
            })
            print(f"  ISL={isl}, OSL={osl}: TTFT={row.get('ttft', 0):.2f}ms, TPOT={row.get('tpot', 0):.4f}ms, Mem={row.get('memory', 0):.2f}GB")
    except Exception as e:
        print(f"  ISL={isl}, OSL={osl}: Error - {e}")

    if (i + 1) % 32 == 0 or (i + 1) == total:
        print(f"  Progress: {i+1}/{total} ({(i+1)*100//total}%)")

df = pd.DataFrame(results)
df.to_csv(output_file, index=False)
print(f"Done! Success: {len(results)}/256")
print(f"Saved to: {output_file}")
PYTHON_EOF

echo "============================================================"
echo "HunYuanDense 7B SGLang Full Hardware Test"
echo "============================================================"
echo "Start time: $(date)"
echo "Systems: $SYSTEMS"
echo "GPU configs: $GPUS"
echo "Total: $TOTAL_SYSTEMS systems x $TOTAL_GPUS GPU configs x 256 ISL/OSL = 7168 tests"
echo "Log file: $LOG_FILE"
echo "Python script: $PYTHON_SCRIPT"
echo "============================================================"
echo ""

TEST_NUM=0
START_TIME=$(date +%s)

for SYSTEM in $SYSTEMS; do
    for GPU in $GPUS; do
        TEST_NUM=$((TEST_NUM + 1))
        TEST_START=$(date +%s)

        echo "============================================================"
        echo "[$TEST_NUM/$TOTAL_COMBINATIONS] Testing $SYSTEM @ ${GPU} GPU"
        echo "Start time: $(date)"
        echo "============================================================"

        OUTPUT_FILE="test_results/hunyuan_sglang_${SYSTEM}_${GPU}gpu_${TIMESTAMP}.csv"

        python3 "$PYTHON_SCRIPT" "$SYSTEM" "$GPU" "$OUTPUT_FILE" 2>&1 | tee -a "$LOG_FILE"

        TEST_END=$(date +%s)
        TEST_ELAPSED=$((TEST_END - TEST_START))

        TOTAL_ELAPSED=$((TEST_END - START_TIME))
        AVG_TIME=$((TOTAL_ELAPSED / TEST_NUM))
        REMAINING=$((AVG_TIME * (TOTAL_COMBINATIONS - TEST_NUM)))
        PROGRESS=$((TEST_NUM * 100 / TOTAL_COMBINATIONS))

        REMAINING_H=$((REMAINING / 3600))
        REMAINING_M=$((REMAINING % 3600 / 60))

        echo ""
        echo "============================================================"
        echo "This test: ${TEST_ELAPSED}s"
        echo "Total elapsed: ${TOTAL_ELAPSED}s (${REMAINING_H}h ${REMAINING_M}m remaining)"
        echo "Progress: $PROGRESS%"
        echo "============================================================"
        echo ""

        sleep 2
    done
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_H=$((TOTAL_TIME / 3600))
TOTAL_M=$((TOTAL_TIME % 3600 / 60))

echo ""
echo "============================================================"
echo "All tests completed!"
echo "End time: $(date)"
echo "Total time: ${TOTAL_H}h ${TOTAL_M}m"
echo "============================================================"

# Generate summary
echo "Generating summary..."
cat test_results/hunyuan_sglang_*_${TIMESTAMP}.csv 2>/dev/null | head -1 > "test_results/hunyuan_sglang_summary_${TIMESTAMP}.csv"
cat test_results/hunyuan_sglang_*_${TIMESTAMP}.csv 2>/dev/null | grep -v "^model," >> "test_results/hunyuan_sglang_summary_${TIMESTAMP}.csv"

echo "Summary: test_results/hunyuan_sglang_summary_${TIMESTAMP}.csv"
echo "Log: $LOG_FILE"

# Cleanup
rm -f "$PYTHON_SCRIPT"
