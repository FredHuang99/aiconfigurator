#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Quick verification test for HunYuanDense 7B SGLang on different systems and GPU configs
# Usage: bash verify_sglang_tests.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

SYSTEMS="h100_sxm h200_sxm a100_sxm b200_sxm gb200 gb300 l40s"
GPUS="1 2 4 8"

# Quick test with small ISL/OSL
ISL=512
OSL=256

echo "============================================================"
echo "HunYuanDense 7B SGLang Quick Verification Test"
echo "============================================================"
echo "Start time: $(date)"
echo "Test config: ISL=$ISL, OSL=$OSL"
echo "Systems: $SYSTEMS"
echo "GPU configs: $GPUS"
echo "============================================================"
echo ""

PYTHON_SCRIPT="/tmp/verify_sglang_${TIMESTAMP}.py"

cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3
import sys
from aiconfigurator.cli.api import cli_default
import pandas as pd

MODEL_PATH = "/data/projects/myproject/aiconfigurator/enhancer_models/tencent--HunyuanImage-2.1--reprompt"

system = sys.argv[1]
gpu = int(sys.argv[2])
isl = int(sys.argv[3])
osl = int(sys.argv[4])

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
        ttft = row.get("ttft", 0)
        tpot = row.get("tpot", 0)
        memory = row.get("memory", 0)
        tokens_s = row.get("tokens/s", 0)
        print(f"SUCCESS: TTFT={ttft:.2f}ms, TPOT={tpot:.4f}ms, Tokens/s={tokens_s:.1f}, Mem={memory:.2f}GB")
        sys.exit(0)
    else:
        print("FAIL: No agg results returned")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
PYTHON_EOF

PASSED=0
FAILED=0
FAILED_LIST=""

for SYSTEM in $SYSTEMS; do
    for GPU in $GPUS; do
        echo -n "[$SYSTEM @ ${GPU} GPU] Testing... "

        OUTPUT=$(python3 "$PYTHON_SCRIPT" "$SYSTEM" "$GPU" "$ISL" "$OSL" 2>&1)
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "PASS"
            PASSED=$((PASSED + 1))
        else
            echo "FAIL"
            echo "  $OUTPUT" | head -3
            FAILED=$((FAILED + 1))
            FAILED_LIST="$FAILED_LIST $SYSTEM@${GPU}GPU"
        fi
    done
done

echo ""
echo "============================================================"
echo "Verification Results"
echo "============================================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
if [ $FAILED -gt 0 ]; then
    echo "Failed combinations:$FAILED_LIST"
fi
echo "============================================================"

rm -f "$PYTHON_SCRIPT"

if [ $FAILED -gt 0 ]; then
    exit 1
fi

echo "All combinations verified! You can now run the full test safely."
