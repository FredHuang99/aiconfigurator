#!/usr/bin/env python3
import itertools
from aiconfigurator.cli.api import cli_default
import pandas as pd

MODEL_PATH = "/data/projects/myproject/aiconfigurator/enhancer_models/tencent--HunyuanImage-2.1--reprompt"
ISL_VALUES = [1024]
OSL_VALUES = [512]

results = []
for isl, osl in itertools.product(ISL_VALUES, OSL_VALUES):
    try:
        result = cli_default(
            model_path=MODEL_PATH,
            total_gpus=1,
            system="h100_sxm",
            backend="sglang",
            isl=isl,
            osl=osl,
            database_mode="HYBRID",
        )
        agg_df = result.best_configs.get("agg")
        if agg_df is not None and not agg_df.empty:
            row = agg_df.iloc[0]
            results.append({
                "model": MODEL_PATH,
                "system": "h100_sxm",
                "backend": "sglang",
                "gpu": 1,
                "isl": isl,
                "osl": osl,
                "ttft": row.get("ttft", 0),
                "tpot": row.get("tpot", 0),
                "tokens_per_s": row.get("tokens/s", 0),
                "memory_gb": row.get("memory", 0),
            })
            print(f"TTFT={row.get('ttft', 0):.2f}ms, TPOT={row.get('tpot', 0):.4f}ms, Mem={row.get('memory', 0):.2f}GB")
    except Exception as e:
        print(f"Error: {e}")

df = pd.DataFrame(results)
df.to_csv("test_results/quick_test.csv", index=False)
print("Done!")
