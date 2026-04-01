#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sweep feasible aggregated SGLang points for prompt-enhancer style models.

This script can profile multiple dense prompt-enhancer presets, including:

- HunyuanImage-2.1 reprompt
- PromptEnhancer-7B
- PromptEnhancer-32B

It profiles all feasible `(batch_size, ttft, tpot)` points under:

- system support available in the local AIC repository
- model context / output length limits when they can be inferred
- GPU memory limits, including KV cache, via AIC's existing OOM checks

The script writes:

- feasible_points_<model>.csv: every non-OOM point
- skipped_cases_<model>.csv: unsupported systems or sequence-length combinations skipped
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.sdk.perf_database import get_latest_database_version
from aiconfigurator.sdk.utils import get_model_config_from_model_path


DEFAULT_MODEL_PRESET = "promptenhancer_32b"
DEFAULT_BACKEND = "sglang"
DEFAULT_DATABASE_MODE = "HYBRID"
DEFAULT_SYSTEM_ALIASES = ["h200", "h100", "h800", "a100", "a800", "5090", "4090"]
DEFAULT_ISL_LIST = [128, 256]
DEFAULT_OSL_LIST = list(range(256, 2049, 128))
DEFAULT_TP_LIST = [1, 2, 4, 8]
MODEL_PRESETS = {
    "promptenhancer_7b": {
        "model_path": "/home/heyang/models/promptenhancer-7b",
    },
    "promptenhancer_32b": {
        "model_path": "/home/heyang/models/promptenhancer-32b",
    },
    "hunyuan_reprompt": {
        "model_path": "tencent/HunyuanImage-2.1/reprompt",
    },
}

# Keep the same dense SGLang batch sweep shape as AIC's agg search path.
'''
DEFAULT_BATCH_CANDIDATES = (
    list(range(1, 16, 1))
    + list(range(16, 32, 4))
    + list(range(32, 64, 8))
    + list(range(64, 256, 16))
    + list(range(256, 512, 32))
    + list(range(512, 1024, 256))
    + [1024]
)
'''
DEFAULT_BATCH_CANDIDATES = [1, 2, 4, 8, 16, 32]

SYSTEM_ALIAS_TO_AIC = {
    "h200": "h200_sxm",
    "h100": "h100_sxm",
    "a100": "a100_sxm",
}


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _sanitize_model_label(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_").lower()


def _resolve_model_selection(model_preset: str, model_path: str | None) -> tuple[str, str]:
    preset = MODEL_PRESETS[model_preset]
    resolved_model_path = model_path or preset["model_path"]
    return resolved_model_path, _sanitize_model_label(model_preset)


def _extract_model_limits(model_path: str, requested_max_osl: int) -> tuple[int, int]:
    model_info = get_model_config_from_model_path(model_path)
    raw_config = model_info.get("raw_config", {})
    generation_config = raw_config.get("generation_config", {})

    context_limit = int(model_info["context"])
    output_candidates = [
        raw_config.get("max_output_token_length"),
        raw_config.get("max_new_tokens"),
        raw_config.get("max_length"),
        generation_config.get("max_new_tokens"),
        generation_config.get("max_length"),
    ]
    positive_candidates = [int(v) for v in output_candidates if isinstance(v, int) and v > 0]
    output_limit = min(positive_candidates) if positive_candidates else requested_max_osl
    output_limit = min(output_limit, requested_max_osl)
    return context_limit, output_limit


def _resolve_supported_systems(system_aliases: list[str], backend: str) -> tuple[list[tuple[str, str, str]], list[dict[str, str]]]:
    supported = []
    skipped = []

    for alias in system_aliases:
        system_name = SYSTEM_ALIAS_TO_AIC.get(alias.lower())
        if system_name is None:
            skipped.append(
                {
                    "scope": "system",
                    "system_alias": alias,
                    "reason": "no AIC system definition in this repository",
                }
            )
            continue

        backend_version = get_latest_database_version(system_name, backend)
        if backend_version is None:
            skipped.append(
                {
                    "scope": "system",
                    "system_alias": alias,
                    "reason": f"no performance database found for backend={backend}",
                }
            )
            continue

        supported.append((alias, system_name, backend_version))

    return supported, skipped


def run_sweep(
    *,
    model_label: str,
    model_path: str,
    backend: str,
    database_mode: str,
    system_aliases: list[str],
    isl_list: list[int],
    osl_list: list[int],
    tp_list: list[int],
    batch_candidates: list[int],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    feasible_csv = output_dir / f"feasible_points_{model_label}.csv"
    skipped_csv = output_dir / f"skipped_cases_{model_label}.csv"

    context_limit, output_limit = _extract_model_limits(model_path, requested_max_osl=max(osl_list))
    supported_systems, skipped_records = _resolve_supported_systems(system_aliases, backend)

    feasible_rows: list[dict[str, object]] = []

    for system_alias, system_name, backend_version in supported_systems:
        for tp_size in tp_list:
            for isl in isl_list:
                for osl in osl_list:
                    if osl > output_limit:
                        skipped_records.append(
                            {
                                "scope": "shape",
                                "system_alias": system_alias,
                                "reason": f"osl={osl} exceeds model output limit={output_limit}",
                            }
                        )
                        continue

                    if isl + osl > context_limit:
                        skipped_records.append(
                            {
                                "scope": "shape",
                                "system_alias": system_alias,
                                "reason": f"isl+osl={isl + osl} exceeds model context limit={context_limit}",
                            }
                        )
                        continue

                    for batch_size in batch_candidates:
                        try:
                            result = cli_estimate(
                                model_path=model_path,
                                system_name=system_name,
                                mode="agg",
                                backend_name=backend,
                                backend_version=backend_version,
                                database_mode=database_mode,
                                isl=isl,
                                osl=osl,
                                batch_size=batch_size,
                                tp_size=tp_size,
                                pp_size=1,
                                attention_dp_size=1,
                            )
                        except RuntimeError as exc:
                            if "OOM" in str(exc):
                                break
                            skipped_records.append(
                                {
                                    "scope": "estimate",
                                    "system_alias": system_alias,
                                    "reason": f"tp={tp_size}, isl={isl}, osl={osl}, bs={batch_size}: {exc}",
                                }
                            )
                            break
                        except Exception as exc:  # pragma: no cover - defensive CLI wrapper
                            skipped_records.append(
                                {
                                    "scope": "estimate",
                                    "system_alias": system_alias,
                                    "reason": f"tp={tp_size}, isl={isl}, osl={osl}, bs={batch_size}: {exc}",
                                }
                            )
                            break

                        feasible_rows.append(
                            {
                                "model_label": model_label,
                                "model_path": model_path,
                                "system_alias": system_alias,
                                "system_name": system_name,
                                "backend": backend,
                                "backend_version": result.backend_version,
                                "database_mode": database_mode,
                                "tp": tp_size,
                                "pp": 1,
                                "dp": 1,
                                "prefix": 0,
                                "isl": isl,
                                "osl": osl,
                                "bs": batch_size,
                                "ttft_ms": result.ttft,
                                "tpot_ms": result.tpot,
                                "power_w": result.power_w,
                            }
                        )

    with feasible_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_path",
                "model_label",
                "system_alias",
                "system_name",
                "backend",
                "backend_version",
                "database_mode",
                "tp",
                "pp",
                "dp",
                "prefix",
                "isl",
                "osl",
                "bs",
                "ttft_ms",
                "tpot_ms",
                "power_w",
            ],
        )
        writer.writeheader()
        writer.writerows(feasible_rows)

    with skipped_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "system_alias",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(skipped_records)

    print(f"Wrote feasible points to: {feasible_csv}")
    print(f"Wrote skipped cases to:   {skipped_csv}")
    print(
        "Summary: "
        f"feasible_points={len(feasible_rows)}, skipped_cases={len(skipped_records)}, "
        f"context_limit={context_limit}, output_limit={output_limit}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep feasible TTFT/TPOT/BS points for prompt-enhancer style models.")
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS.keys()), default=DEFAULT_MODEL_PRESET)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--database-mode", default=DEFAULT_DATABASE_MODE)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEM_ALIASES))
    parser.add_argument("--isl", default=",".join(str(v) for v in DEFAULT_ISL_LIST))
    parser.add_argument("--osl", default=",".join(str(v) for v in DEFAULT_OSL_LIST))
    parser.add_argument("--tp", default=",".join(str(v) for v in DEFAULT_TP_LIST))
    parser.add_argument("--batch-candidates", default=",".join(str(v) for v in DEFAULT_BATCH_CANDIDATES))
    parser.add_argument("--output-dir", default="/home/heyang/outputs/aic2")
    args = parser.parse_args()
    model_path, model_label = _resolve_model_selection(args.model_preset, args.model_path)

    run_sweep(
        model_label=model_label,
        model_path=model_path,
        backend=args.backend,
        database_mode=args.database_mode,
        system_aliases=[item.strip() for item in args.systems.split(",") if item.strip()],
        isl_list=_parse_csv_ints(args.isl),
        osl_list=_parse_csv_ints(args.osl),
        tp_list=_parse_csv_ints(args.tp),
        batch_candidates=_parse_csv_ints(args.batch_candidates),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
