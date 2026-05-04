#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run fixed-bs TTFT/TPOT estimates for the promptenhancer-7b cases."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_MPLCONFIG_BASE = Path.cwd() / "test_results" / ".matplotlib-cache"
_MPLCONFIG_BASE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_", dir=_MPLCONFIG_BASE))
logging.getLogger("matplotlib").setLevel(logging.ERROR)

from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.sdk.perf_database import get_latest_database_version


MODEL_LABEL = "promptenhancer-7b"
DEFAULT_BACKEND = "sglang"
DEFAULT_DATABASE_MODE = "HYBRID"
DEFAULT_TP_LIST = [1, 2, 4, 8]
DEFAULT_SYSTEMS = ["h100_sxm", "a100_sxm"]

# Minimal config from https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt.
# A local config keeps this sweep reproducible in offline/sandboxed environments.
DEFAULT_REPROMPT_CONFIG: dict[str, Any] = {
    "add_classification_head": False,
    "architectures": ["HunYuanDenseV1ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.1,
    "attention_head_dim": 128,
    "bos_token_id": 1,
    "cla_share_factor": 2,
    "class_num": 0,
    "dense_list": [4096, 0],
    "eos_token_id": 127960,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "im_end_id": 5,
    "im_newline_id": 11,
    "im_start_id": 4,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "mask_init_id": 12,
    "max_position_embeddings": 32768,
    "mlp_bias": False,
    "model_type": "hunyuan_v1_dense",
    "norm_type": "rms",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "org_vocab_size": 128167,
    "pad_id": 127961,
    "pad_token_id": 127961,
    "pool_type": "last",
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "alpha": 1000.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "type": "dynamic",
    },
    "rope_theta": 10000.0,
    "sep_token_id": 127962,
    "text_end_id": 7,
    "text_start_id": 6,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.55.0.dev0",
    "use_cache": False,
    "use_cla": False,
    "use_qk_norm": True,
    "use_rotary_pos_emb": True,
    "vocab_size": 128167,
}


@dataclass(frozen=True, order=True)
class TestPoint:
    system: str
    tp: int
    isl: int
    osl: int


def _case1_shapes() -> list[tuple[int, int]]:
    return [(128, 512), (128, 2048)]


def _case2_shapes() -> list[tuple[int, int]]:
    paired_shapes = [(isl, 2176 - isl) for isl in range(128, 2176, 128)]
    extra_shapes = [(128, 512), (256, 256), (384, 128)]
    return paired_shapes + extra_shapes


def _case_points() -> dict[str, list[TestPoint]]:
    return {
        "case1": [
            TestPoint(system="h100_sxm", tp=tp, isl=isl, osl=osl)
            for isl, osl in _case1_shapes()
            for tp in DEFAULT_TP_LIST
        ],
        "case2": [
            TestPoint(system=system, tp=tp, isl=isl, osl=osl)
            for system in DEFAULT_SYSTEMS
            for isl, osl in _case2_shapes()
            for tp in DEFAULT_TP_LIST
        ],
    }


def _unique_points(points_by_case: dict[str, list[TestPoint]]) -> list[TestPoint]:
    return sorted({point for points in points_by_case.values() for point in points})


def _write_default_model_config(output_dir: Path) -> Path:
    model_dir = output_dir / "promptenhancer_7b_reprompt_config"
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_REPROMPT_CONFIG, f, indent=2, sort_keys=True)
        f.write("\n")
    return model_dir


def _check_invariants(raw: dict[str, Any]) -> list[str]:
    expected = {
        "bs": 1,
        "global_bs": 1,
        "concurrency": 1,
        "dp": 1,
    }
    errors = []
    for key, expected_value in expected.items():
        actual_value = raw.get(key)
        if int(actual_value) != expected_value:
            errors.append(f"{key}={actual_value}, expected {expected_value}")
    return errors


def _estimate_point(
    *,
    point: TestPoint,
    model_path: str,
    backend: str,
    database_mode: str,
) -> dict[str, Any]:
    backend_version = get_latest_database_version(point.system, backend)
    if backend_version is None:
        return {
            "model": MODEL_LABEL,
            "system": point.system,
            "backend": backend,
            "backend_version": "",
            "database_mode": database_mode,
            "tp": point.tp,
            "isl": point.isl,
            "osl": point.osl,
            "bs": 1,
            "concurrency": 1,
            "global_bs": 1,
            "ttft_ms": "",
            "tpot_ms": "",
            "status": "ERROR",
            "error": f"no performance database found for backend={backend}",
        }

    try:
        result = cli_estimate(
            model_path=model_path,
            system_name=point.system,
            mode="agg",
            backend_name=backend,
            backend_version=backend_version,
            database_mode=database_mode,
            isl=point.isl,
            osl=point.osl,
            batch_size=1,
            ctx_tokens=point.isl,
            tp_size=point.tp,
            pp_size=1,
            attention_dp_size=1,
        )
        raw = result.raw
        invariant_errors = _check_invariants(raw)
        status = "OK" if not invariant_errors else "ERROR"
        error = "; ".join(invariant_errors)
        return {
            "model": MODEL_LABEL,
            "system": point.system,
            "backend": backend,
            "backend_version": result.backend_version,
            "database_mode": database_mode,
            "tp": point.tp,
            "isl": point.isl,
            "osl": point.osl,
            "bs": int(raw.get("bs", 1)),
            "concurrency": int(raw.get("concurrency", 1)),
            "global_bs": int(raw.get("global_bs", 1)),
            "ttft_ms": result.ttft,
            "tpot_ms": result.tpot,
            "status": status,
            "error": error,
        }
    except Exception as exc:  # pragma: no cover - execution diagnostics
        return {
            "model": MODEL_LABEL,
            "system": point.system,
            "backend": backend,
            "backend_version": backend_version,
            "database_mode": database_mode,
            "tp": point.tp,
            "isl": point.isl,
            "osl": point.osl,
            "bs": 1,
            "concurrency": 1,
            "global_bs": 1,
            "ttft_ms": "",
            "tpot_ms": "",
            "status": "ERROR",
            "error": str(exc),
        }


def _format_ms(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.3f}"


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ordered_unique_shapes(points: Iterable[TestPoint]) -> list[tuple[int, int]]:
    shapes = []
    seen = set()
    for point in points:
        shape = (point.isl, point.osl)
        if shape in seen:
            continue
        shapes.append(shape)
        seen.add(shape)
    return shapes


def _case_rows(
    *,
    points_by_case: dict[str, list[TestPoint]],
    results_by_point: dict[TestPoint, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case_name, points in points_by_case.items():
        for point in points:
            result = results_by_point[point]
            rows.append(
                {
                    "case": case_name,
                    "model": result["model"],
                    "system": point.system,
                    "tp": point.tp,
                    "isl": point.isl,
                    "osl": point.osl,
                    "ttft_ms": _format_ms(result["ttft_ms"]),
                    "tpot_ms": _format_ms(result["tpot_ms"]),
                    "status": result["status"],
                    "error": result["error"],
                }
            )
    return rows


def _write_txt_summary(
    *,
    path: Path,
    points_by_case: dict[str, list[TestPoint]],
    results_by_point: dict[TestPoint, dict[str, Any]],
    unique_count: int,
    logical_count: int,
    output_dir: Path,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"model: {MODEL_LABEL}\n")
        f.write(f"unique executed test points: {unique_count}\n")
        f.write(f"logical case result rows: {logical_count}\n")
        f.write(f"output_dir: {output_dir}\n\n")

        for case_name, points in points_by_case.items():
            systems = sorted({point.system for point in points})
            shapes = _ordered_unique_shapes(points)
            f.write(f"{case_name} setup\n")
            f.write(f"  systems: {', '.join(systems)}\n")
            f.write(f"  tp: {', '.join(str(tp) for tp in DEFAULT_TP_LIST)}\n")
            f.write("  bs: 1, concurrency: 1, global_bs: 1\n")
            f.write(f"  shapes: {', '.join(f'({isl},{osl})' for isl, osl in shapes)}\n")
            f.write(f"{case_name} test summary\n")
            for point in points:
                result = results_by_point[point]
                prefix = f"  system={point.system}, tp={point.tp}, isl={point.isl}, osl={point.osl}"
                if result["status"] == "OK":
                    f.write(
                        f"{prefix} -> TTFT={_format_ms(result['ttft_ms'])} ms, "
                        f"TPOT={_format_ms(result['tpot_ms'])} ms\n"
                    )
                else:
                    f.write(f"{prefix} -> ERROR: {result['error']}\n")
            f.write("\n")


def run_cases(
    *,
    model_path: str | None,
    output_dir: Path,
    backend: str,
    database_mode: str,
    limit: int | None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_path = model_path or str(_write_default_model_config(output_dir))

    points_by_case = _case_points()
    unique_points = _unique_points(points_by_case)
    if limit is not None:
        unique_points = unique_points[:limit]
        points_by_case = {
            case_name: [point for point in points if point in set(unique_points)]
            for case_name, points in points_by_case.items()
        }

    results_by_point: dict[TestPoint, dict[str, Any]] = {}
    for index, point in enumerate(unique_points, start=1):
        print(
            f"[{index}/{len(unique_points)}] system={point.system}, tp={point.tp}, "
            f"isl={point.isl}, osl={point.osl}",
            flush=True,
        )
        results_by_point[point] = _estimate_point(
            point=point,
            model_path=resolved_model_path,
            backend=backend,
            database_mode=database_mode,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_csv = output_dir / f"{MODEL_LABEL}_unique_results_{timestamp}.csv"
    case_csv = output_dir / f"{MODEL_LABEL}_case_results_{timestamp}.csv"
    summary_txt = output_dir / f"{MODEL_LABEL}_case_results_{timestamp}.txt"

    unique_rows = [
        {
            **result,
            "ttft_ms": _format_ms(result["ttft_ms"]),
            "tpot_ms": _format_ms(result["tpot_ms"]),
        }
        for _, result in sorted(results_by_point.items())
    ]
    case_rows = _case_rows(points_by_case=points_by_case, results_by_point=results_by_point)

    _write_csv(
        unique_csv,
        [
            "model",
            "system",
            "backend",
            "backend_version",
            "database_mode",
            "tp",
            "isl",
            "osl",
            "bs",
            "concurrency",
            "global_bs",
            "ttft_ms",
            "tpot_ms",
            "status",
            "error",
        ],
        unique_rows,
    )
    _write_csv(
        case_csv,
        ["case", "model", "system", "tp", "isl", "osl", "ttft_ms", "tpot_ms", "status", "error"],
        case_rows,
    )
    _write_txt_summary(
        path=summary_txt,
        points_by_case=points_by_case,
        results_by_point=results_by_point,
        unique_count=len(unique_points),
        logical_count=sum(len(points) for points in points_by_case.values()),
        output_dir=output_dir,
    )
    return unique_csv, case_csv, summary_txt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run promptenhancer-7b TTFT/TPOT case estimates.")
    parser.add_argument("--model-path", default=None, help="Optional local model directory containing config.json.")
    parser.add_argument("--output-dir", default="test_results", help="Directory for CSV/TXT outputs.")
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--database-mode", default=DEFAULT_DATABASE_MODE)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick validation.")
    args = parser.parse_args()

    unique_csv, case_csv, summary_txt = run_cases(
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
        backend=args.backend,
        database_mode=args.database_mode,
        limit=args.limit,
    )
    print(f"Wrote unique results: {unique_csv}")
    print(f"Wrote case results:   {case_csv}")
    print(f"Wrote text summary:   {summary_txt}")


if __name__ == "__main__":
    main()
