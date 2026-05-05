# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from aiconfigurator.td_deployment_search.lcm import build_lcm_results
from aiconfigurator.td_deployment_search.models import (
    SEARCH_KIND_FULL_ENCODER_CPU,
    SEARCH_KIND_FULL_ENCODER_GPU,
    SEARCH_KIND_PRUNED,
    ModelSelection,
)
from aiconfigurator.td_deployment_search.planner import build_default_resource_groups, solve_group
from aiconfigurator.td_deployment_search.profiles import DEFAULT_PROFILE_DATA_PATH, ProfileCatalog, model_slug
from aiconfigurator.td_deployment_search.reports import (
    LCM_REPORT_PATH,
    MAIN_REPORT_PATH,
    render_full_report,
    render_lcm_report,
    render_main_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TD profiler-backed deployment search planner.")
    parser.add_argument("--legacy-profile-data", type=Path, default=None)
    parser.add_argument("--pe-model", default="pe7b")
    parser.add_argument(
        "--generator-model",
        action="append",
        default=None,
        help="Generator model to run. May be passed multiple times. Default runs wan2.2, wan2.1, and z-image.",
    )
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\aiconfigurator\td_outputs"))
    parser.add_argument("--mip-time-limit-s", type=float, default=120.0)
    parser.add_argument("--skip-backend-compare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = ProfileCatalog(args.legacy_profile_data, use_legacy_data=args.legacy_profile_data is not None)
    groups = build_default_resource_groups()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator_models = args.generator_model or ["wan2.2-ti2v-5b", "wan2.1-t2v-1.3b", "z-image"]

    for generator_model in generator_models:
        if not args.skip_backend_compare:
            _compare_matrix_and_enumeration(catalog, groups[0], args.pe_model, generator_model, args.input_tokens, args.mip_time_limit_s)

        grouped_results = _solve_report_set(
            catalog,
            groups,
            args.pe_model,
            generator_model,
            args.input_tokens,
            args.top_k,
            args.mip_time_limit_s,
            SEARCH_KIND_PRUNED,
        )
        full_grouped_results = {
            group.name: {
                "Encoder_CPU": _solve_group_outputs(
                    catalog,
                    group,
                    args.pe_model,
                    generator_model,
                    args.input_tokens,
                    args.top_k,
                    args.mip_time_limit_s,
                    SEARCH_KIND_FULL_ENCODER_CPU,
                ),
                "Encoder_GPU": _solve_group_outputs(
                    catalog,
                    group,
                    args.pe_model,
                    generator_model,
                    args.input_tokens,
                    args.top_k,
                    args.mip_time_limit_s,
                    SEARCH_KIND_FULL_ENCODER_GPU,
                ),
            }
            for group in groups
        }

        slug = model_slug(generator_model)
        main_path = args.output_dir / f"{slug}_main_report.txt"
        full_path = args.output_dir / f"{slug}_full_report.txt"
        lcm_path = args.output_dir / f"{slug}_lcm_report.txt"
        base_model = ModelSelection(args.pe_model, generator_model, args.input_tokens, 512)
        render_main_report(catalog, grouped_results, main_path)
        render_full_report(catalog, full_grouped_results, full_path)
        render_lcm_report(catalog, base_model, build_lcm_results(catalog, base_model), lcm_path)
        print(f"Wrote {main_path}")
        print(f"Wrote {full_path}")
        print(f"Wrote {lcm_path}")

        if slug == "wan22_ti2v_5b":
            shutil.copyfile(main_path, MAIN_REPORT_PATH)
            shutil.copyfile(lcm_path, LCM_REPORT_PATH)
            print(f"Wrote compatibility copy {MAIN_REPORT_PATH}")
            print(f"Wrote compatibility copy {LCM_REPORT_PATH}")


def _solve_report_set(catalog, groups, pe_model, generator_model, input_tokens, top_k, mip_time_limit_s, search_kind):
    return {
        group.name: _solve_group_outputs(catalog, group, pe_model, generator_model, input_tokens, top_k, mip_time_limit_s, search_kind)
        for group in groups
    }


def _solve_group_outputs(catalog, group, pe_model, generator_model, input_tokens, top_k, mip_time_limit_s, search_kind):
    return {
        output_tokens: solve_group(
            catalog=catalog,
            group=group,
            model=ModelSelection(pe_model=pe_model, generator_model=generator_model, input_tokens=input_tokens, output_tokens=output_tokens),
            top_k=top_k,
            mip_time_limit_s=mip_time_limit_s,
            search_kind=search_kind,
            solver_backend="matrix",
        )
        for output_tokens in (512, 2048)
    }


def _compare_matrix_and_enumeration(catalog, group, pe_model, generator_model, input_tokens, mip_time_limit_s):
    model = ModelSelection(pe_model=pe_model, generator_model=generator_model, input_tokens=input_tokens, output_tokens=512)
    matrix = solve_group(catalog, group, model, top_k=1, mip_time_limit_s=mip_time_limit_s, search_kind=SEARCH_KIND_PRUNED, solver_backend="matrix")
    enumeration = solve_group(catalog, group, model, top_k=1, mip_time_limit_s=mip_time_limit_s, search_kind=SEARCH_KIND_PRUNED, solver_backend="enumeration")
    if not matrix or not enumeration:
        raise RuntimeError(f"Backend comparison failed for {generator_model}: one backend returned no result.")
    if abs(matrix[0].throughput - enumeration[0].throughput) > 1e-6:
        raise RuntimeError(
            f"Backend mismatch for {generator_model}: matrix={matrix[0].throughput:.8f}, "
            f"enumeration={enumeration[0].throughput:.8f}"
        )
    if matrix[0].bottleneck_stage != enumeration[0].bottleneck_stage:
        raise RuntimeError(
            f"Backend bottleneck mismatch for {generator_model}: "
            f"matrix={matrix[0].bottleneck_stage}, enumeration={enumeration[0].bottleneck_stage}"
        )
    print(
        f"Backend comparison ok for {generator_model}: "
        f"throughput={matrix[0].throughput:.4f} req/s, bottleneck={matrix[0].bottleneck_stage}"
    )


if __name__ == "__main__":
    main()
