# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from aiconfigurator.td_deployment_search.models import (
    SEARCH_KIND_FULL_ENCODER_CPU,
    SEARCH_KIND_FULL_ENCODER_GPU,
    SEARCH_KIND_PRUNED,
    ModelSelection,
    NodeSpec,
    ResourceGroupSpec,
)
from aiconfigurator.td_deployment_search.planner import (
    build_default_resource_groups,
    bundle_partitions,
    enumerate_node_configs,
    solve_group,
)
from aiconfigurator.td_deployment_search.profiles import DEFAULT_PROFILE_DATA_PATH, ProfileCatalog
from aiconfigurator.td_deployment_search.reports import (
    _base_utilization,
    _fragment_completion,
    _render_flip_plan,
    render_main_report,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def catalog() -> ProfileCatalog:
    return ProfileCatalog()


@pytest.fixture(scope="module")
def wan22_group1_results(catalog: ProfileCatalog):
    group1, _group2 = build_default_resource_groups()
    return {
        output_tokens: solve_group(
            catalog,
            group1,
            ModelSelection(generator_model="wan2.2-ti2v-5b", output_tokens=output_tokens),
            top_k=1,
            mip_time_limit_s=120,
            search_kind=SEARCH_KIND_PRUNED,
            solver_backend="matrix",
        )[0]
        for output_tokens in (512, 2048)
    }


def test_profile_loading_and_unit_conversions(catalog: ProfileCatalog):
    model = ModelSelection(output_tokens=512)
    profile = catalog.pe_stage_profile(model, "A800", 2)
    assert profile.latency_s == pytest.approx((30.38936 + 511 * 6.867308) / 1000.0)
    assert profile.memory_gb > 0
    assert profile.throughput == pytest.approx(1.0 / profile.latency_s)
    assert catalog.encoder_cpu_latency_s() == pytest.approx(6.769752)


def test_memory_legality_uses_bundle_total_hbm(catalog: ProfileCatalog):
    node = NodeSpec(name="A800-NVLink-1", gpu_type="A800", gpu_count=8, gpu_memory_gb=80)
    model = ModelSelection(output_tokens=512)
    configs = enumerate_node_configs(catalog, model, node)
    assert configs
    for config in configs:
        for assignment in config.active_assignments:
            assert assignment.template is not None
            assert assignment.template.memory_gb <= assignment.bundle.size * node.gpu_memory_gb


def test_bundle_partition_enumeration():
    partitions = bundle_partitions(8)
    assert (8,) in partitions
    assert (4, 4) in partitions
    assert (2, 2, 2, 2) in partitions
    assert (1, 1, 1, 1, 1, 1, 1, 1) in partitions
    assert len(partitions) == 10


def test_small_milp_smoke(catalog: ProfileCatalog):
    group = ResourceGroupSpec(
        name="Tiny",
        nodes=(
            NodeSpec(name="H100-NVLink-1", gpu_type="H100", gpu_count=8, gpu_memory_gb=80),
            NodeSpec(name="A800-NVLink-1", gpu_type="A800", gpu_count=8, gpu_memory_gb=80),
        ),
    )
    results = solve_group(catalog, group, ModelSelection(output_tokens=512), top_k=2, mip_time_limit_s=30)
    assert results
    assert results[0].throughput > 0
    assert results[0].encoder_cpu_instances >= 1


def test_profile_registry_loads_all_generators(catalog: ProfileCatalog):
    for generator in ("wan2.2-ti2v-5b", "wan2.1-t2v-1.3b", "z-image"):
        profile = catalog.generator_profile(generator)
        assert profile.encoder_cpu_latency_s > 0
        assert profile.main_dit_bundle_size in {2, 4, 8}


def test_full_template_counts(catalog: ProfileCatalog):
    model = ModelSelection(generator_model="wan2.2-ti2v-5b", output_tokens=512)
    assert len(catalog.template_options(model, "H100", 8, search_kind=SEARCH_KIND_FULL_ENCODER_CPU)) == 7
    assert len(catalog.template_options(model, "H100", 8, search_kind=SEARCH_KIND_FULL_ENCODER_GPU)) == 15


def test_matrix_and_enumeration_match_bottleneck(catalog: ProfileCatalog):
    group = ResourceGroupSpec(
        name="Tiny",
        nodes=(
            NodeSpec(name="H100-NVLink-1", gpu_type="H100", gpu_count=8, gpu_memory_gb=80),
            NodeSpec(name="A800-NVLink-1", gpu_type="A800", gpu_count=8, gpu_memory_gb=80),
        ),
    )
    model = ModelSelection(generator_model="wan2.2-ti2v-5b", output_tokens=512)
    matrix = solve_group(catalog, group, model, top_k=1, mip_time_limit_s=30, search_kind=SEARCH_KIND_PRUNED)
    enumeration = solve_group(
        catalog,
        group,
        model,
        top_k=1,
        mip_time_limit_s=30,
        search_kind=SEARCH_KIND_PRUNED,
        solver_backend="enumeration",
    )
    assert matrix[0].throughput == pytest.approx(enumeration[0].throughput)
    assert matrix[0].bottleneck_stage == enumeration[0].bottleneck_stage


def test_report_paths_are_txt():
    assert Path(r"C:\aiconfigurator\td_main_search_report.txt").suffix == ".txt"
    assert Path(r"C:\aiconfigurator\td_lcm_report.txt").suffix == ".txt"


def test_fragment_completion_fills_group1_wan22_a800_fragment(catalog: ProfileCatalog, wan22_group1_results):
    result = wan22_group1_results[512]
    base_utilization = _base_utilization(result)
    completed = _fragment_completion(catalog, result, "same_model")
    assert base_utilization["A800"] == (62, 64)
    assert completed.utilization["A800"] == (64, 64)
    assert completed.counter[("PE_Only", 2, "A800")] == 4


def test_restricted_flip_uses_whole_bundle_and_matches_fragment_filled_target(catalog: ProfileCatalog, wan22_group1_results):
    lines = _render_flip_plan(catalog, wan22_group1_results[512], wan22_group1_results[2048])
    text = "\n".join(lines)
    assert "Flip 2x DiT_VAE_Comb_A800_P8 -> 8x PE_Only_A800_P2." in text
    assert "gap=0.0000 req/s" in text


def test_report_contains_throughput_arithmetic(catalog: ProfileCatalog, wan22_group1_results, tmp_path):
    grouped = {
        "Group 1": {output_tokens: [result] for output_tokens, result in wan22_group1_results.items()},
        "Group 2": {output_tokens: [result] for output_tokens, result in wan22_group1_results.items()},
    }
    text = render_main_report(catalog, grouped, tmp_path / "report.txt")
    assert "Throughput arithmetic:" in text
    assert "latency=(29.94049+(2048-1)*6.880903)/1000=14.1151s" in text
    assert "overall=min(...)" in text


def test_package_profile_registry_matches_legacy_data_for_planner_fields(catalog: ProfileCatalog):
    if not DEFAULT_PROFILE_DATA_PATH.exists():
        pytest.skip("legacy profile data.py is not available on this machine")
    spec = importlib.util.spec_from_file_location("legacy_td_profile_data", DEFAULT_PROFILE_DATA_PATH)
    assert spec is not None and spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    pe7b = catalog.profile_data.pe_models["pe7b"]
    assert pe7b.hardware["H100"].ttft_ms == legacy.pe7b_h100_nvlink_simu_ttft_ms
    assert pe7b.hardware["H100"].tpot_ms == legacy.pe7b_h100_nvlink_simu_tpot_ms
    assert pe7b.hardware["A800"].ttft_ms == legacy.pe7b_a800_nvlink_ttft_ms
    assert pe7b.hardware["A800"].tpot_ms == legacy.pe7b_a800_nvlink_tpot_ms

    for generator_name, prefix in (
        ("wan2.2-ti2v-5b", "wan22_ti2v_5b"),
        ("wan2.1-t2v-1.3b", "wan21_t2v_1_3b"),
        ("z-image", "z_image"),
    ):
        generator = catalog.generator_profile(generator_name)
        assert generator.encoder_cpu_latency_s == pytest.approx(legacy.wan22_ti2v_5b_encoder_cpu_ms / 1000.0)
        for attr, legacy_stage in (("dit", "denoiser"), ("vae", "decoder")):
            stage = getattr(generator, attr)
            assert stage["A800"].latency_s == getattr(legacy, f"{prefix}_{legacy_stage}_a800_nvlink_duration_s")
            expected_h100 = {
                parallelism: latency_ms / 1000.0
                for parallelism, latency_ms in getattr(legacy, f"{prefix}_{legacy_stage}_h200_nvlink_duration_ms").items()
            }
            assert stage["H100"].latency_s == expected_h100
            assert stage["A800"].memory.components_gb == getattr(
                legacy,
                f"{prefix}_{legacy_stage}_a800_nvlink_memory_overhead_gb",
            )
