# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv

import pytest

from aiconfigurator.td_deployment_search.flip_simulation import (
    MODE_CAN_FLIP,
    TEMPLATE_DIT_ONLY,
    FlipSimulator,
    SimulationConfig,
    build_requests,
    config_from_args,
    load_dominant_intervals,
    run_suite,
    smoke_intervals,
)
from aiconfigurator.td_deployment_search.profiles import ProfileCatalog

pytestmark = pytest.mark.unit


def test_dominant_intervals_resolve_ties_to_previous(tmp_path):
    path = tmp_path / "intervals.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start_minute",
                "end_minute",
                "start_time",
                "end_time",
                "dominant",
                "duration_min",
                "num_minutes",
                "total_short_output",
                "total_long_output",
                "avg_short_ratio",
                "avg_long_ratio",
            ]
        )
        writer.writerow([0, 1, "", "", "Long", 1, 1, 1, 2, 0.3, 0.7])
        writer.writerow([1, 2, "", "", "Tie", 1, 1, 1, 1, 0.5, 0.5])
        writer.writerow([2, 3, "", "", "Short", 1, 1, 2, 1, 0.7, 0.3])

    intervals = load_dominant_intervals(path)

    assert [interval.resolved_dominant for interval in intervals] == ["Long", "Long", "Short"]
    assert [interval.output_tokens for interval in intervals] == [2048, 2048, 512]


def test_uniform_request_generation_uses_interval_at_arrival_time():
    intervals = smoke_intervals()

    requests = build_requests(intervals, rate_per_min=2, duration_min=3, input_tokens=128)

    assert [request.arrival_time_s for request in requests] == [0, 30, 60, 90, 120, 150]
    assert [request.output_tokens for request in requests] == [512, 512, 512, 512, 2048, 2048]


def test_pe_reprefill_lookup_uses_nearest_available_key_and_actual_remaining_decode():
    config = SimulationConfig(scenario="smoke", deployment_preset="smoke", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )

    matched_prefill, matched_remaining, ttft_s, tpot_s = sim.lookup_pe_reprefill("A800", 2, 228, 412)

    assert (matched_prefill, matched_remaining) == (256, 256)
    assert ttft_s == pytest.approx(22.974 / 1000.0)
    assert tpot_s == pytest.approx(6.58 / 1000.0)
    assert ttft_s + 412 * tpot_s == pytest.approx((22.974 + 412 * 6.58) / 1000.0)


def test_flip_source_selection_prefers_non_comb_instances():
    config = SimulationConfig(scenario="default", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=48,
        monitor_window_sec=30,
        mode=MODE_CAN_FLIP,
    )
    non_comb = sim.bundles["A800-NVLink-7/bundle-1"]
    non_comb.template_name = TEMPLATE_DIT_ONLY

    selected = sim.select_bundles_for_flip()

    assert selected == [non_comb]


def test_weighted_slo_baseline_uses_hardware_instance_counts():
    config = SimulationConfig(scenario="smoke", deployment_preset="smoke", flip_source_count=1)
    catalog = ProfileCatalog()
    sim = FlipSimulator(
        config=config,
        catalog=catalog,
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )

    baselines = sim.compute_slo_baselines()
    generator = catalog.generator_profile("wan2.2-ti2v-5b")
    pe = catalog.profile_data.pe_models["pe7b"].hardware["A800"]
    expected_pe_512 = (pe.ttft_ms[128][512][2] + 511 * pe.tpot_ms[128][512][2]) / 1000.0
    expected_dit = (4 * generator.dit["H100"].latency_s[8] + 19 * generator.dit["A800"].latency_s[8]) / 23
    expected_vae = (4 * generator.vae["H100"].latency_s[8] + 19 * generator.vae["A800"].latency_s[8]) / 23

    expected = expected_pe_512 + generator.encoder_cpu_latency_s + expected_dit + expected_vae
    assert baselines[512] == pytest.approx(expected)


def test_smoke_suite_emits_flip_and_reflip():
    config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        request_rates_per_min=(12.0,),
        monitor_windows_sec=(10.0,),
        duration_min=6.0,
        flip_source_count=1,
    )

    results, _intervals = run_suite(config)
    can_flip = next(result for result in results if result.mode == MODE_CAN_FLIP)

    assert can_flip.summary["finished_requests"] == 72
    assert [event.direction for event in can_flip.flip_events] == ["short_to_long", "long_to_short"]
    assert all(event.cold_start_done_time_s is not None for event in can_flip.flip_events)


def test_parser_smoke_defaults_are_small():
    parser_config = config_from_args(
        type(
            "Args",
            (),
            {
                "dominant_intervals_csv": SimulationConfig.dominant_intervals_csv,
                "out_dir": SimulationConfig.out_dir,
                "request_rates_per_min": None,
                "monitor_windows_sec": None,
                "duration_min": None,
                "token_threshold": 1280.0,
                "seed": 0,
                "mode": "both",
                "pe_model": "pe7b",
                "generator_model": "wan2.2-ti2v-5b",
                "input_tokens": 128,
                "short_output_tokens": 512,
                "long_output_tokens": 2048,
                "denoising_steps": 50,
                "scenario": "smoke",
                "deployment_preset": None,
                "flip_plan_preset": "wan22_group1_restricted",
                "flip_source_count": None,
                "trace_start": "2024-10-15T12:00:00+00:00",
                "verbose": False,
            },
        )()
    )

    assert parser_config.request_rates_per_min == (12.0,)
    assert parser_config.monitor_windows_sec == (10.0,)
    assert parser_config.duration_min == 6.0
    assert parser_config.flip_source_count == 1
