# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv

import pytest

from aiconfigurator.td_deployment_search.flip_simulation import (
    COLD_START_MODE_ZERO,
    GENERATOR_INIT_TIME_MODE_OPTIMIZED,
    GENERATOR_INIT_TIME_MODE_PROFILE,
    MODE_CAN_FLIP,
    PE_INIT_TIME_MODE_NON_OPTIMIZED,
    PE_INIT_TIME_MODE_OPTIMIZED,
    PE_OUTPUT_ESTIMATE_MODE_ORACLE,
    STAGE_DIT,
    STAGE_VAE,
    TEMPLATE_DIT_ONLY,
    FlipSimulator,
    SimulationConfig,
    StageRun,
    add_cold_start_optimization_fields,
    add_comparison_fields,
    build_requests,
    config_from_args,
    load_dominant_intervals,
    margin_label,
    render_margin_comparison_markdown,
    render_cold_start_optimization_markdown,
    render_non_margin_summary_from_run_summary_csv,
    run_suite,
    smoke_intervals,
)
from aiconfigurator.td_deployment_search.profiles import ProfileCatalog

pytestmark = pytest.mark.unit


def restrict_a800_flip_candidates(sim: FlipSimulator, keep_keys: set[str]) -> None:
    for key, bundle in sim.bundles.items():
        if bundle.gpu_type == "A800" and key not in keep_keys:
            bundle.active = False


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

    assert list(selected.bundles) == [non_comb]


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


def test_hysteresis_margin_has_deadband():
    config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        flip_source_count=1,
        threshold_margin_ratio=0.10,
    )
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )

    assert sim.threshold_bounds() == pytest.approx((1280.0, 1126.4, 1433.6))
    assert sim.monitor_decision(1280.0) == "deadband_keep_current"
    assert sim.monitor_decision(1434.0) == "short_to_long"
    sim.current_target = "long"
    assert sim.monitor_decision(1200.0) == "deadband_keep_current"
    assert sim.monitor_decision(1126.0) == "long_to_short"


def test_zero_margin_preserves_single_threshold_decision():
    config = SimulationConfig(scenario="smoke", deployment_preset="smoke", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )

    assert sim.monitor_decision(1281.0) == "short_to_long"
    assert sim.monitor_decision(1280.0) == "long_to_short"


def test_time_based_dispatch_prefers_faster_hardware_when_queue_work_ties():
    config = SimulationConfig(scenario="default", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=48,
        monitor_window_sec=30,
        mode=MODE_CAN_FLIP,
    )
    sim.requests = build_requests(smoke_intervals(), rate_per_min=3, duration_min=1, input_tokens=128)
    h100 = next(inst for inst in sim.stage_instances[STAGE_DIT] if inst.gpu_type == "H100")
    a800 = next(inst for inst in sim.stage_instances[STAGE_DIT] if inst.gpu_type == "A800")
    h100.queue.append((0, 0))
    a800.queue.append((1, 0))

    assert sim.choose_least_waiting(STAGE_DIT, [a800, h100]) is h100


def test_estimated_load_updates_per_request_for_migration_style_assignment():
    config = SimulationConfig(scenario="default", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )
    targets = sim.stage_instances["PE"][:2]
    requests = build_requests(smoke_intervals(), rate_per_min=2, duration_min=1, input_tokens=128)[:2]
    sim.requests = requests
    estimated = sim.estimated_work_scores(targets, 0.0)

    first = sim.choose_least_waiting("PE", targets, estimated, incoming_req=requests[0])
    estimated[first.instance_id] += sim.incoming_load_increment(targets, first, requests[0])
    second = sim.choose_least_waiting("PE", targets, estimated, incoming_req=requests[1])

    assert first is not second


def test_flip_selection_prefers_shorter_running_boundary_delay():
    config = SimulationConfig(scenario="default", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=48,
        monitor_window_sec=30,
        mode=MODE_CAN_FLIP,
    )
    sim.requests = build_requests(smoke_intervals(), rate_per_min=4, duration_min=1, input_tokens=128)
    fast_bundle = sim.bundles["A800-NVLink-1/bundle-1"]
    slow_bundle = sim.bundles["A800-NVLink-2/bundle-1"]
    restrict_a800_flip_candidates(sim, {fast_bundle.bundle_key, slow_bundle.bundle_key})
    fast_dit = fast_bundle.stage_instances[STAGE_DIT]
    slow_dit = slow_bundle.stage_instances[STAGE_DIT]
    fast_dit.current_req_id = 0
    fast_dit.current_start_time_s = 0.0
    fast_dit.current_run = StageRun(total_duration_s=100.0, dit_step_s=10.0)
    slow_dit.current_req_id = 1
    slow_dit.current_start_time_s = 5.0
    slow_dit.current_run = StageRun(total_duration_s=100.0, dit_step_s=10.0)

    selected = sim.select_bundles_for_flip(now_s=10.0)

    assert selected.bundles[0] is fast_bundle
    assert selected.score.restart_ready_delay_s == pytest.approx(sim.pe_init_time_s("A800", 2))


def test_flip_score_uses_complement_work_not_raw_count():
    config = SimulationConfig(scenario="default", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=48,
        monitor_window_sec=30,
        mode=MODE_CAN_FLIP,
    )
    sim.requests = build_requests(smoke_intervals(), rate_per_min=4, duration_min=1, input_tokens=128)
    dit_bundle = sim.bundles["A800-NVLink-1/bundle-1"]
    vae_bundle = sim.bundles["A800-NVLink-2/bundle-1"]
    restrict_a800_flip_candidates(sim, {dit_bundle.bundle_key, vae_bundle.bundle_key})
    dit_bundle.stage_instances[STAGE_DIT].queue.append((0, 0))
    vae_bundle.stage_instances[STAGE_VAE].queue.append((1, 0))

    dit_score = sim.score_flip_bundle_combination([dit_bundle], "short_to_long", 0.0)
    vae_score = sim.score_flip_bundle_combination([vae_bundle], "short_to_long", 0.0)

    assert dit_score.migrated_request_count == vae_score.migrated_request_count == 1
    assert vae_score.complement_added_work_s < dit_score.complement_added_work_s


def test_comb_bundle_selection_score_takes_max_of_dit_and_vae_work():
    config = SimulationConfig(scenario="default", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=48,
        monitor_window_sec=30,
        mode=MODE_CAN_FLIP,
    )
    sim.requests = build_requests(smoke_intervals(), rate_per_min=4, duration_min=1, input_tokens=128)
    bundle = sim.bundles["A800-NVLink-1/bundle-1"]
    dit = bundle.stage_instances[STAGE_DIT]
    vae = bundle.stage_instances[STAGE_VAE]
    dit.queue.append((0, 0))
    vae.queue.append((1, 0))

    score = sim.score_flip_bundle_combination([bundle], "short_to_long", 0.0)

    assert score.migrated_request_count == 2
    dit_score = sim.stage_work_units_to_seconds(STAGE_DIT, 1.0, {dit.instance_id, vae.instance_id}, 0.0)
    vae_score = sim.stage_work_units_to_seconds(STAGE_VAE, 1.0, {dit.instance_id, vae.instance_id}, 0.0)
    assert score.complement_added_work_s == pytest.approx(max(dit_score, vae_score))


def test_multi_source_flip_selection_uses_cutoff_tie_complement_score():
    config = SimulationConfig(scenario="default", flip_source_count=2)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=48,
        monitor_window_sec=30,
        mode=MODE_CAN_FLIP,
    )
    sim.requests = build_requests(smoke_intervals(), rate_per_min=10, duration_min=1, input_tokens=128)
    heavy_a = sim.bundles["A800-NVLink-1/bundle-1"]
    heavy_b = sim.bundles["A800-NVLink-2/bundle-1"]
    empty = sim.bundles["A800-NVLink-3/bundle-1"]
    restrict_a800_flip_candidates(sim, {heavy_a.bundle_key, heavy_b.bundle_key, empty.bundle_key})
    for idx in range(4):
        heavy_a.stage_instances[STAGE_DIT].queue.append((idx, 0))
        heavy_b.stage_instances[STAGE_DIT].queue.append((idx + 4, 0))

    selected = sim.select_bundles_for_flip(now_s=0.0)

    assert empty in selected.bundles
    assert not (heavy_a in selected.bundles and heavy_b in selected.bundles)


def test_pe_instance_monitor_classifies_bins_without_margin():
    config = SimulationConfig(scenario="smoke", deployment_preset="smoke", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )
    inst = sim.stage_instances["PE"][0]
    sim.pe_output_log.append((9.0, 512, 0, inst.instance_id))

    assert sim.pe_estimated_output_bin(inst, 10.0) == 512
    sim.pe_output_log.append((10.0, 2048, 1, inst.instance_id))
    assert sim.pe_estimated_output_bin(inst, 10.0) == 2048


def test_pe_instance_monitor_window_excludes_expired_boundary_samples():
    config = SimulationConfig(scenario="smoke", deployment_preset="smoke", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )
    inst = sim.stage_instances["PE"][0]
    sim.pe_output_log.append((0.0, 2048, 0, inst.instance_id))
    sim.pe_output_log.append((1.0, 512, 1, inst.instance_id))
    sim.pe_output_log.append((10.0, 512, 2, inst.instance_id))

    assert sim.pe_monitor_avg_output_tokens(10.0, inst.instance_id) == 512
    assert sim.pe_instance_tokens_per_s(inst, 10.0) == pytest.approx(1024 / 10.0)


def test_pe_waiting_estimate_uses_monitor_bin_not_true_output_tokens():
    config = SimulationConfig(scenario="smoke", deployment_preset="smoke", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )
    inst = sim.stage_instances["PE"][0]
    req = build_requests(smoke_intervals(), rate_per_min=1, duration_min=3, input_tokens=128)[2]
    sim.pe_output_log.append((9.0, 512, 0, inst.instance_id))
    pe = sim.catalog.profile_data.pe_models["pe7b"].hardware["A800"]

    estimate = sim.estimate_stage_service_time_s(req, inst, 10.0)
    expected = (pe.ttft_ms[128][512][2] + 511 * pe.tpot_ms[128][512][2]) / 1000.0

    assert req.output_tokens == 2048
    assert estimate == pytest.approx(expected)


def test_oracle_pe_estimate_uses_true_output_tokens_for_waiting_and_running():
    config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        flip_source_count=1,
        pe_output_estimate_mode=PE_OUTPUT_ESTIMATE_MODE_ORACLE,
    )
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )
    inst = sim.stage_instances["PE"][0]
    req = build_requests(smoke_intervals(), rate_per_min=1, duration_min=3, input_tokens=128)[2]
    sim.pe_output_log.append((9.0, 512, 0, inst.instance_id))
    pe = sim.catalog.profile_data.pe_models["pe7b"].hardware["A800"]

    waiting_estimate = sim.estimate_stage_service_time_s(req, inst, 10.0)
    running_estimate = sim.estimate_pe_service_time_s(req, inst, 600, 10.0)

    expected_waiting = (pe.ttft_ms[128][2048][2] + 2047 * pe.tpot_ms[128][2048][2]) / 1000.0
    expected_running = (2048 - 600) * pe.tpot_ms[128][2048][2] / 1000.0
    assert waiting_estimate == pytest.approx(expected_waiting)
    assert running_estimate == pytest.approx(expected_running)


def test_oracle_pe_complement_work_uses_true_waiting_tokens():
    config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        flip_source_count=1,
        pe_output_estimate_mode=PE_OUTPUT_ESTIMATE_MODE_ORACLE,
    )
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )
    inst = sim.stage_instances["PE"][0]
    sim.requests = build_requests(smoke_intervals(), rate_per_min=1, duration_min=3, input_tokens=128)
    inst.queue.append((0, 0))
    inst.queue.append((2, 0))

    assert sim.pe_waiting_tokens_for_work(inst, 10.0) == 512 + 2048


def test_pe_running_estimate_upgrades_bin_when_generated_exceeds_short_bin():
    config = SimulationConfig(scenario="smoke", deployment_preset="smoke", flip_source_count=1)
    sim = FlipSimulator(
        config=config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )
    inst = sim.stage_instances["PE"][0]
    req = build_requests(smoke_intervals(), rate_per_min=1, duration_min=3, input_tokens=128)[2]
    sim.pe_output_log.append((9.0, 512, 0, inst.instance_id))
    pe = sim.catalog.profile_data.pe_models["pe7b"].hardware["A800"]

    estimate = sim.estimate_pe_service_time_s(req, inst, 600, 10.0)
    expected = (2048 - 600) * pe.tpot_ms[128][2048][2] / 1000.0

    assert estimate == pytest.approx(expected)


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


def test_zero_cold_start_finishes_at_cold_start_begin():
    config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        request_rates_per_min=(12.0,),
        monitor_windows_sec=(10.0,),
        duration_min=6.0,
        flip_source_count=1,
        cold_start_mode=COLD_START_MODE_ZERO,
    )

    results, _intervals = run_suite(config)
    can_flip = next(result for result in results if result.mode == MODE_CAN_FLIP)

    assert can_flip.flip_events
    assert all(event.cold_start_done_time_s == event.cold_start_time_s for event in can_flip.flip_events)
    assert all(event.selection_restart_ready_delay_s < 1.0 for event in can_flip.flip_events)


def test_cold_start_init_time_modes_select_expected_profile_values():
    baseline_config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        flip_source_count=1,
        pe_init_time_mode=PE_INIT_TIME_MODE_NON_OPTIMIZED,
        generator_init_time_mode=GENERATOR_INIT_TIME_MODE_PROFILE,
    )
    optimized_config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        flip_source_count=1,
        pe_init_time_mode=PE_INIT_TIME_MODE_OPTIMIZED,
        generator_init_time_mode=GENERATOR_INIT_TIME_MODE_OPTIMIZED,
    )
    baseline = FlipSimulator(
        config=baseline_config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )
    optimized = FlipSimulator(
        config=optimized_config,
        catalog=ProfileCatalog(),
        intervals=smoke_intervals(),
        request_rate_per_min=12,
        monitor_window_sec=10,
        mode=MODE_CAN_FLIP,
    )

    assert baseline.pe_init_time_s("A800", 2) == pytest.approx(29.624)
    assert optimized.pe_init_time_s("A800", 2) == pytest.approx(20.963828)
    assert baseline.generator_init_time_s(8) == pytest.approx(61.020383)
    assert optimized.generator_init_time_s(8) == pytest.approx(37.61)


def test_cold_start_optimization_summary_compares_same_margin_variants():
    base_config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        request_rates_per_min=(12.0,),
        monitor_windows_sec=(10.0,),
        duration_min=6.0,
        flip_source_count=1,
        mode=MODE_CAN_FLIP,
        threshold_margin_ratio=0.05,
        pe_init_time_mode=PE_INIT_TIME_MODE_NON_OPTIMIZED,
        generator_init_time_mode=GENERATOR_INIT_TIME_MODE_PROFILE,
    )
    optimized_config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        request_rates_per_min=(12.0,),
        monitor_windows_sec=(10.0,),
        duration_min=6.0,
        flip_source_count=1,
        mode=MODE_CAN_FLIP,
        threshold_margin_ratio=0.05,
        pe_init_time_mode=PE_INIT_TIME_MODE_OPTIMIZED,
        generator_init_time_mode=GENERATOR_INIT_TIME_MODE_OPTIMIZED,
    )
    base_results, _ = run_suite(base_config)
    optimized_results, _ = run_suite(optimized_config)
    results = base_results + optimized_results
    add_cold_start_optimization_fields(results)
    text = render_cold_start_optimization_markdown(results, 0.05)

    optimized = next(result for result in results if result.summary["cold_start_variant"] == "optimized")
    assert "Cold-Start Optimization Comparison" in text
    assert "baseline thpt" in text
    assert "optimized thpt" in text
    assert "cold_start_throughput_lift_pct" not in text
    assert optimized.summary["cold_start_throughput_lift_pct"] == pytest.approx(
        (optimized.summary["throughput_req_s"] / base_results[0].summary["throughput_req_s"] - 1.0) * 100.0
    )


def test_margin_comparison_summary_groups_by_rate_and_window():
    base_config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        request_rates_per_min=(12.0,),
        monitor_windows_sec=(10.0,),
        duration_min=6.0,
        flip_source_count=1,
    )
    margin_config = SimulationConfig(
        scenario="smoke",
        deployment_preset="smoke",
        request_rates_per_min=(12.0,),
        monitor_windows_sec=(10.0,),
        duration_min=6.0,
        flip_source_count=1,
        mode=MODE_CAN_FLIP,
        threshold_margin_ratio=0.05,
    )
    base_results, _ = run_suite(base_config)
    margin_results, _ = run_suite(margin_config)
    results = base_results + margin_results
    add_comparison_fields(results)
    text = render_margin_comparison_markdown(results, 0.05)

    assert margin_label(0.05) == "margin_0p05"
    assert "## Request Rate 12 req/min" in text
    assert "throughput lift non-margin" in text
    assert "| 10 |" in text


def test_non_margin_summary_filters_margin_rows(tmp_path):
    path = tmp_path / "run_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "margin_label",
                "request_rate_per_min",
                "monitor_window_sec",
                "throughput_req_s",
                "throughput_vs_no_flip_pct",
                "slo10_pass_ratio",
                "slo10_vs_no_flip_delta",
                "slo5_pass_ratio",
                "slo5_vs_no_flip_delta",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "mode": "no_flip",
                "margin_label": "non_margin",
                "request_rate_per_min": 12,
                "monitor_window_sec": 10,
                "throughput_req_s": 1.0,
                "throughput_vs_no_flip_pct": 0.0,
                "slo10_pass_ratio": 0.1,
                "slo10_vs_no_flip_delta": 0.0,
                "slo5_pass_ratio": 0.05,
                "slo5_vs_no_flip_delta": 0.0,
            }
        )
        writer.writerow(
            {
                "mode": "can_flip",
                "margin_label": "non_margin",
                "request_rate_per_min": 12,
                "monitor_window_sec": 10,
                "throughput_req_s": 2.0,
                "throughput_vs_no_flip_pct": 100.0,
                "slo10_pass_ratio": 0.2,
                "slo10_vs_no_flip_delta": 0.1,
                "slo5_pass_ratio": 0.1,
                "slo5_vs_no_flip_delta": 0.05,
            }
        )
        writer.writerow(
            {
                "mode": "can_flip",
                "margin_label": "margin_0p10",
                "request_rate_per_min": 12,
                "monitor_window_sec": 10,
                "throughput_req_s": 3.0,
                "throughput_vs_no_flip_pct": 200.0,
                "slo10_pass_ratio": 0.3,
                "slo10_vs_no_flip_delta": 0.2,
                "slo5_pass_ratio": 0.2,
                "slo5_vs_no_flip_delta": 0.15,
            }
        )

    text = render_non_margin_summary_from_run_summary_csv(path)

    assert "| 12 | 10 | 1.000000 | 2.000000 | 100.000 |" in text
    assert "200.000" not in text


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
                "threshold_margin_ratio": 0.0,
                "threshold_margin_mode": "hysteresis",
                "compare_threshold_margin_ratios": None,
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
                "cold_start_mode": "profile",
                "pe_output_estimate_mode": "monitor",
                "pe_init_time_mode": "optimized",
                "generator_init_time_mode": "profile",
                "compare_cold_start_optimization": False,
                "trace_start": "2024-10-15T12:00:00+00:00",
                "verbose": False,
            },
        )()
    )

    assert parser_config.request_rates_per_min == (12.0,)
    assert parser_config.monitor_windows_sec == (10.0,)
    assert parser_config.duration_min == 6.0
    assert parser_config.flip_source_count == 1
