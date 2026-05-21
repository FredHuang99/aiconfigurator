# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import statistics
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Iterable

from aiconfigurator.td_deployment_search.profiles import ProfileCatalog, canonical_generator_name


STAGE_PE = "PE"
STAGE_ENCODER = "Encoder_CPU"
STAGE_DIT = "DiT"
STAGE_VAE = "VAE"
STAGE_SEQUENCE = (STAGE_PE, STAGE_ENCODER, STAGE_DIT, STAGE_VAE)

TEMPLATE_PE_ONLY = "PE_Only"
TEMPLATE_DIT_ONLY = "DiT_Only"
TEMPLATE_VAE_ONLY = "VAE_Only"
TEMPLATE_DIT_VAE_COMB = "DiT_VAE_Comb"

MODE_NO_FLIP = "no_flip"
MODE_CAN_FLIP = "can_flip"
MARGIN_MODE_HYSTERESIS = "hysteresis"
MARGIN_MODE_SINGLE = "single"
COLD_START_MODE_PROFILE = "profile"
COLD_START_MODE_ZERO = "zero"
PE_OUTPUT_ESTIMATE_MODE_MONITOR = "monitor"
PE_OUTPUT_ESTIMATE_MODE_ORACLE = "oracle"
PE_INIT_TIME_MODE_OPTIMIZED = "optimized"
PE_INIT_TIME_MODE_NON_OPTIMIZED = "non_optimized"
GENERATOR_INIT_TIME_MODE_PROFILE = "profile"
GENERATOR_INIT_TIME_MODE_OPTIMIZED = "optimized"
EPS = 1e-9

DEFAULT_DOMINANT_INTERVALS_CSV = Path(
    r"C:\Users\woshi\Downloads\AzureLMMInferenceTrace_multimodal\data\new"
    r"\window1_hour1_dominant_intervals_thr98.csv"
)
DEFAULT_OUTPUT_DIR = Path("td_outputs") / "flip_simulation"
DEFAULT_TRACE_START = "2024-10-15T12:00:00+00:00"


def margin_label(ratio: float) -> str:
    if abs(ratio) <= EPS:
        return "non_margin"
    return f"margin_{ratio:.2f}".replace(".", "p")


@dataclass(frozen=True)
class DominantInterval:
    start_s: float
    end_s: float
    source_dominant: str
    resolved_dominant: str
    output_tokens: int
    start_time: str = ""
    end_time: str = ""


@dataclass(frozen=True)
class SimulationConfig:
    dominant_intervals_csv: Path = DEFAULT_DOMINANT_INTERVALS_CSV
    out_dir: Path = DEFAULT_OUTPUT_DIR
    request_rates_per_min: tuple[float, ...] = (48.0, 60.0, 90.0)
    monitor_windows_sec: tuple[float, ...] = (10.0, 20.0, 30.0, 60.0)
    duration_min: float = 60.0
    token_threshold: float = 1280.0
    threshold_margin_ratio: float = 0.0
    threshold_margin_mode: str = MARGIN_MODE_HYSTERESIS
    compare_threshold_margin_ratios: tuple[float, ...] = ()
    seed: int = 0
    mode: str = "both"
    pe_model: str = "pe7b"
    generator_model: str = "wan2.2-ti2v-5b"
    input_tokens: int = 128
    short_output_tokens: int = 512
    long_output_tokens: int = 2048
    denoising_steps: int = 50
    scenario: str = "default"
    deployment_preset: str = "wan22_group1_fragment"
    flip_plan_preset: str = "wan22_group1_restricted"
    flip_source_count: int = 2
    cold_start_mode: str = COLD_START_MODE_PROFILE
    pe_output_estimate_mode: str = PE_OUTPUT_ESTIMATE_MODE_MONITOR
    pe_init_time_mode: str = PE_INIT_TIME_MODE_OPTIMIZED
    generator_init_time_mode: str = GENERATOR_INIT_TIME_MODE_PROFILE
    compare_cold_start_optimization: bool = False
    trace_start: str = DEFAULT_TRACE_START
    verbose: bool = False


@dataclass
class Request:
    req_id: int
    arrival_time_s: float
    input_tokens: int
    output_tokens: int
    request_type: str
    trace_interval_start_s: float
    trace_interval_end_s: float
    stage_index: int = 0
    pe_generated_tokens: int = 0
    dit_completed_steps: int = 0
    finish_time_s: float | None = None

    def current_stage(self) -> str | None:
        if self.stage_index >= len(STAGE_SEQUENCE):
            return None
        return STAGE_SEQUENCE[self.stage_index]


@dataclass
class StageAttempt:
    attempt_id: int
    req_id: int
    stage: str
    attempt_index: int
    instance_id: str
    template_name: str
    node_name: str
    gpu_type: str
    parallelism: int
    bundle_key: str
    enter_time_s: float
    queue_insert: str
    start_time_s: float | None = None
    exit_time_s: float | None = None
    exit_reason: str = ""
    service_time_s: float | None = None
    pe_actual_prefill_tokens: int | None = None
    pe_actual_remaining_output_tokens: int | None = None
    pe_matched_prefill_tokens: int | None = None
    pe_matched_remaining_output_tokens: int | None = None
    pe_ttft_s: float | None = None
    pe_tpot_s: float | None = None


@dataclass(frozen=True)
class PEExecution:
    start_generated_tokens: int
    output_tokens: int
    ttft_s: float
    tpot_s: float
    decode_tokens: int
    ttft_generates_token: bool

    @property
    def total_duration_s(self) -> float:
        return self.ttft_s + self.decode_tokens * self.tpot_s

    def generated_after_elapsed(self, elapsed_s: float) -> int:
        elapsed_s = max(0.0, elapsed_s)
        generated = self.start_generated_tokens
        if elapsed_s + EPS < self.ttft_s:
            return min(self.output_tokens, generated)
        if self.ttft_generates_token:
            generated += 1
        if self.tpot_s > 0 and self.decode_tokens > 0:
            decode_elapsed = max(0.0, elapsed_s - self.ttft_s)
            generated += min(self.decode_tokens, int(math.floor((decode_elapsed + EPS) / self.tpot_s)))
        return min(self.output_tokens, generated)

    def next_step_boundary_time(self, start_time_s: float, now_s: float) -> float:
        completion_time = start_time_s + self.total_duration_s
        elapsed_s = max(0.0, now_s - start_time_s)
        if elapsed_s + EPS < self.ttft_s:
            return min(completion_time, start_time_s + self.ttft_s)
        if self.decode_tokens <= 0 or self.tpot_s <= 0:
            return completion_time
        decode_elapsed = max(0.0, elapsed_s - self.ttft_s)
        quotient = decode_elapsed / self.tpot_s
        if abs(quotient - round(quotient)) <= 1e-7:
            return min(completion_time, now_s)
        next_token_index = int(math.floor(quotient)) + 1
        return min(completion_time, start_time_s + self.ttft_s + next_token_index * self.tpot_s)


@dataclass(frozen=True)
class StageRun:
    total_duration_s: float
    pe_execution: PEExecution | None = None
    dit_start_steps: int = 0
    dit_step_s: float = 0.0

    def dit_completed_after_elapsed(self, elapsed_s: float, denoising_steps: int) -> int:
        if self.dit_step_s <= 0:
            return denoising_steps
        elapsed_s = max(0.0, elapsed_s)
        completed = self.dit_start_steps + int(math.floor((elapsed_s + EPS) / self.dit_step_s))
        return min(denoising_steps, completed)

    def dit_next_step_boundary_time(self, start_time_s: float, now_s: float) -> float:
        completion_time = start_time_s + self.total_duration_s
        if self.dit_step_s <= 0:
            return completion_time
        elapsed_s = max(0.0, now_s - start_time_s)
        quotient = elapsed_s / self.dit_step_s
        if abs(quotient - round(quotient)) <= 1e-7:
            return min(completion_time, now_s)
        next_step_index = int(math.floor(quotient)) + 1
        return min(completion_time, start_time_s + next_step_index * self.dit_step_s)


@dataclass
class StageInstance:
    instance_id: str
    stage: str
    template_name: str
    node_name: str
    gpu_type: str
    parallelism: int
    bundle_key: str
    order: int
    base_latency_s: float = 0.0
    active: bool = True
    ready: bool = True
    launching: bool = False
    draining: bool = False
    origin_bundle_key: str | None = None
    queue: Deque[tuple[int, int]] = field(default_factory=deque)
    current_req_id: int | None = None
    current_attempt_id: int | None = None
    current_start_time_s: float | None = None
    current_run: StageRun | None = None
    generation: int = 0

    def active_ready(self) -> bool:
        return self.active and self.ready and (not self.launching) and (not self.draining)

    def load_count(self) -> int:
        return (1 if self.current_req_id is not None else 0) + len(self.queue)


@dataclass
class BundleState:
    bundle_key: str
    template_name: str
    node_name: str
    gpu_type: str
    parallelism: int
    bundle_size: int
    order: int
    stage_instances: dict[str, StageInstance] = field(default_factory=dict)
    pe_target_instances: list[StageInstance] = field(default_factory=list)
    active: bool = True
    converted_to_pe: bool = False
    draining: bool = False

    def has_running(self) -> bool:
        return any(inst.current_req_id is not None for inst in self.stage_instances.values())

    def waiting_count(self) -> int:
        return sum(inst.load_count() for inst in self.stage_instances.values())


@dataclass
class FlipEvent:
    flip_id: int
    mode: str
    request_rate_per_min: float
    monitor_window_sec: float
    direction: str
    detect_time_s: float
    avg_output_tokens: float
    window_count: int
    threshold_mid: float
    threshold_low: float
    threshold_high: float
    margin_ratio: float
    margin_label: str
    monitor_decision: str
    trace_change_time_s: float | None
    detection_delay_s: float | None
    selected_bundle_keys: list[str]
    selected_instance_ids: list[str]
    migrated_waiting_requests: int = 0
    migrated_running_requests: int = 0
    drain_done_time_s: float | None = None
    cold_start_time_s: float | None = None
    cold_start_done_time_s: float | None = None
    launch_target_instance_ids: list[str] = field(default_factory=list)
    selection_restart_ready_delay_s: float = 0.0
    selection_complement_added_work_s: float = 0.0
    selection_running_added_work_s: float = 0.0
    selection_waiting_added_work_s: float = 0.0
    selection_pe_bin_tokens: str = ""
    selection_migrated_request_count: int = 0
    selection_template_penalty: int = 0
    selection_round_robin_rank: int = 0


@dataclass(frozen=True)
class FlipSelectionScore:
    restart_ready_delay_s: float
    complement_added_work_s: float
    running_added_work_s: float
    waiting_added_work_s: float
    pe_bin_tokens: str
    migrated_request_count: int
    template_penalty: int
    round_robin_rank: int
    bundle_order_key: tuple[int, ...] = ()

    def sort_key(self) -> tuple[float, float, int, int, tuple[int, ...]]:
        return (
            self.restart_ready_delay_s,
            self.complement_added_work_s,
            self.migrated_request_count,
            self.template_penalty,
            self.round_robin_rank,
            self.bundle_order_key,
        )


@dataclass(frozen=True)
class FlipSelection:
    bundles: tuple[BundleState, ...]
    score: FlipSelectionScore


@dataclass
class RunResult:
    run_id: str
    mode: str
    request_rate_per_min: float
    monitor_window_sec: float
    requests: list[Request]
    attempts: list[StageAttempt]
    flip_events: list[FlipEvent]
    summary: dict[str, object]


def load_dominant_intervals(
    path: Path,
    *,
    short_output_tokens: int = 512,
    long_output_tokens: int = 2048,
) -> list[DominantInterval]:
    intervals: list[DominantInterval] = []
    previous = "Short"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row["dominant"].strip()
            resolved = previous if source.lower() == "tie" else source
            if resolved not in {"Short", "Long"}:
                raise ValueError(f"Unsupported dominant request type: {source}")
            previous = resolved
            output_tokens = short_output_tokens if resolved == "Short" else long_output_tokens
            intervals.append(
                DominantInterval(
                    start_s=float(row["start_minute"]) * 60.0,
                    end_s=float(row["end_minute"]) * 60.0,
                    source_dominant=source,
                    resolved_dominant=resolved,
                    output_tokens=output_tokens,
                    start_time=row.get("start_time", ""),
                    end_time=row.get("end_time", ""),
                )
            )
    if not intervals:
        raise ValueError(f"No dominant intervals found in {path}")
    return intervals


def smoke_intervals(short_output_tokens: int = 512, long_output_tokens: int = 2048) -> list[DominantInterval]:
    return [
        DominantInterval(0.0, 120.0, "Short", "Short", short_output_tokens),
        DominantInterval(120.0, 240.0, "Long", "Long", long_output_tokens),
        DominantInterval(240.0, 360.0, "Short", "Short", short_output_tokens),
    ]


def output_for_time(intervals: list[DominantInterval], t_s: float) -> DominantInterval:
    for interval in intervals:
        if interval.start_s <= t_s < interval.end_s:
            return interval
    if abs(t_s - intervals[-1].end_s) <= EPS:
        return intervals[-1]
    raise ValueError(f"No dominant interval covers t={t_s:.6f}s")


def build_requests(
    intervals: list[DominantInterval],
    *,
    rate_per_min: float,
    duration_min: float,
    input_tokens: int,
) -> list[Request]:
    if rate_per_min <= 0:
        raise ValueError("request rate must be positive")
    total_requests = int(round(rate_per_min * duration_min))
    arrival_interval_s = 60.0 / rate_per_min
    requests: list[Request] = []
    for req_id in range(total_requests):
        arrival = req_id * arrival_interval_s
        interval = output_for_time(intervals, arrival)
        requests.append(
            Request(
                req_id=req_id,
                arrival_time_s=arrival,
                input_tokens=input_tokens,
                output_tokens=interval.output_tokens,
                request_type=interval.resolved_dominant,
                trace_interval_start_s=interval.start_s,
                trace_interval_end_s=interval.end_s,
            )
        )
    return requests


def dominant_timeline_rows(intervals: list[DominantInterval], trace_start: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start_dt = _parse_datetime(trace_start)
    for interval in intervals:
        rows.append(
            {
                "start_s": interval.start_s,
                "end_s": interval.end_s,
                "start_time": interval.start_time or (start_dt + timedelta(seconds=interval.start_s)).isoformat(),
                "end_time": interval.end_time or (start_dt + timedelta(seconds=interval.end_s)).isoformat(),
                "source_dominant": interval.source_dominant,
                "resolved_dominant": interval.resolved_dominant,
                "output_tokens": interval.output_tokens,
            }
        )
    return rows


class FlipSimulator:
    def __init__(
        self,
        *,
        config: SimulationConfig,
        catalog: ProfileCatalog,
        intervals: list[DominantInterval],
        request_rate_per_min: float,
        monitor_window_sec: float,
        mode: str,
    ) -> None:
        if mode not in {MODE_NO_FLIP, MODE_CAN_FLIP}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.config = config
        self.catalog = catalog
        self.intervals = intervals
        self.request_rate_per_min = request_rate_per_min
        self.monitor_window_sec = monitor_window_sec
        self.mode = mode
        run_suffix_parts = [margin_label(config.threshold_margin_ratio)]
        if config.cold_start_mode != COLD_START_MODE_PROFILE:
            run_suffix_parts.append(f"cold_{config.cold_start_mode}")
        if config.pe_init_time_mode != PE_INIT_TIME_MODE_OPTIMIZED:
            run_suffix_parts.append(f"pe_init_{config.pe_init_time_mode}")
        if config.generator_init_time_mode != GENERATOR_INIT_TIME_MODE_PROFILE:
            run_suffix_parts.append(f"gen_init_{config.generator_init_time_mode}")
        if config.pe_output_estimate_mode != PE_OUTPUT_ESTIMATE_MODE_MONITOR:
            run_suffix_parts.append(f"pe_est_{config.pe_output_estimate_mode}")
        self.run_id = (
            f"rate{request_rate_per_min:g}_window{monitor_window_sec:g}_{mode}_"
            f"{'_'.join(run_suffix_parts)}"
        )

        self.event_q: list[tuple[float, int, int, str, dict[str, object]]] = []
        self.event_counter = 0
        self.requests: list[Request] = []
        self.attempts: list[StageAttempt] = []
        self.attempt_counts: Counter[tuple[int, str]] = Counter()
        self.pe_output_log: list[tuple[float, int, int, str]] = []
        self.pe_output_window: Deque[tuple[float, int, int, str]] = deque()
        self.pe_window_tokens_by_instance: Counter[str] = Counter()
        self.pe_window_counts_by_instance: Counter[str] = Counter()
        self.pe_window_total_tokens = 0
        self.pe_window_total_count = 0
        self.pe_window_loaded_index = 0
        self.pe_window_now_s = float("-inf")
        self.flip_events: list[FlipEvent] = []
        self.stage_instances: dict[str, list[StageInstance]] = {stage: [] for stage in STAGE_SEQUENCE}
        self.bundles: dict[str, BundleState] = {}
        self.converted_bundle_keys: list[str] = []
        self.current_target = "short"
        self.launching_direction: str | None = None
        self.flip_round_robin_cursor = 0
        self.next_order = 0

        self._build_deployment()

    def _build_deployment(self) -> None:
        if self.config.deployment_preset == "smoke":
            self._build_smoke_deployment()
            return
        if self.config.deployment_preset != "wan22_group1_fragment":
            raise ValueError(f"Unsupported deployment preset: {self.config.deployment_preset}")
        for bundle_idx in range(1, 5):
            self._add_pe_instance(
                instance_id=f"PE_A800_8_{bundle_idx}",
                node_name="A800-NVLink-8",
                bundle_label=f"bundle-{bundle_idx}",
                origin_bundle_key=None,
            )
        for idx in range(5):
            self._add_stage_instance(
                stage=STAGE_ENCODER,
                template_name="Encoder_CPU",
                node_name=f"CPU-{idx + 1}",
                gpu_type="CPU",
                parallelism=1,
                bundle_key=f"CPU-{idx + 1}/encoder",
                base_latency_s=self.catalog.encoder_cpu_latency_s(self.config.generator_model),
            )
        for node_idx in range(1, 5):
            self._add_comb_bundle(f"H100-NVLink-{node_idx}", "H100")
        for node_idx in range(1, 8):
            self._add_comb_bundle(f"A800-NVLink-{node_idx}", "A800")
        for node_idx in range(1, 13):
            self._add_comb_bundle(f"A100-NVLink-{node_idx}", "A100")

    def _build_smoke_deployment(self) -> None:
        self._add_pe_instance(
            instance_id="PE_SMOKE_BASE_1",
            node_name="A800-NVLink-smoke-base",
            bundle_label="bundle-1",
            origin_bundle_key=None,
        )
        self._add_stage_instance(
            stage=STAGE_ENCODER,
            template_name="Encoder_CPU",
            node_name="CPU-smoke-1",
            gpu_type="CPU",
            parallelism=1,
            bundle_key="CPU-smoke-1/encoder",
            base_latency_s=self.catalog.encoder_cpu_latency_s(self.config.generator_model),
        )
        self._add_comb_bundle("A800-NVLink-smoke-1", "A800")
        self._add_comb_bundle("A800-NVLink-smoke-2", "A800")

    def _add_stage_instance(
        self,
        *,
        stage: str,
        template_name: str,
        node_name: str,
        gpu_type: str,
        parallelism: int,
        bundle_key: str,
        base_latency_s: float = 0.0,
        active: bool = True,
        ready: bool = True,
        origin_bundle_key: str | None = None,
    ) -> StageInstance:
        instance_id = f"{stage}_{node_name}_{parallelism}_{len(self.stage_instances[stage]) + 1}"
        if stage == STAGE_ENCODER:
            instance_id = f"ENC_{len(self.stage_instances[stage]) + 1}"
        inst = StageInstance(
            instance_id=instance_id,
            stage=stage,
            template_name=template_name,
            node_name=node_name,
            gpu_type=gpu_type,
            parallelism=parallelism,
            bundle_key=bundle_key,
            order=self.next_order,
            base_latency_s=base_latency_s,
            active=active,
            ready=ready,
            origin_bundle_key=origin_bundle_key,
        )
        self.next_order += 1
        self.stage_instances[stage].append(inst)
        return inst

    def _add_pe_instance(
        self,
        *,
        instance_id: str,
        node_name: str,
        bundle_label: str,
        origin_bundle_key: str | None,
        active: bool = True,
        ready: bool = True,
    ) -> StageInstance:
        bundle_key = f"{node_name}/{bundle_label}"
        inst = StageInstance(
            instance_id=instance_id,
            stage=STAGE_PE,
            template_name=TEMPLATE_PE_ONLY,
            node_name=node_name,
            gpu_type="A800",
            parallelism=2,
            bundle_key=bundle_key,
            order=self.next_order,
            active=active,
            ready=ready,
            launching=active and not ready,
            origin_bundle_key=origin_bundle_key,
        )
        self.next_order += 1
        self.stage_instances[STAGE_PE].append(inst)
        return inst

    def _add_comb_bundle(self, node_name: str, gpu_type: str) -> BundleState:
        bundle_key = f"{node_name}/bundle-1"
        generator = canonical_generator_name(self.config.generator_model)
        dit_profile = self.catalog.generator_stage_profile(generator, STAGE_DIT, gpu_type, 8)
        vae_profile = self.catalog.generator_stage_profile(generator, STAGE_VAE, gpu_type, 8)
        bundle = BundleState(
            bundle_key=bundle_key,
            template_name=TEMPLATE_DIT_VAE_COMB,
            node_name=node_name,
            gpu_type=gpu_type,
            parallelism=8,
            bundle_size=8,
            order=self.next_order,
        )
        self.next_order += 1
        dit = self._add_stage_instance(
            stage=STAGE_DIT,
            template_name=TEMPLATE_DIT_VAE_COMB,
            node_name=node_name,
            gpu_type=gpu_type,
            parallelism=8,
            bundle_key=bundle_key,
            base_latency_s=dit_profile.latency_s,
        )
        vae = self._add_stage_instance(
            stage=STAGE_VAE,
            template_name=TEMPLATE_DIT_VAE_COMB,
            node_name=node_name,
            gpu_type=gpu_type,
            parallelism=8,
            bundle_key=bundle_key,
            base_latency_s=vae_profile.latency_s,
        )
        bundle.stage_instances = {STAGE_DIT: dit, STAGE_VAE: vae}
        self.bundles[bundle_key] = bundle
        return bundle

    def _ensure_pe_targets(self, bundle: BundleState) -> list[StageInstance]:
        if bundle.pe_target_instances:
            return bundle.pe_target_instances
        for idx in range(4):
            inst = self._add_pe_instance(
                instance_id=f"PE_FLIP_{bundle.node_name}_{idx + 1}",
                node_name=bundle.node_name,
                bundle_label=f"flip-pe-{idx + 1}",
                origin_bundle_key=bundle.bundle_key,
                active=False,
                ready=False,
            )
            bundle.pe_target_instances.append(inst)
        return bundle.pe_target_instances

    def schedule_event(self, time_s: float, priority: int, event_type: str, payload: dict[str, object]) -> None:
        heapq.heappush(self.event_q, (time_s, priority, self.event_counter, event_type, payload))
        self.event_counter += 1

    def schedule_completion(self, time_s: float, inst: StageInstance, req_id: int, generation: int) -> None:
        self.schedule_event(
            time_s,
            0,
            "completion",
            {
                "stage": inst.stage,
                "instance_id": inst.instance_id,
                "req_id": req_id,
                "generation": generation,
            },
        )

    def add_requests(self, requests: list[Request]) -> None:
        self.requests = requests
        for req in requests:
            self.schedule_event(req.arrival_time_s, 2, "arrival", {"req_id": req.req_id})
        if self.mode == MODE_CAN_FLIP:
            self.schedule_event(self.monitor_window_sec, 3, "monitor", {})

    def run(self) -> RunResult:
        while self.event_q:
            time_s, _priority, _counter, event_type, payload = heapq.heappop(self.event_q)
            if event_type == "arrival":
                self.handle_arrival(int(payload["req_id"]), time_s)
            elif event_type == "completion":
                self.handle_completion(
                    str(payload["stage"]),
                    str(payload["instance_id"]),
                    int(payload["req_id"]),
                    int(payload["generation"]),
                    time_s,
                )
            elif event_type == "monitor":
                self.handle_monitor(time_s)
            elif event_type == "source_drain":
                self.handle_source_drain(str(payload["instance_id"]), int(payload["flip_id"]), time_s)
            elif event_type == "cold_start_begin":
                self.handle_cold_start_begin(int(payload["flip_id"]), time_s)
            elif event_type == "launch_done":
                self.handle_launch_done(int(payload["flip_id"]), time_s)
            else:
                raise RuntimeError(f"Unknown event type: {event_type}")

            if self.requests and all(req.finish_time_s is not None for req in self.requests):
                self.event_q.clear()

        summary = self.compute_summary()
        return RunResult(
            run_id=self.run_id,
            mode=self.mode,
            request_rate_per_min=self.request_rate_per_min,
            monitor_window_sec=self.monitor_window_sec,
            requests=self.requests,
            attempts=self.attempts,
            flip_events=self.flip_events,
            summary=summary,
        )

    def instance_by_id(self, instance_id: str) -> StageInstance:
        for instances in self.stage_instances.values():
            for inst in instances:
                if inst.instance_id == instance_id:
                    return inst
        raise KeyError(instance_id)

    def handle_arrival(self, req_id: int, now_s: float) -> None:
        req = self.requests[req_id]
        self.dispatch_to_stage(req, STAGE_PE, now_s)

    def handle_completion(
        self,
        stage: str,
        instance_id: str,
        req_id: int,
        generation: int,
        now_s: float,
    ) -> None:
        inst = self.instance_by_id(instance_id)
        if inst.generation != generation or inst.current_req_id != req_id:
            return
        req = self.requests[req_id]
        attempt = self.attempts[inst.current_attempt_id] if inst.current_attempt_id is not None else None
        if attempt is not None:
            attempt.exit_time_s = now_s
            attempt.exit_reason = "completed"
        if stage == STAGE_PE:
            req.pe_generated_tokens = req.output_tokens
            self.pe_output_log.append((now_s, req.output_tokens, req.req_id, inst.instance_id))
        elif stage == STAGE_DIT:
            req.dit_completed_steps = self.config.denoising_steps

        inst.current_req_id = None
        inst.current_attempt_id = None
        inst.current_start_time_s = None
        inst.current_run = None
        inst.generation += 1
        self.start_next_if_possible(inst, now_s)

        req.stage_index += 1
        next_stage = req.current_stage()
        if next_stage is None:
            req.finish_time_s = now_s
        else:
            self.dispatch_to_stage(req, next_stage, now_s)

    def dispatch_to_stage(self, req: Request, stage: str, now_s: float) -> None:
        inst = self.choose_least_waiting(stage, incoming_req=req, now_s=now_s)
        self.enqueue_request(req, inst, now_s, queue_insert="tail", at_head=False)
        self.start_next_if_possible(inst, now_s)

    def enqueue_request(
        self,
        req: Request,
        inst: StageInstance,
        now_s: float,
        *,
        queue_insert: str,
        at_head: bool,
    ) -> StageAttempt:
        key = (req.req_id, inst.stage)
        self.attempt_counts[key] += 1
        attempt = StageAttempt(
            attempt_id=len(self.attempts),
            req_id=req.req_id,
            stage=inst.stage,
            attempt_index=self.attempt_counts[key],
            instance_id=inst.instance_id,
            template_name=inst.template_name,
            node_name=inst.node_name,
            gpu_type=inst.gpu_type,
            parallelism=inst.parallelism,
            bundle_key=inst.bundle_key,
            enter_time_s=now_s,
            queue_insert=queue_insert,
        )
        self.attempts.append(attempt)
        item = (req.req_id, attempt.attempt_id)
        if at_head:
            inst.queue.appendleft(item)
        else:
            inst.queue.append(item)
        return attempt

    def start_next_if_possible(self, inst: StageInstance, now_s: float) -> None:
        if not inst.active_ready() or inst.current_req_id is not None or not inst.queue:
            return
        req_id, attempt_id = inst.queue.popleft()
        req = self.requests[req_id]
        attempt = self.attempts[attempt_id]
        run = self.build_stage_run(req, inst, attempt)
        inst.current_req_id = req_id
        inst.current_attempt_id = attempt_id
        inst.current_start_time_s = now_s
        inst.current_run = run
        inst.generation += 1
        attempt.start_time_s = now_s
        attempt.service_time_s = run.total_duration_s
        self.schedule_completion(now_s + run.total_duration_s, inst, req_id, inst.generation)

    def build_stage_run(self, req: Request, inst: StageInstance, attempt: StageAttempt) -> StageRun:
        if inst.stage == STAGE_PE:
            pe_execution = self.build_pe_execution(req, inst, attempt)
            return StageRun(total_duration_s=pe_execution.total_duration_s, pe_execution=pe_execution)
        if inst.stage == STAGE_ENCODER:
            return StageRun(total_duration_s=inst.base_latency_s)
        if inst.stage == STAGE_DIT:
            remaining_steps = max(0, self.config.denoising_steps - req.dit_completed_steps)
            step_s = inst.base_latency_s / self.config.denoising_steps
            return StageRun(
                total_duration_s=remaining_steps * step_s,
                dit_start_steps=req.dit_completed_steps,
                dit_step_s=step_s,
            )
        if inst.stage == STAGE_VAE:
            return StageRun(total_duration_s=inst.base_latency_s)
        raise ValueError(f"Unsupported stage: {inst.stage}")

    def estimate_stage_service_time_s(self, req: Request, inst: StageInstance, now_s: float = 0.0) -> float:
        if inst.stage == STAGE_PE:
            return self.estimate_pe_service_time_s(req, inst, req.pe_generated_tokens, now_s)
        if inst.stage == STAGE_ENCODER:
            return max(inst.base_latency_s, EPS)
        if inst.stage == STAGE_DIT:
            return self.estimate_dit_service_time_s(req, inst, req.dit_completed_steps)
        if inst.stage == STAGE_VAE:
            return max(inst.base_latency_s, EPS)
        raise ValueError(f"Unsupported stage: {inst.stage}")

    def estimate_dit_service_time_s(
        self,
        req: Request,
        inst: StageInstance,
        completed_steps: int,
    ) -> float:
        _ = req
        remaining_steps = max(0, self.config.denoising_steps - completed_steps)
        step_s = inst.base_latency_s / self.config.denoising_steps
        return max(remaining_steps * step_s, 0.0)

    def estimate_pe_service_time_s(
        self,
        req: Request,
        inst: StageInstance,
        generated_tokens: int,
        now_s: float = 0.0,
    ) -> float:
        output_tokens = self.pe_output_tokens_for_estimate(req, inst, now_s, generated_tokens)
        return self.estimate_pe_service_time_for_output_s(req, inst, output_tokens, generated_tokens)

    def pe_output_tokens_for_estimate(
        self,
        req: Request,
        inst: StageInstance,
        now_s: float,
        generated_tokens: int = 0,
    ) -> int:
        if self.config.pe_output_estimate_mode == PE_OUTPUT_ESTIMATE_MODE_ORACLE:
            return req.output_tokens
        if self.config.pe_output_estimate_mode != PE_OUTPUT_ESTIMATE_MODE_MONITOR:
            raise ValueError(f"Unsupported PE output estimate mode: {self.config.pe_output_estimate_mode}")
        return self.pe_estimated_output_bin(inst, now_s, generated_tokens)

    def estimate_pe_service_time_for_output_s(
        self,
        req: Request,
        inst: StageInstance,
        output_tokens: int,
        generated_tokens: int,
    ) -> float:
        hardware = self.catalog.profile_data.pe_models[self.config.pe_model].hardware[inst.gpu_type]
        generated = max(0, generated_tokens)
        ttft_s = hardware.ttft_ms[req.input_tokens][output_tokens][inst.parallelism] / 1000.0
        tpot_s = hardware.tpot_ms[req.input_tokens][output_tokens][inst.parallelism] / 1000.0
        if generated <= 0:
            return max(ttft_s + max(0, output_tokens - 1) * tpot_s, EPS)
        return max(max(0, output_tokens - generated) * tpot_s, 0.0)

    def build_pe_execution(self, req: Request, inst: StageInstance, attempt: StageAttempt) -> PEExecution:
        hardware = self.catalog.profile_data.pe_models[self.config.pe_model].hardware[inst.gpu_type]
        generated = req.pe_generated_tokens
        if generated <= 0:
            ttft_s = hardware.ttft_ms[req.input_tokens][req.output_tokens][inst.parallelism] / 1000.0
            tpot_s = hardware.tpot_ms[req.input_tokens][req.output_tokens][inst.parallelism] / 1000.0
            attempt.pe_actual_prefill_tokens = req.input_tokens
            attempt.pe_actual_remaining_output_tokens = req.output_tokens
            attempt.pe_matched_prefill_tokens = req.input_tokens
            attempt.pe_matched_remaining_output_tokens = req.output_tokens
            attempt.pe_ttft_s = ttft_s
            attempt.pe_tpot_s = tpot_s
            return PEExecution(
                start_generated_tokens=0,
                output_tokens=req.output_tokens,
                ttft_s=ttft_s,
                tpot_s=tpot_s,
                decode_tokens=max(0, req.output_tokens - 1),
                ttft_generates_token=True,
            )

        actual_prefill = req.input_tokens + generated
        actual_remaining = max(0, req.output_tokens - generated)
        matched_prefill, matched_remaining, ttft_s, tpot_s = self.lookup_pe_reprefill(
            inst.gpu_type,
            inst.parallelism,
            actual_prefill,
            actual_remaining,
        )
        attempt.pe_actual_prefill_tokens = actual_prefill
        attempt.pe_actual_remaining_output_tokens = actual_remaining
        attempt.pe_matched_prefill_tokens = matched_prefill
        attempt.pe_matched_remaining_output_tokens = matched_remaining
        attempt.pe_ttft_s = ttft_s
        attempt.pe_tpot_s = tpot_s
        return PEExecution(
            start_generated_tokens=generated,
            output_tokens=req.output_tokens,
            ttft_s=ttft_s,
            tpot_s=tpot_s,
            decode_tokens=actual_remaining,
            ttft_generates_token=False,
        )

    def lookup_pe_reprefill(
        self,
        gpu_type: str,
        parallelism: int,
        actual_prefill: int,
        actual_remaining: int,
    ) -> tuple[int, int, float, float]:
        hardware = self.catalog.profile_data.pe_models[self.config.pe_model].hardware[gpu_type]
        ttft_table = hardware.sim_ttft_ms or hardware.ttft_ms
        tpot_table = hardware.sim_tpot_ms or hardware.tpot_ms
        prefill_keys = sorted(
            k for k, inner in ttft_table.items() if any(parallelism in vals for vals in inner.values())
        )
        if not prefill_keys:
            raise KeyError(f"No PE sim prefill keys for {self.config.pe_model} {gpu_type} P{parallelism}")
        matched_prefill = min(prefill_keys, key=lambda key: (abs(key - actual_prefill), key))
        remaining_keys = sorted(k for k, vals in ttft_table[matched_prefill].items() if parallelism in vals)
        matched_remaining = min(remaining_keys, key=lambda key: (abs(key - actual_remaining), key))
        return (
            matched_prefill,
            matched_remaining,
            ttft_table[matched_prefill][matched_remaining][parallelism] / 1000.0,
            tpot_table[matched_prefill][matched_remaining][parallelism] / 1000.0,
        )

    def choose_least_waiting(
        self,
        stage: str,
        candidates: Iterable[StageInstance] | None = None,
        estimated_scores: dict[str, float] | None = None,
        incoming_req: Request | None = None,
        now_s: float = 0.0,
    ) -> StageInstance:
        pool = list(candidates) if candidates is not None else self.stage_instances[stage]
        ready = [inst for inst in pool if inst.active_ready()]
        if not ready:
            raise RuntimeError(f"No active ready instances for stage {stage}")
        if estimated_scores is None:
            estimated_scores = self.estimated_work_scores(ready, now_s)

        def candidate_score(inst: StageInstance) -> float:
            score = estimated_scores[inst.instance_id]
            if incoming_req is not None:
                score += self.estimate_stage_service_time_s(incoming_req, inst, now_s)
            return score

        return min(ready, key=lambda inst: (candidate_score(inst), inst.order))

    def estimated_work_scores(self, candidates: Iterable[StageInstance], now_s: float) -> dict[str, float]:
        return {inst.instance_id: self.instance_remaining_work_s(inst, now_s) for inst in candidates}

    def incoming_load_increment(
        self,
        candidates: Iterable[StageInstance],
        inst: StageInstance,
        req: Request,
        now_s: float = 0.0,
    ) -> float:
        _ = candidates
        return self.estimate_stage_service_time_s(req, inst, now_s)

    def estimated_load_scores(self, candidates: Iterable[StageInstance]) -> dict[str, float]:
        return self.estimated_work_scores(candidates, 0.0)

    def instance_remaining_work_s(self, inst: StageInstance, now_s: float) -> float:
        total = self.running_remaining_work_s(inst, now_s)
        for req_id, _attempt_id in inst.queue:
            total += self.estimate_stage_service_time_s(self.requests[req_id], inst, now_s)
        return total

    def running_remaining_work_s(self, inst: StageInstance, now_s: float) -> float:
        if inst.current_run is None or inst.current_start_time_s is None:
            return 0.0
        if inst.stage == STAGE_PE and inst.current_run.pe_execution is not None:
            generated = inst.current_run.pe_execution.generated_after_elapsed(now_s - inst.current_start_time_s)
            return self.estimate_pe_service_time_s(self.requests[inst.current_req_id], inst, generated, now_s)
        return max(0.0, inst.current_start_time_s + inst.current_run.total_duration_s - now_s)

    def pe_output_bins(self) -> list[int]:
        return sorted({self.config.short_output_tokens, self.config.long_output_tokens})

    def classify_output_bin(self, avg_output_tokens: float) -> int:
        bins = self.pe_output_bins()
        if len(bins) == 1:
            return bins[0]
        for lower, upper in zip(bins, bins[1:]):
            midpoint = (lower + upper) / 2.0
            if avg_output_tokens < midpoint:
                return lower
        return bins[-1]

    def pe_monitor_samples(
        self,
        now_s: float,
        *,
        instance_id: str | None = None,
    ) -> list[tuple[float, int, int, str]]:
        self.refresh_pe_monitor_window(now_s)
        if instance_id is None:
            return list(self.pe_output_window)
        return [sample for sample in self.pe_output_window if sample[3] == instance_id]

    def refresh_pe_monitor_window(self, now_s: float) -> None:
        if now_s + EPS < self.pe_window_now_s:
            self.reset_pe_monitor_window()
        lo = now_s - self.monitor_window_sec
        while self.pe_window_loaded_index < len(self.pe_output_log):
            sample = self.pe_output_log[self.pe_window_loaded_index]
            if sample[0] > now_s + EPS:
                break
            if sample[0] > lo + EPS:
                self.add_pe_window_sample(sample)
            self.pe_window_loaded_index += 1
        while self.pe_output_window and self.pe_output_window[0][0] <= lo + EPS:
            self.remove_pe_window_sample(self.pe_output_window.popleft())
        self.pe_window_now_s = now_s

    def reset_pe_monitor_window(self) -> None:
        self.pe_output_window.clear()
        self.pe_window_tokens_by_instance.clear()
        self.pe_window_counts_by_instance.clear()
        self.pe_window_total_tokens = 0
        self.pe_window_total_count = 0
        self.pe_window_loaded_index = 0
        self.pe_window_now_s = float("-inf")

    def add_pe_window_sample(self, sample: tuple[float, int, int, str]) -> None:
        _time_s, tokens, _req_id, instance_id = sample
        self.pe_output_window.append(sample)
        self.pe_window_tokens_by_instance[instance_id] += tokens
        self.pe_window_counts_by_instance[instance_id] += 1
        self.pe_window_total_tokens += tokens
        self.pe_window_total_count += 1

    def remove_pe_window_sample(self, sample: tuple[float, int, int, str]) -> None:
        _time_s, tokens, _req_id, instance_id = sample
        self.pe_window_tokens_by_instance[instance_id] -= tokens
        self.pe_window_counts_by_instance[instance_id] -= 1
        if self.pe_window_tokens_by_instance[instance_id] <= 0:
            del self.pe_window_tokens_by_instance[instance_id]
        if self.pe_window_counts_by_instance[instance_id] <= 0:
            del self.pe_window_counts_by_instance[instance_id]
        self.pe_window_total_tokens -= tokens
        self.pe_window_total_count -= 1

    def pe_monitor_avg_output_tokens(self, now_s: float, instance_id: str | None = None) -> float | None:
        self.refresh_pe_monitor_window(now_s)
        if instance_id is None:
            count = self.pe_window_total_count
            tokens = self.pe_window_total_tokens
        else:
            count = self.pe_window_counts_by_instance[instance_id]
            tokens = self.pe_window_tokens_by_instance[instance_id]
        if count <= 0:
            return None
        return tokens / count

    def pe_estimated_output_bin(
        self,
        inst: StageInstance,
        now_s: float,
        generated_tokens: int = 0,
    ) -> int:
        avg_tokens = self.pe_monitor_avg_output_tokens(now_s, inst.instance_id)
        if avg_tokens is None:
            avg_tokens = self.pe_monitor_avg_output_tokens(now_s, None)
        if avg_tokens is None:
            estimated = self.config.long_output_tokens if self.current_target == "long" else self.config.short_output_tokens
        else:
            estimated = self.classify_output_bin(avg_tokens)
        for output_tokens in self.pe_output_bins():
            if generated_tokens <= output_tokens:
                return max(estimated, output_tokens)
        return self.pe_output_bins()[-1]

    def pe_instance_tokens_per_s(self, inst: StageInstance, now_s: float) -> float:
        self.refresh_pe_monitor_window(now_s)
        instance_tokens = self.pe_window_tokens_by_instance[inst.instance_id]
        if instance_tokens > 0:
            return max(instance_tokens / self.monitor_window_sec, EPS)
        if self.pe_window_total_count > 0:
            sampled_instance_count = len(self.pe_window_counts_by_instance)
            return max(
                self.pe_window_total_tokens / self.monitor_window_sec / max(1, sampled_instance_count),
                EPS,
            )
        output_tokens = self.pe_estimated_output_bin(inst, now_s)
        profile_s = self.estimate_pe_service_time_for_output_s(
            Request(-1, 0.0, self.config.input_tokens, output_tokens, "Estimate", 0.0, 0.0),
            inst,
            output_tokens,
            0,
        )
        return max(output_tokens / profile_s, EPS)

    def pe_remaining_tokens_for_work(self, inst: StageInstance, now_s: float) -> int:
        if inst.current_req_id is None or inst.current_run is None or inst.current_run.pe_execution is None:
            return 0
        req = self.requests[inst.current_req_id]
        generated = inst.current_run.pe_execution.generated_after_elapsed(now_s - inst.current_start_time_s)
        output_tokens = self.pe_output_tokens_for_estimate(req, inst, now_s, generated)
        return max(0, output_tokens - generated)

    def pe_waiting_tokens_for_work(self, inst: StageInstance, now_s: float) -> int:
        if self.config.pe_output_estimate_mode == PE_OUTPUT_ESTIMATE_MODE_ORACLE:
            return sum(self.requests[req_id].output_tokens for req_id, _attempt_id in inst.queue)
        output_tokens = self.pe_estimated_output_bin(inst, now_s)
        return len(inst.queue) * output_tokens

    def handle_monitor(self, now_s: float) -> None:
        if self.mode != MODE_CAN_FLIP:
            return
        self.refresh_pe_monitor_window(now_s)
        if self.pe_window_total_count:
            avg_tokens = self.pe_window_total_tokens / self.pe_window_total_count
            decision = self.monitor_decision(avg_tokens)
            if decision in {"short_to_long", "long_to_short"}:
                self.begin_flip(decision, now_s, avg_tokens, self.pe_window_total_count, decision)
        if any(req.finish_time_s is None for req in self.requests):
            self.schedule_event(now_s + self.monitor_window_sec, 3, "monitor", {})

    def threshold_bounds(self) -> tuple[float, float, float]:
        mid = self.config.token_threshold
        token_range = self.config.long_output_tokens - self.config.short_output_tokens
        delta = self.config.threshold_margin_ratio * token_range
        if self.config.threshold_margin_ratio <= 0:
            return mid, mid, mid
        return mid, mid - delta, mid + delta

    def monitor_decision(self, avg_tokens: float) -> str:
        mid, low, high = self.threshold_bounds()
        if self.config.threshold_margin_ratio <= 0:
            return "short_to_long" if avg_tokens > mid else "long_to_short"
        if self.config.threshold_margin_mode == MARGIN_MODE_SINGLE:
            return "short_to_long" if avg_tokens > high else "long_to_short"
        if self.config.threshold_margin_mode != MARGIN_MODE_HYSTERESIS:
            raise ValueError(f"Unsupported threshold margin mode: {self.config.threshold_margin_mode}")
        if self.current_target == "short" and avg_tokens > high:
            return "short_to_long"
        if self.current_target == "long" and avg_tokens < low:
            return "long_to_short"
        return "deadband_keep_current"

    def begin_flip(
        self,
        direction: str,
        now_s: float,
        avg_tokens: float,
        window_count: int,
        monitor_decision: str,
    ) -> None:
        if self.launching_direction is not None:
            return
        if direction == "short_to_long":
            if self.current_target == "long":
                return
            selection = self.select_bundles_for_flip(now_s)
            selected_bundles = list(selection.bundles)
            selection_score = selection.score
            target = "long"
        elif direction == "long_to_short":
            if self.current_target == "short":
                return
            selected_bundles = [self.bundles[key] for key in self.converted_bundle_keys]
            selection_score = self.score_flip_bundle_combination(selected_bundles, direction, now_s)
            target = "short"
        else:
            raise ValueError(direction)
        if not selected_bundles:
            return

        threshold_mid, threshold_low, threshold_high = self.threshold_bounds()
        trace_change = self.find_trace_change_time(direction, now_s)
        flip = FlipEvent(
            flip_id=len(self.flip_events),
            mode=self.mode,
            request_rate_per_min=self.request_rate_per_min,
            monitor_window_sec=self.monitor_window_sec,
            direction=direction,
            detect_time_s=now_s,
            avg_output_tokens=avg_tokens,
            window_count=window_count,
            threshold_mid=threshold_mid,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            margin_ratio=self.config.threshold_margin_ratio,
            margin_label=margin_label(self.config.threshold_margin_ratio),
            monitor_decision=monitor_decision,
            trace_change_time_s=trace_change,
            detection_delay_s=(now_s - trace_change) if trace_change is not None else None,
            selected_bundle_keys=[bundle.bundle_key for bundle in selected_bundles],
            selected_instance_ids=[],
            selection_restart_ready_delay_s=selection_score.restart_ready_delay_s,
            selection_complement_added_work_s=selection_score.complement_added_work_s,
            selection_running_added_work_s=selection_score.running_added_work_s,
            selection_waiting_added_work_s=selection_score.waiting_added_work_s,
            selection_pe_bin_tokens=selection_score.pe_bin_tokens,
            selection_migrated_request_count=selection_score.migrated_request_count,
            selection_template_penalty=selection_score.template_penalty,
            selection_round_robin_rank=selection_score.round_robin_rank,
        )
        self.flip_events.append(flip)
        self.launching_direction = direction
        self.current_target = target

        if direction == "short_to_long":
            drain_times = self.prepare_short_to_long(selected_bundles, flip, now_s)
        else:
            drain_times = self.prepare_long_to_short(selected_bundles, flip, now_s)
        drain_done = max(drain_times) if drain_times else now_s
        flip.drain_done_time_s = drain_done
        self.schedule_event(drain_done, 2, "cold_start_begin", {"flip_id": flip.flip_id})

    def select_bundles_for_flip(self, now_s: float = 0.0) -> FlipSelection:
        candidates = [
            bundle
            for bundle in self.bundles.values()
            if bundle.active
            and not bundle.converted_to_pe
            and not bundle.draining
            and bundle.gpu_type == "A800"
            and bundle.bundle_size == 8
            and bundle.template_name in {TEMPLATE_DIT_ONLY, TEMPLATE_VAE_ONLY, TEMPLATE_DIT_VAE_COMB}
        ]
        if len(candidates) < self.config.flip_source_count:
            raise RuntimeError(
                f"Need {self.config.flip_source_count} flip sources but only found {len(candidates)} candidates"
            )
        selected = self.greedy_select_flip_bundles(candidates, "short_to_long", now_s)
        self.flip_round_robin_cursor = (self.flip_round_robin_cursor + self.config.flip_source_count) % len(candidates)
        return FlipSelection(
            bundles=tuple(sorted(selected, key=lambda bundle: bundle.order)),
            score=self.score_flip_bundle_combination(selected, "short_to_long", now_s),
        )

    def greedy_select_flip_bundles(
        self,
        candidates: list[BundleState],
        direction: str,
        now_s: float,
    ) -> list[BundleState]:
        count = self.config.flip_source_count
        ordered = sorted(
            candidates,
            key=lambda bundle: (self.restart_ready_delay_s([bundle], direction, now_s), bundle.order),
        )
        if len(ordered) <= count:
            return ordered
        cutoff = self.restart_ready_delay_s([ordered[count - 1]], direction, now_s)
        selected = [
            bundle for bundle in ordered if self.restart_ready_delay_s([bundle], direction, now_s) < cutoff - EPS
        ]
        tie_set = [
            bundle for bundle in ordered if abs(self.restart_ready_delay_s([bundle], direction, now_s) - cutoff) <= EPS
        ]
        needed = count - len(selected)
        if needed >= len(tie_set):
            return selected + tie_set

        complement_sorted = sorted(
            tie_set,
            key=lambda bundle: (
                self.complement_added_work_score([bundle], direction, now_s).complement_added_work_s,
                bundle.order,
            ),
        )
        cutoff_work = self.complement_added_work_score(
            [complement_sorted[needed - 1]],
            direction,
            now_s,
        ).complement_added_work_s
        selected_work = [
            bundle
            for bundle in complement_sorted
            if self.complement_added_work_score([bundle], direction, now_s).complement_added_work_s
            < cutoff_work - EPS
        ]
        work_tie = [
            bundle
            for bundle in complement_sorted
            if abs(
                self.complement_added_work_score([bundle], direction, now_s).complement_added_work_s - cutoff_work
            )
            <= EPS
        ]
        remaining = needed - len(selected_work)
        if remaining >= len(work_tie):
            return selected + selected_work + work_tie

        candidates_by_template = sorted(
            work_tie,
            key=lambda bundle: (
                self.template_penalty(bundle),
                self.round_robin_rank(bundle, candidates),
                bundle.order,
            ),
        )
        return selected + selected_work + candidates_by_template[:remaining]

    def score_flip_bundle_combination(
        self,
        bundles: Iterable[BundleState],
        direction: str,
        now_s: float,
    ) -> FlipSelectionScore:
        selected_bundles = tuple(sorted(bundles, key=lambda bundle: bundle.order))
        complement_score = self.complement_added_work_score(selected_bundles, direction, now_s)
        return FlipSelectionScore(
            restart_ready_delay_s=self.restart_ready_delay_s(selected_bundles, direction, now_s),
            complement_added_work_s=complement_score.complement_added_work_s,
            running_added_work_s=complement_score.running_added_work_s,
            waiting_added_work_s=complement_score.waiting_added_work_s,
            pe_bin_tokens=complement_score.pe_bin_tokens,
            migrated_request_count=complement_score.migrated_request_count,
            template_penalty=sum(self.template_penalty(bundle) for bundle in selected_bundles),
            round_robin_rank=sum(
                self.round_robin_rank(bundle, list(selected_bundles)) for bundle in selected_bundles
            ),
            bundle_order_key=tuple(bundle.order for bundle in selected_bundles),
        )

    def restart_ready_delay_s(
        self,
        bundles: Iterable[BundleState],
        direction: str,
        now_s: float,
    ) -> float:
        boundary_delay = 0.0
        for source in self.flip_source_instances(bundles, direction):
            boundary_time = self.current_boundary_time(source, now_s, finish_vae=direction == "short_to_long")
            boundary_delay = max(boundary_delay, max(0.0, boundary_time - now_s))
        cold_start_s = self.flip_cold_start_s(direction)
        return boundary_delay + cold_start_s

    def flip_cold_start_s(self, direction: str) -> float:
        if self.config.cold_start_mode == COLD_START_MODE_ZERO:
            return 0.0
        if self.config.cold_start_mode != COLD_START_MODE_PROFILE:
            raise ValueError(f"Unsupported cold start mode: {self.config.cold_start_mode}")
        if direction == "short_to_long":
            return self.pe_init_time_s("A800", 2)
        if direction == "long_to_short":
            return self.generator_init_time_s(8)
        raise ValueError(direction)

    def complement_added_work_score(
        self,
        bundles: Iterable[BundleState],
        direction: str,
        now_s: float,
    ) -> FlipSelectionScore:
        selected_bundles = tuple(sorted(bundles, key=lambda bundle: bundle.order))
        stage_running: Counter[str] = Counter()
        stage_waiting: Counter[str] = Counter()
        migrated_request_count = 0
        pe_bin_tags: list[str] = []

        for source in self.flip_source_instances(selected_bundles, direction):
            if source.stage == STAGE_PE:
                running_tokens = self.pe_remaining_tokens_for_work(source, now_s)
                waiting_tokens = self.pe_waiting_tokens_for_work(source, now_s)
                stage_running[STAGE_PE] += running_tokens
                stage_waiting[STAGE_PE] += waiting_tokens
                pe_bin_tags.append(f"{source.instance_id}:{self.pe_estimated_output_bin(source, now_s)}")
                migrated_request_count += len(source.queue) + (1 if running_tokens > 0 else 0)
            elif source.stage == STAGE_DIT:
                running_units = self.dit_running_units_for_work(source, now_s)
                stage_running[STAGE_DIT] += running_units
                stage_waiting[STAGE_DIT] += len(source.queue)
                migrated_request_count += len(source.queue) + (1 if running_units > 0 else 0)
            elif source.stage == STAGE_VAE:
                running_units = self.vae_running_units_for_work(source, now_s)
                stage_running[STAGE_VAE] += running_units
                stage_waiting[STAGE_VAE] += len(source.queue)
                migrated_request_count += len(source.queue) + (1 if running_units > 0 else 0)

        selected_ids = {inst.instance_id for inst in self.flip_source_instances(selected_bundles, direction)}
        stage_totals: dict[str, tuple[float, float]] = {}
        for stage in (STAGE_PE, STAGE_DIT, STAGE_VAE):
            running_s = self.stage_work_units_to_seconds(stage, stage_running[stage], selected_ids, now_s)
            waiting_s = self.stage_work_units_to_seconds(stage, stage_waiting[stage], selected_ids, now_s)
            if running_s or waiting_s:
                stage_totals[stage] = (running_s, waiting_s)

        if STAGE_DIT in stage_totals and STAGE_VAE in stage_totals:
            complement_added = max(sum(stage_totals[STAGE_DIT]), sum(stage_totals[STAGE_VAE]))
        else:
            complement_added = sum(running_s + waiting_s for running_s, waiting_s in stage_totals.values())
        return FlipSelectionScore(
            restart_ready_delay_s=0.0,
            complement_added_work_s=complement_added,
            running_added_work_s=sum(running_s for running_s, _waiting_s in stage_totals.values()),
            waiting_added_work_s=sum(waiting_s for _running_s, waiting_s in stage_totals.values()),
            pe_bin_tokens=";".join(pe_bin_tags),
            migrated_request_count=migrated_request_count,
            template_penalty=sum(self.template_penalty(bundle) for bundle in selected_bundles),
            round_robin_rank=0,
        )

    def stage_work_units_to_seconds(
        self,
        stage: str,
        units: float,
        selected_ids: set[str],
        now_s: float,
    ) -> float:
        if units <= 0:
            return 0.0
        if stage == STAGE_PE:
            throughput = sum(
                self.pe_instance_tokens_per_s(inst, now_s)
                for inst in self.stage_instances[STAGE_PE]
                if inst.instance_id not in selected_ids and inst.active_ready()
            )
        else:
            throughput = sum(
                1.0 / max(inst.base_latency_s, EPS)
                for inst in self.stage_instances[stage]
                if inst.instance_id not in selected_ids and inst.active_ready()
            )
        if throughput <= 0:
            return float("inf")
        return units / throughput

    def dit_running_units_for_work(self, inst: StageInstance, now_s: float) -> float:
        if inst.current_req_id is None or inst.current_run is None or inst.current_start_time_s is None:
            return 0.0
        boundary_time = self.current_boundary_time(inst, now_s, finish_vae=False)
        elapsed = max(0.0, boundary_time - inst.current_start_time_s)
        completed = inst.current_run.dit_completed_after_elapsed(elapsed, self.config.denoising_steps)
        return max(0, self.config.denoising_steps - completed) / self.config.denoising_steps

    def vae_running_units_for_work(self, inst: StageInstance, now_s: float) -> float:
        if inst.current_req_id is None or inst.current_run is None or inst.current_start_time_s is None:
            return 0.0
        remaining_s = max(0.0, inst.current_start_time_s + inst.current_run.total_duration_s - now_s)
        return remaining_s / max(inst.base_latency_s, EPS)

    def template_penalty(self, bundle: BundleState) -> int:
        return 1 if bundle.template_name == TEMPLATE_DIT_VAE_COMB else 0

    def round_robin_rank(self, bundle: BundleState, candidates: list[BundleState]) -> int:
        ordered = sorted(candidates, key=lambda candidate: candidate.order)
        if not ordered:
            return 0
        index_by_key = {candidate.bundle_key: idx for idx, candidate in enumerate(ordered)}
        return (index_by_key[bundle.bundle_key] - self.flip_round_robin_cursor) % len(ordered)

    def flip_source_instances(
        self,
        bundles: Iterable[BundleState],
        direction: str,
    ) -> list[StageInstance]:
        sources: list[StageInstance] = []
        for bundle in bundles:
            if direction == "short_to_long":
                sources.extend(sorted(bundle.stage_instances.values(), key=lambda inst: inst.order))
            elif direction == "long_to_short":
                sources.extend(
                    inst
                    for inst in sorted(bundle.pe_target_instances, key=lambda candidate: candidate.order)
                    if inst.active
                )
            else:
                raise ValueError(direction)
        return sources

    def estimate_running_migration_service_time_s(
        self,
        source: StageInstance,
        target: StageInstance,
        boundary_time_s: float,
    ) -> float:
        if source.current_req_id is None or source.current_run is None or source.current_start_time_s is None:
            return 0.0
        req = self.requests[source.current_req_id]
        elapsed_s = max(0.0, boundary_time_s - source.current_start_time_s)
        if source.stage == STAGE_PE and source.current_run.pe_execution is not None:
            generated = source.current_run.pe_execution.generated_after_elapsed(elapsed_s)
            return self.estimate_pe_service_time_s(req, target, generated, boundary_time_s)
        if source.stage == STAGE_DIT:
            completed_steps = source.current_run.dit_completed_after_elapsed(
                elapsed_s,
                self.config.denoising_steps,
            )
            return self.estimate_dit_service_time_s(req, target, completed_steps)
        return self.estimate_stage_service_time_s(req, target, boundary_time_s)

    def prepare_short_to_long(
        self,
        bundles: list[BundleState],
        flip: FlipEvent,
        now_s: float,
    ) -> list[float]:
        drain_times = [now_s]
        for bundle in bundles:
            bundle.draining = True
            for inst in bundle.stage_instances.values():
                inst.draining = True
                flip.selected_instance_ids.append(inst.instance_id)
        for bundle in bundles:
            for inst in bundle.stage_instances.values():
                flip.migrated_waiting_requests += self.migrate_waiting(inst, now_s)
                if inst.current_req_id is None:
                    continue
                drain_time = self.current_boundary_time(inst, now_s, finish_vae=True)
                drain_times.append(drain_time)
                if inst.stage != STAGE_VAE:
                    self.schedule_event(
                        drain_time,
                        1,
                        "source_drain",
                        {"instance_id": inst.instance_id, "flip_id": flip.flip_id},
                    )
        return drain_times

    def prepare_long_to_short(
        self,
        bundles: list[BundleState],
        flip: FlipEvent,
        now_s: float,
    ) -> list[float]:
        drain_times = [now_s]
        for bundle in bundles:
            for inst in bundle.pe_target_instances:
                if not inst.active:
                    continue
                inst.draining = True
                flip.selected_instance_ids.append(inst.instance_id)
        for bundle in bundles:
            for inst in bundle.pe_target_instances:
                if not inst.active:
                    continue
                flip.migrated_waiting_requests += self.migrate_waiting(inst, now_s)
                if inst.current_req_id is None:
                    continue
                drain_time = self.current_boundary_time(inst, now_s, finish_vae=False)
                drain_times.append(drain_time)
                self.schedule_event(
                    drain_time,
                    1,
                    "source_drain",
                    {"instance_id": inst.instance_id, "flip_id": flip.flip_id},
                )
        return drain_times

    def current_boundary_time(self, inst: StageInstance, now_s: float, *, finish_vae: bool) -> float:
        if inst.current_run is None or inst.current_start_time_s is None:
            return now_s
        if inst.stage == STAGE_PE and inst.current_run.pe_execution is not None:
            return inst.current_run.pe_execution.next_step_boundary_time(inst.current_start_time_s, now_s)
        if inst.stage == STAGE_DIT:
            return inst.current_run.dit_next_step_boundary_time(inst.current_start_time_s, now_s)
        if inst.stage == STAGE_VAE and finish_vae:
            return inst.current_start_time_s + inst.current_run.total_duration_s
        return now_s

    def migrate_waiting(self, inst: StageInstance, now_s: float) -> int:
        if not inst.queue:
            return 0
        targets = [
            candidate
            for candidate in self.stage_instances[inst.stage]
            if candidate is not inst and candidate.active_ready()
        ]
        estimated = self.estimated_work_scores(targets, now_s)
        migrated = 0
        while inst.queue:
            req_id, attempt_id = inst.queue.popleft()
            attempt = self.attempts[attempt_id]
            attempt.exit_time_s = now_s
            attempt.exit_reason = "migrated_waiting"
            req = self.requests[req_id]
            target = self.choose_least_waiting(inst.stage, targets, estimated, incoming_req=req, now_s=now_s)
            estimated[target.instance_id] += self.incoming_load_increment(targets, target, req, now_s)
            self.enqueue_request(self.requests[req_id], target, now_s, queue_insert="tail_migrated", at_head=False)
            self.start_next_if_possible(target, now_s)
            migrated += 1
        return migrated

    def handle_source_drain(self, instance_id: str, flip_id: int, now_s: float) -> None:
        inst = self.instance_by_id(instance_id)
        if inst.current_req_id is None:
            return
        flip = self.flip_events[flip_id]
        req = self.requests[inst.current_req_id]
        self.update_running_progress(inst, req, now_s)
        attempt = self.attempts[inst.current_attempt_id] if inst.current_attempt_id is not None else None
        if attempt is not None:
            attempt.exit_time_s = now_s
            attempt.exit_reason = "migrated_running"
        old_req_id = inst.current_req_id
        inst.current_req_id = None
        inst.current_attempt_id = None
        inst.current_start_time_s = None
        inst.current_run = None
        inst.generation += 1

        targets = [
            candidate
            for candidate in self.stage_instances[inst.stage]
            if candidate is not inst and candidate.active_ready()
        ]
        estimated = self.estimated_work_scores(targets, now_s)
        target = self.choose_least_waiting(
            inst.stage,
            targets,
            estimated,
            incoming_req=self.requests[old_req_id],
            now_s=now_s,
        )
        self.enqueue_request(self.requests[old_req_id], target, now_s, queue_insert="head_migrated", at_head=True)
        self.start_next_if_possible(target, now_s)
        flip.migrated_running_requests += 1

    def update_running_progress(self, inst: StageInstance, req: Request, now_s: float) -> None:
        if inst.current_run is None or inst.current_start_time_s is None:
            return
        elapsed = max(0.0, now_s - inst.current_start_time_s)
        if inst.stage == STAGE_PE and inst.current_run.pe_execution is not None:
            req.pe_generated_tokens = inst.current_run.pe_execution.generated_after_elapsed(elapsed)
        elif inst.stage == STAGE_DIT:
            req.dit_completed_steps = inst.current_run.dit_completed_after_elapsed(
                elapsed,
                self.config.denoising_steps,
            )

    def handle_cold_start_begin(self, flip_id: int, now_s: float) -> None:
        flip = self.flip_events[flip_id]
        flip.cold_start_time_s = now_s
        target_instances: list[StageInstance] = []
        if flip.direction == "short_to_long":
            for bundle_key in flip.selected_bundle_keys:
                bundle = self.bundles[bundle_key]
                for inst in bundle.stage_instances.values():
                    inst.active = False
                    inst.ready = False
                    inst.draining = False
                    inst.launching = False
                    inst.generation += 1
                bundle.active = False
                bundle.converted_to_pe = True
                bundle.draining = False
                if bundle.bundle_key not in self.converted_bundle_keys:
                    self.converted_bundle_keys.append(bundle.bundle_key)
                for pe_inst in self._ensure_pe_targets(bundle):
                    pe_inst.active = True
                    pe_inst.ready = False
                    pe_inst.launching = True
                    pe_inst.draining = False
                    pe_inst.generation += 1
                    target_instances.append(pe_inst)
            cold_start_s = self.flip_cold_start_s(flip.direction)
        else:
            for bundle_key in flip.selected_bundle_keys:
                bundle = self.bundles[bundle_key]
                for pe_inst in bundle.pe_target_instances:
                    pe_inst.active = False
                    pe_inst.ready = False
                    pe_inst.launching = False
                    pe_inst.draining = False
                    pe_inst.generation += 1
                for inst in bundle.stage_instances.values():
                    inst.active = True
                    inst.ready = False
                    inst.launching = True
                    inst.draining = False
                    inst.generation += 1
                    target_instances.append(inst)
                bundle.active = True
                bundle.converted_to_pe = False
                bundle.draining = False
                if bundle.bundle_key in self.converted_bundle_keys:
                    self.converted_bundle_keys.remove(bundle.bundle_key)
            cold_start_s = self.flip_cold_start_s(flip.direction)

        flip.cold_start_done_time_s = now_s + cold_start_s
        flip.cold_start_time_s = now_s
        flip.launch_target_instance_ids = [inst.instance_id for inst in target_instances]
        self.schedule_event(now_s + cold_start_s, -1, "launch_done", {"flip_id": flip_id})

    def handle_launch_done(self, flip_id: int, now_s: float) -> None:
        flip = self.flip_events[flip_id]
        for instance_id in flip.launch_target_instance_ids:
            inst = self.instance_by_id(instance_id)
            inst.ready = True
            inst.launching = False
            inst.generation += 1
            self.start_next_if_possible(inst, now_s)
        self.launching_direction = None

    def pe_init_time_s(self, gpu_type: str, parallelism: int) -> float:
        hardware = self.catalog.profile_data.pe_models[self.config.pe_model].hardware[gpu_type]
        if self.config.pe_init_time_mode == PE_INIT_TIME_MODE_OPTIMIZED:
            table = hardware.init_time_s
            label = "optimized"
        elif self.config.pe_init_time_mode == PE_INIT_TIME_MODE_NON_OPTIMIZED:
            table = hardware.non_optimized_init_time_s
            label = "non-optimized"
        else:
            raise ValueError(f"Unsupported PE init time mode: {self.config.pe_init_time_mode}")
        if parallelism not in table:
            raise KeyError(f"Missing {label} PE init time for {self.config.pe_model} {gpu_type} P{parallelism}")
        return table[parallelism]

    def generator_init_time_s(self, parallelism: int) -> float:
        generator = self.catalog.generator_profile(self.config.generator_model)
        if self.config.generator_init_time_mode == GENERATOR_INIT_TIME_MODE_PROFILE:
            table = generator.init_time_s
            label = "profile"
        elif self.config.generator_init_time_mode == GENERATOR_INIT_TIME_MODE_OPTIMIZED:
            table = generator.optimized_init_time_s
            label = "optimized"
        else:
            raise ValueError(f"Unsupported generator init time mode: {self.config.generator_init_time_mode}")
        if parallelism not in table:
            raise KeyError(f"Missing {label} generator init time for {generator.name} P{parallelism}")
        return table[parallelism]

    def cold_start_variant_label(self) -> str:
        if (
            self.config.pe_init_time_mode == PE_INIT_TIME_MODE_NON_OPTIMIZED
            and self.config.generator_init_time_mode == GENERATOR_INIT_TIME_MODE_PROFILE
        ):
            return "non_optimized"
        if (
            self.config.pe_init_time_mode == PE_INIT_TIME_MODE_OPTIMIZED
            and self.config.generator_init_time_mode == GENERATOR_INIT_TIME_MODE_OPTIMIZED
        ):
            return "optimized"
        return f"pe_{self.config.pe_init_time_mode}_gen_{self.config.generator_init_time_mode}"

    def find_trace_change_time(self, direction: str, detect_time_s: float) -> float | None:
        want = ("Short", "Long") if direction == "short_to_long" else ("Long", "Short")
        previous = None
        candidate = None
        for interval in self.intervals:
            if previous is None:
                if interval.resolved_dominant == want[1] and interval.start_s <= detect_time_s:
                    candidate = interval.start_s
            elif previous.resolved_dominant == want[0] and interval.resolved_dominant == want[1]:
                if interval.start_s <= detect_time_s:
                    candidate = interval.start_s
            previous = interval
        return candidate

    def compute_summary(self) -> dict[str, object]:
        finished = [req for req in self.requests if req.finish_time_s is not None]
        first_arrival = min(req.arrival_time_s for req in self.requests) if self.requests else 0.0
        last_finish = max(req.finish_time_s for req in finished) if finished else float("nan")
        makespan = last_finish - first_arrival if finished else float("nan")
        latencies = [req.finish_time_s - req.arrival_time_s for req in finished if req.finish_time_s is not None]
        slo_baselines = self.compute_slo_baselines()
        slo5 = [
            req.finish_time_s - req.arrival_time_s <= slo_baselines[req.output_tokens] * 5
            for req in finished
            if req.finish_time_s is not None
        ]
        slo10 = [
            req.finish_time_s - req.arrival_time_s <= slo_baselines[req.output_tokens] * 10
            for req in finished
            if req.finish_time_s is not None
        ]
        summary = {
            "run_id": self.run_id,
            "mode": self.mode,
            "margin_ratio": self.config.threshold_margin_ratio,
            "margin_label": margin_label(self.config.threshold_margin_ratio),
            "threshold_margin_mode": self.config.threshold_margin_mode,
            "cold_start_mode": self.config.cold_start_mode,
            "pe_output_estimate_mode": self.config.pe_output_estimate_mode,
            "pe_init_time_mode": self.config.pe_init_time_mode,
            "generator_init_time_mode": self.config.generator_init_time_mode,
            "cold_start_variant": self.cold_start_variant_label(),
            "threshold_mid": self.threshold_bounds()[0],
            "threshold_low": self.threshold_bounds()[1],
            "threshold_high": self.threshold_bounds()[2],
            "request_rate_per_min": self.request_rate_per_min,
            "monitor_window_sec": self.monitor_window_sec,
            "total_requests": len(self.requests),
            "finished_requests": len(finished),
            "first_arrival_s": first_arrival,
            "last_finish_s": last_finish,
            "makespan_s": makespan,
            "throughput_req_s": len(finished) / makespan if makespan > 0 else float("nan"),
            "num_flip_events": len(self.flip_events),
            "latency_mean_s": statistics.fmean(latencies) if latencies else float("nan"),
            "latency_p50_s": percentile(latencies, 50),
            "latency_p90_s": percentile(latencies, 90),
            "latency_p99_s": percentile(latencies, 99),
            "slo5_pass_ratio": sum(slo5) / len(slo5) if slo5 else float("nan"),
            "slo10_pass_ratio": sum(slo10) / len(slo10) if slo10 else float("nan"),
        }
        for output_tokens, baseline_s in sorted(slo_baselines.items()):
            summary[f"slo_baseline_{output_tokens}_s"] = baseline_s
        return summary

    def compute_slo_baselines(self) -> dict[int, float]:
        generator = self.catalog.generator_profile(self.config.generator_model)
        pe_hardware = self.catalog.profile_data.pe_models[self.config.pe_model].hardware["A800"]
        dit_weighted = weighted_latency(
            [
                (4, generator.dit["H100"].latency_s[8]),
                (19, generator.dit["A800"].latency_s[8]),
            ]
        )
        vae_weighted = weighted_latency(
            [
                (4, generator.vae["H100"].latency_s[8]),
                (19, generator.vae["A800"].latency_s[8]),
            ]
        )
        out: dict[int, float] = {}
        for output_tokens in (self.config.short_output_tokens, self.config.long_output_tokens):
            ttft = pe_hardware.ttft_ms[self.config.input_tokens][output_tokens][2]
            tpot = pe_hardware.tpot_ms[self.config.input_tokens][output_tokens][2]
            pe_latency = (ttft + (output_tokens - 1) * tpot) / 1000.0
            out[output_tokens] = pe_latency + generator.encoder_cpu_latency_s + dit_weighted + vae_weighted
        return out


def weighted_latency(items: list[tuple[int, float]]) -> float:
    total = sum(count for count, _latency in items)
    if total <= 0:
        return float("nan")
    return sum(count * latency for count, latency in items) / total


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def run_suite(config: SimulationConfig) -> tuple[list[RunResult], list[DominantInterval]]:
    catalog = ProfileCatalog()
    intervals = smoke_intervals(config.short_output_tokens, config.long_output_tokens)
    if config.scenario != "smoke":
        intervals = load_dominant_intervals(
            config.dominant_intervals_csv,
            short_output_tokens=config.short_output_tokens,
            long_output_tokens=config.long_output_tokens,
        )
    modes = [MODE_NO_FLIP, MODE_CAN_FLIP] if config.mode == "both" else [config.mode]
    results: list[RunResult] = []
    for rate in config.request_rates_per_min:
        for window in config.monitor_windows_sec:
            for mode in modes:
                sim = FlipSimulator(
                    config=config,
                    catalog=catalog,
                    intervals=intervals,
                    request_rate_per_min=rate,
                    monitor_window_sec=window,
                    mode=mode,
                )
                requests = build_requests(
                    intervals,
                    rate_per_min=rate,
                    duration_min=config.duration_min,
                    input_tokens=config.input_tokens,
                )
                sim.add_requests(requests)
                results.append(sim.run())
    add_comparison_fields(results)
    return results, intervals


def add_comparison_fields(results: list[RunResult]) -> None:
    baseline_by_case: dict[tuple[float, float], RunResult] = {}
    for result in results:
        result.summary.setdefault("throughput_vs_no_flip_pct", float("nan"))
        result.summary.setdefault("slo10_vs_no_flip_delta", float("nan"))
        result.summary.setdefault("slo5_vs_no_flip_delta", float("nan"))
        if result.mode != MODE_NO_FLIP:
            continue
        key = (result.request_rate_per_min, result.monitor_window_sec)
        if float(result.summary.get("margin_ratio", 0.0)) == 0.0 or key not in baseline_by_case:
            baseline_by_case[key] = result

    for result in results:
        key = (result.request_rate_per_min, result.monitor_window_sec)
        baseline = baseline_by_case.get(key)
        if baseline is None:
            continue
        nf = baseline.summary
        current = result.summary
        if result.mode == MODE_NO_FLIP:
            current["throughput_vs_no_flip_pct"] = 0.0
            current["slo10_vs_no_flip_delta"] = 0.0
            current["slo5_vs_no_flip_delta"] = 0.0
            continue
        if result.mode != MODE_CAN_FLIP:
            continue
        current["throughput_vs_no_flip_pct"] = (
            (float(current["throughput_req_s"]) / float(nf["throughput_req_s"]) - 1.0) * 100.0
        )
        current["slo10_vs_no_flip_delta"] = float(current["slo10_pass_ratio"]) - float(nf["slo10_pass_ratio"])
        current["slo5_vs_no_flip_delta"] = float(current["slo5_pass_ratio"]) - float(nf["slo5_pass_ratio"])


def write_outputs(
    config: SimulationConfig,
    results: list[RunResult],
    intervals: list[DominantInterval],
    *,
    summary_text: str | None = None,
) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(config.out_dir / "run_summary.csv", [result.summary for result in results])
    write_csv(config.out_dir / "request_stage_events.csv", request_stage_rows(results))
    write_csv(config.out_dir / "flip_events.csv", flip_event_rows(results))
    write_csv(config.out_dir / "dominant_timeline.csv", dominant_timeline_rows(intervals, config.trace_start))
    (config.out_dir / "run_summary.json").write_text(
        json.dumps([result.summary for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config.out_dir / "summary.md").write_text(summary_text or render_summary_markdown(results), encoding="utf-8")


def request_stage_rows(results: list[RunResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        requests = {req.req_id: req for req in result.requests}
        slo_baseline_by_output = _slo_baselines_from_summary(result.summary)
        for attempt in result.attempts:
            req = requests[attempt.req_id]
            latency = req.finish_time_s - req.arrival_time_s if req.finish_time_s is not None else float("nan")
            baseline = slo_baseline_by_output.get(req.output_tokens, float("nan"))
            row = asdict(attempt)
            row.update(
                {
                    "run_id": result.run_id,
                    "mode": result.mode,
                    "margin_ratio": result.summary.get("margin_ratio", 0.0),
                    "margin_label": result.summary.get("margin_label", "non_margin"),
                    "request_rate_per_min": result.request_rate_per_min,
                    "monitor_window_sec": result.monitor_window_sec,
                    "request_type": req.request_type,
                    "input_tokens": req.input_tokens,
                    "output_tokens": req.output_tokens,
                    "arrival_time_s": req.arrival_time_s,
                    "finish_time_s": req.finish_time_s,
                    "latency_s": latency,
                    "queue_time_s": (
                        attempt.start_time_s - attempt.enter_time_s
                        if attempt.start_time_s is not None
                        else float("nan")
                    ),
                    "attempt_duration_s": (
                        attempt.exit_time_s - attempt.enter_time_s if attempt.exit_time_s is not None else float("nan")
                    ),
                    "slo_baseline_s": baseline,
                    "slo5_met": latency <= baseline * 5 if math.isfinite(latency) and math.isfinite(baseline) else "",
                    "slo10_met": latency <= baseline * 10 if math.isfinite(latency) and math.isfinite(baseline) else "",
                }
            )
            rows.append(row)
    return rows


def _slo_baselines_from_summary(summary: dict[str, object]) -> dict[int, float]:
    baselines: dict[int, float] = {}
    prefix = "slo_baseline_"
    suffix = "_s"
    for key, value in summary.items():
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        token_text = key[len(prefix) : -len(suffix)]
        if token_text.isdigit():
            baselines[int(token_text)] = float(value)
    return baselines


def flip_event_rows(results: list[RunResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        for event in result.flip_events:
            row = asdict(event)
            row["run_id"] = result.run_id
            row["selected_bundle_keys"] = ",".join(event.selected_bundle_keys)
            row["selected_instance_ids"] = ",".join(event.selected_instance_ids)
            row["launch_target_instance_ids"] = ",".join(event.launch_target_instance_ids)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_summary_markdown(results: list[RunResult]) -> str:
    by_case: dict[tuple[float, float], dict[str, RunResult]] = {}
    for result in results:
        by_case.setdefault((result.request_rate_per_min, result.monitor_window_sec), {})[result.mode] = result
    lines = ["# Flip Simulation Summary", ""]
    lines.append(
        "| rate req/min | window s | no-flip thpt | can-flip thpt | thpt lift % | "
        "no-flip SLOx10 | can-flip SLOx10 | SLOx10 delta | no-flip SLOx5 | can-flip SLOx5 | SLOx5 delta |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rate, window in sorted(by_case):
        case = by_case[(rate, window)]
        nf = case.get(MODE_NO_FLIP)
        cf = case.get(MODE_CAN_FLIP)
        if nf is None or cf is None:
            continue
        lines.append(
            f"| {rate:g} | {window:g} | {float(nf.summary['throughput_req_s']):.6f} | "
            f"{float(cf.summary['throughput_req_s']):.6f} | "
            f"{float(cf.summary.get('throughput_vs_no_flip_pct', float('nan'))):.3f} | "
            f"{float(nf.summary['slo10_pass_ratio']):.6f} | {float(cf.summary['slo10_pass_ratio']):.6f} | "
            f"{float(cf.summary.get('slo10_vs_no_flip_delta', float('nan'))):.6f} | "
            f"{float(nf.summary['slo5_pass_ratio']):.6f} | {float(cf.summary['slo5_pass_ratio']):.6f} | "
            f"{float(cf.summary.get('slo5_vs_no_flip_delta', float('nan'))):.6f} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_non_margin_summary_from_run_summary_csv(
    run_summary_csv: Path,
    *,
    title: str = "Weighted Dispatch Non-Margin Flip Simulation Summary",
) -> str:
    with run_summary_csv.open(newline="", encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row.get("margin_label", "non_margin") == "non_margin"
        ]
    by_case: dict[tuple[float, float], dict[str, dict[str, str]]] = {}
    for row in rows:
        rate = float(row["request_rate_per_min"])
        window = float(row["monitor_window_sec"])
        by_case.setdefault((rate, window), {})[row["mode"]] = row

    lines = [f"# {title}", ""]
    lines.append(
        "This summary uses the same slot meaning as `td_outputs\\flip_simulation_default\\summary.md`: "
        "each row is one `(request rate, monitor window)` case. The numbers are filtered to "
        "`margin_label=non_margin` from the current simulator outputs."
    )
    lines.append("")
    lines.append(f"Source CSV: `{run_summary_csv}`.")
    lines.append("")
    lines.append(
        "| rate req/min | window s | no-flip thpt | can-flip thpt | thpt lift % | "
        "no-flip SLOx10 | can-flip SLOx10 | SLOx10 delta | no-flip SLOx5 | can-flip SLOx5 | SLOx5 delta |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rate, window in sorted(by_case):
        case = by_case[(rate, window)]
        nf = case.get(MODE_NO_FLIP)
        cf = case.get(MODE_CAN_FLIP)
        if nf is None or cf is None:
            continue
        lines.append(
            f"| {rate:g} | {window:g} | {float(nf['throughput_req_s']):.6f} | "
            f"{float(cf['throughput_req_s']):.6f} | "
            f"{float(cf['throughput_vs_no_flip_pct']):.3f} | "
            f"{float(nf['slo10_pass_ratio']):.6f} | {float(cf['slo10_pass_ratio']):.6f} | "
            f"{float(cf['slo10_vs_no_flip_delta']):.6f} | "
            f"{float(nf['slo5_pass_ratio']):.6f} | {float(cf['slo5_pass_ratio']):.6f} | "
            f"{float(cf['slo5_vs_no_flip_delta']):.6f} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("")
    lines.append("- `thpt` is finished requests divided by `last_finish_s - first_arrival_s`.")
    lines.append("- SLO columns use the current weighted SLO baseline implementation.")
    lines.append("- `thpt lift %`, `SLOx10 delta`, and `SLOx5 delta` are can-flip relative to no-flip for the same slot.")
    return "\n".join(lines).rstrip() + "\n"


def render_margin_comparison_markdown(results: list[RunResult], margin_ratio: float) -> str:
    margin_name = margin_label(margin_ratio)
    by_key: dict[tuple[float, float, str, str], RunResult] = {}
    rates = set()
    windows = set()
    for result in results:
        label = str(result.summary.get("margin_label", "non_margin"))
        by_key[(result.request_rate_per_min, result.monitor_window_sec, result.mode, label)] = result
        rates.add(result.request_rate_per_min)
        windows.add(result.monitor_window_sec)

    lines = [f"# Margin Comparison: a={margin_ratio:.2f}", ""]
    lines.append(
        "Each margin column is the can-flip result with hysteresis margin. "
        "Each non-margin column is the can-flip result with `a=0`, using the same no-flip baseline."
    )
    lines.append("")
    for rate in sorted(rates):
        lines.append(f"## Request Rate {rate:g} req/min")
        lines.append("")
        lines.append(
            "| window s | throughput lift non-margin % | throughput lift margin % | "
            "SLOx10 delta non-margin | SLOx10 delta margin | "
            "SLOx5 delta non-margin | SLOx5 delta margin |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for window in sorted(windows):
            non_margin = by_key.get((rate, window, MODE_CAN_FLIP, "non_margin"))
            margin = by_key.get((rate, window, MODE_CAN_FLIP, margin_name))
            if non_margin is None or margin is None:
                continue
            lines.append(
                f"| {window:g} | "
                f"{float(non_margin.summary['throughput_vs_no_flip_pct']):.3f} | "
                f"{float(margin.summary['throughput_vs_no_flip_pct']):.3f} | "
                f"{float(non_margin.summary['slo10_vs_no_flip_delta']):.6f} | "
                f"{float(margin.summary['slo10_vs_no_flip_delta']):.6f} | "
                f"{float(non_margin.summary['slo5_vs_no_flip_delta']):.6f} | "
                f"{float(margin.summary['slo5_vs_no_flip_delta']):.6f} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def add_cold_start_optimization_fields(results: list[RunResult]) -> None:
    baseline_by_case: dict[tuple[float, float, str], RunResult] = {}
    for result in results:
        result.summary.setdefault("cold_start_throughput_lift_pct", float("nan"))
        result.summary.setdefault("cold_start_slo10_delta", float("nan"))
        result.summary.setdefault("cold_start_slo5_delta", float("nan"))
        result.summary.setdefault("cold_start_flip_event_delta", float("nan"))
        if result.mode != MODE_CAN_FLIP:
            continue
        if result.summary.get("cold_start_variant") != "non_optimized":
            continue
        key = (
            result.request_rate_per_min,
            result.monitor_window_sec,
            str(result.summary.get("margin_label", "non_margin")),
        )
        baseline_by_case[key] = result

    for result in results:
        if result.mode != MODE_CAN_FLIP:
            continue
        key = (
            result.request_rate_per_min,
            result.monitor_window_sec,
            str(result.summary.get("margin_label", "non_margin")),
        )
        baseline = baseline_by_case.get(key)
        if baseline is None:
            continue
        if result.summary.get("cold_start_variant") == "non_optimized":
            result.summary["cold_start_throughput_lift_pct"] = 0.0
            result.summary["cold_start_slo10_delta"] = 0.0
            result.summary["cold_start_slo5_delta"] = 0.0
            result.summary["cold_start_flip_event_delta"] = 0
            continue
        result.summary["cold_start_throughput_lift_pct"] = (
            float(result.summary["throughput_req_s"]) / float(baseline.summary["throughput_req_s"]) - 1.0
        ) * 100.0
        result.summary["cold_start_slo10_delta"] = (
            float(result.summary["slo10_pass_ratio"]) - float(baseline.summary["slo10_pass_ratio"])
        )
        result.summary["cold_start_slo5_delta"] = (
            float(result.summary["slo5_pass_ratio"]) - float(baseline.summary["slo5_pass_ratio"])
        )
        result.summary["cold_start_flip_event_delta"] = (
            int(result.summary["num_flip_events"]) - int(baseline.summary["num_flip_events"])
        )


def render_cold_start_optimization_markdown(results: list[RunResult], margin_ratio: float) -> str:
    by_key: dict[tuple[float, float, str], RunResult] = {}
    rates = set()
    windows = set()
    for result in results:
        if result.mode != MODE_CAN_FLIP:
            continue
        variant = str(result.summary.get("cold_start_variant", ""))
        by_key[(result.request_rate_per_min, result.monitor_window_sec, variant)] = result
        rates.add(result.request_rate_per_min)
        windows.add(result.monitor_window_sec)

    lines = [f"# Cold-Start Optimization Comparison: margin a={margin_ratio:.2f}", ""]
    lines.append(
        "Baseline uses non-optimized PE launch time and profile generator launch time. "
        "Optimized uses optimized PE launch time and optimized generator launch time. "
        "Both runs use the same margin, trace, request rate, monitor window, dispatch, and flip plan."
    )
    lines.append("")
    lines.append(
        "Expected cold-start durations: PE A800 P2 `29.624s -> 20.963828s`; "
        "WAN2.2 generator P8 `61.020383s -> 37.61s`."
    )
    lines.append("")
    for rate in sorted(rates):
        lines.append(f"## Request Rate {rate:g} req/min")
        lines.append("")
        lines.append(
            "| window s | baseline thpt | optimized thpt | thpt lift % | "
            "baseline SLOx10 | optimized SLOx10 | SLOx10 delta | "
            "baseline SLOx5 | optimized SLOx5 | SLOx5 delta | "
            "baseline flips | optimized flips | flip delta |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for window in sorted(windows):
            baseline = by_key.get((rate, window, "non_optimized"))
            optimized = by_key.get((rate, window, "optimized"))
            if baseline is None or optimized is None:
                continue
            lines.append(
                f"| {window:g} | "
                f"{float(baseline.summary['throughput_req_s']):.6f} | "
                f"{float(optimized.summary['throughput_req_s']):.6f} | "
                f"{float(optimized.summary['cold_start_throughput_lift_pct']):.3f} | "
                f"{float(baseline.summary['slo10_pass_ratio']):.6f} | "
                f"{float(optimized.summary['slo10_pass_ratio']):.6f} | "
                f"{float(optimized.summary['cold_start_slo10_delta']):.6f} | "
                f"{float(baseline.summary['slo5_pass_ratio']):.6f} | "
                f"{float(optimized.summary['slo5_pass_ratio']):.6f} | "
                f"{float(optimized.summary['cold_start_slo5_delta']):.6f} | "
                f"{int(baseline.summary['num_flip_events'])} | "
                f"{int(optimized.summary['num_flip_events'])} | "
                f"{int(optimized.summary['cold_start_flip_event_delta'])} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_cold_start_optimization_comparisons(config: SimulationConfig) -> None:
    ratios = config.compare_threshold_margin_ratios or (config.threshold_margin_ratio,)
    rollup_lines = ["# Cold-Start Optimization Comparison Rollup", ""]
    rollup_lines.append("| margin ratio | output directory |")
    rollup_lines.append("|---:|---|")
    for ratio in ratios:
        label = margin_label(ratio)
        out_dir = config.out_dir / label
        baseline_config = replace(
            config,
            out_dir=out_dir,
            mode=MODE_CAN_FLIP,
            threshold_margin_ratio=ratio,
            compare_threshold_margin_ratios=(),
            compare_cold_start_optimization=False,
            pe_init_time_mode=PE_INIT_TIME_MODE_NON_OPTIMIZED,
            generator_init_time_mode=GENERATOR_INIT_TIME_MODE_PROFILE,
        )
        optimized_config = replace(
            config,
            out_dir=out_dir,
            mode=MODE_CAN_FLIP,
            threshold_margin_ratio=ratio,
            compare_threshold_margin_ratios=(),
            compare_cold_start_optimization=False,
            pe_init_time_mode=PE_INIT_TIME_MODE_OPTIMIZED,
            generator_init_time_mode=GENERATOR_INIT_TIME_MODE_OPTIMIZED,
        )
        baseline_results, intervals = run_suite(baseline_config)
        optimized_results, _ = run_suite(optimized_config)
        results = baseline_results + optimized_results
        add_cold_start_optimization_fields(results)
        summary_text = render_cold_start_optimization_markdown(results, ratio)
        write_outputs(replace(config, out_dir=out_dir), results, intervals, summary_text=summary_text)
        rollup_lines.append(f"| {ratio:.2f} | `{label}/` |")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "cold_start_optimization_rollup.md").write_text(
        "\n".join(rollup_lines).rstrip() + "\n",
        encoding="utf-8",
    )


def run_margin_comparisons(config: SimulationConfig) -> None:
    rollup_lines = ["# Threshold Margin Comparison Rollup", ""]
    rollup_lines.append("| margin ratio | output directory |")
    rollup_lines.append("|---:|---|")
    base_config = replace(
        config,
        mode="both",
        threshold_margin_ratio=0.0,
        compare_threshold_margin_ratios=(),
    )
    base_results, intervals = run_suite(base_config)
    for ratio in config.compare_threshold_margin_ratios:
        label = margin_label(ratio)
        out_dir = config.out_dir / label
        margin_config = replace(
            config,
            out_dir=out_dir,
            mode=MODE_CAN_FLIP,
            threshold_margin_ratio=ratio,
            compare_threshold_margin_ratios=(),
        )
        margin_results, _ = run_suite(margin_config)
        results = base_results + margin_results
        add_comparison_fields(results)
        summary_text = render_margin_comparison_markdown(results, ratio)
        output_config = replace(base_config, out_dir=out_dir)
        write_outputs(output_config, results, intervals, summary_text=summary_text)
        rollup_lines.append(f"| {ratio:.2f} | `{label}/` |")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "margin_comparison_rollup.md").write_text(
        "\n".join(rollup_lines).rstrip() + "\n",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WAN2.2 Group 1 PE-output flip/re-flip simulator.")
    parser.add_argument("--dominant-intervals-csv", type=Path, default=DEFAULT_DOMINANT_INTERVALS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--request-rates-per-min", type=float, nargs="+", default=None)
    parser.add_argument("--monitor-windows-sec", type=float, nargs="+", default=None)
    parser.add_argument("--duration-min", type=float, default=None)
    parser.add_argument("--token-threshold", type=float, default=1280.0)
    parser.add_argument("--threshold-margin-ratio", type=float, default=0.0)
    parser.add_argument(
        "--threshold-margin-mode",
        choices=[MARGIN_MODE_HYSTERESIS, MARGIN_MODE_SINGLE],
        default=MARGIN_MODE_HYSTERESIS,
    )
    parser.add_argument("--compare-threshold-margin-ratios", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["no_flip", "can_flip", "both"], default="both")
    parser.add_argument("--pe-model", default="pe7b")
    parser.add_argument("--generator-model", default="wan2.2-ti2v-5b")
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--short-output-tokens", type=int, default=512)
    parser.add_argument("--long-output-tokens", type=int, default=2048)
    parser.add_argument("--denoising-steps", type=int, default=50)
    parser.add_argument("--scenario", choices=["default", "smoke"], default="default")
    parser.add_argument("--deployment-preset", default=None)
    parser.add_argument("--flip-plan-preset", default="wan22_group1_restricted")
    parser.add_argument("--flip-source-count", type=int, default=None)
    parser.add_argument(
        "--cold-start-mode",
        choices=[COLD_START_MODE_PROFILE, COLD_START_MODE_ZERO],
        default=COLD_START_MODE_PROFILE,
    )
    parser.add_argument(
        "--pe-output-estimate-mode",
        choices=[PE_OUTPUT_ESTIMATE_MODE_MONITOR, PE_OUTPUT_ESTIMATE_MODE_ORACLE],
        default=PE_OUTPUT_ESTIMATE_MODE_MONITOR,
    )
    parser.add_argument(
        "--pe-init-time-mode",
        choices=[PE_INIT_TIME_MODE_OPTIMIZED, PE_INIT_TIME_MODE_NON_OPTIMIZED],
        default=PE_INIT_TIME_MODE_OPTIMIZED,
    )
    parser.add_argument(
        "--generator-init-time-mode",
        choices=[GENERATOR_INIT_TIME_MODE_PROFILE, GENERATOR_INIT_TIME_MODE_OPTIMIZED],
        default=GENERATOR_INIT_TIME_MODE_PROFILE,
    )
    parser.add_argument("--compare-cold-start-optimization", action="store_true")
    parser.add_argument("--trace-start", default=DEFAULT_TRACE_START)
    parser.add_argument("--verbose", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    smoke = args.scenario == "smoke"
    request_rates = tuple(args.request_rates_per_min or ([12.0] if smoke else [48.0, 60.0, 90.0]))
    windows = tuple(args.monitor_windows_sec or ([10.0] if smoke else [10.0, 20.0, 30.0, 60.0]))
    duration = args.duration_min if args.duration_min is not None else (6.0 if smoke else 60.0)
    deployment = args.deployment_preset or ("smoke" if smoke else "wan22_group1_fragment")
    flip_source_count = args.flip_source_count if args.flip_source_count is not None else (1 if smoke else 2)
    return SimulationConfig(
        dominant_intervals_csv=args.dominant_intervals_csv,
        out_dir=args.out_dir,
        request_rates_per_min=request_rates,
        monitor_windows_sec=windows,
        duration_min=duration,
        token_threshold=args.token_threshold,
        threshold_margin_ratio=args.threshold_margin_ratio,
        threshold_margin_mode=args.threshold_margin_mode,
        compare_threshold_margin_ratios=tuple(args.compare_threshold_margin_ratios or ()),
        seed=args.seed,
        mode=args.mode,
        pe_model=args.pe_model,
        generator_model=args.generator_model,
        input_tokens=args.input_tokens,
        short_output_tokens=args.short_output_tokens,
        long_output_tokens=args.long_output_tokens,
        denoising_steps=args.denoising_steps,
        scenario=args.scenario,
        deployment_preset=deployment,
        flip_plan_preset=args.flip_plan_preset,
        flip_source_count=flip_source_count,
        cold_start_mode=args.cold_start_mode,
        pe_output_estimate_mode=args.pe_output_estimate_mode,
        pe_init_time_mode=args.pe_init_time_mode,
        generator_init_time_mode=args.generator_init_time_mode,
        compare_cold_start_optimization=args.compare_cold_start_optimization,
        trace_start=args.trace_start,
        verbose=args.verbose,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if config.compare_cold_start_optimization:
        run_cold_start_optimization_comparisons(config)
        print(f"Wrote cold-start optimization comparison outputs to: {config.out_dir}")
        return
    if config.compare_threshold_margin_ratios:
        run_margin_comparisons(config)
        print(f"Wrote margin comparison outputs to: {config.out_dir}")
        return
    results, intervals = run_suite(config)
    summary_text = render_summary_markdown(results)
    write_outputs(config, results, intervals, summary_text=summary_text)
    print(summary_text)
    print(f"Wrote outputs to: {config.out_dir}")


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


if __name__ == "__main__":
    main()
