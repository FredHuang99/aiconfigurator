# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

from aiconfigurator.td_deployment_search.models import ModelSelection
from aiconfigurator.td_deployment_search.profiles import ProfileCatalog, canonical_generator_name


@dataclass(frozen=True)
class LcmStage:
    name: str
    hardware: str
    parallelism: int | str
    instances: int
    nodes: int | str
    per_instance_latency_s: float
    per_instance_throughput: float
    stage_throughput: float


@dataclass(frozen=True)
class LcmPhase:
    pe_stage: LcmStage
    encoder_stage: LcmStage
    dit_only_stage: LcmStage
    dit_vae_comb_stage: LcmStage | None
    vae_only_stage: LcmStage | None
    throughput: float
    stage_throughputs: dict[str, float]
    memory_feasible: bool


@dataclass(frozen=True)
class LcmFlipPlan:
    flipped_dit_only: int
    flipped_dit_vae_comb: int
    added_pe_instances: int
    throughput: float
    stage_throughputs: dict[str, float]


@dataclass(frozen=True)
class LcmScenarioResult:
    case_name: str
    pe_hardware: str
    pe_bundle_size: int
    pe_node_count: int
    phase1: LcmPhase
    phase2_before_flip: LcmPhase
    phase2_after_flip: LcmFlipPlan


def build_lcm_results(
    catalog: ProfileCatalog,
    base_model: ModelSelection,
) -> dict[tuple[str, int, int], list[LcmScenarioResult]]:
    results: dict[tuple[str, int, int], list[LcmScenarioResult]] = {}
    for pe_hardware in ("A800", "H100"):
        for pe_bundle_size in (1, 2):
            for pe_node_count in (1, 2):
                key = (pe_hardware, pe_bundle_size, pe_node_count)
                results[key] = [
                    _solve_lcm_scenario(catalog, base_model, pe_hardware, pe_bundle_size, pe_node_count, case_name="Case A: VAE via DiT_VAE_Comb"),
                    _solve_lcm_scenario(catalog, base_model, pe_hardware, pe_bundle_size, pe_node_count, case_name="Case B: VAE_Only minimum nodes"),
                ]
    return results


def _solve_lcm_scenario(
    catalog: ProfileCatalog,
    base_model: ModelSelection,
    pe_hardware: str,
    pe_bundle_size: int,
    pe_node_count: int,
    case_name: str,
) -> LcmScenarioResult:
    phase1_model = ModelSelection(
        pe_model=base_model.pe_model,
        generator_model=base_model.generator_model,
        input_tokens=base_model.input_tokens,
        output_tokens=512,
    )
    phase2_model = ModelSelection(
        pe_model=base_model.pe_model,
        generator_model=base_model.generator_model,
        input_tokens=base_model.input_tokens,
        output_tokens=2048,
    )
    phase1 = _build_phase(catalog, phase1_model, pe_hardware, pe_bundle_size, pe_node_count, case_name)
    phase2_before_flip = _build_phase(
        catalog,
        phase2_model,
        pe_hardware,
        pe_bundle_size,
        pe_node_count,
        case_name,
        fixed_dit_only_instances=phase1.dit_only_stage.instances,
        fixed_dit_vae_comb_instances=phase1.dit_vae_comb_stage.instances if phase1.dit_vae_comb_stage else 0,
        fixed_vae_only_stage=phase1.vae_only_stage,
        fixed_encoder_stage=phase1.encoder_stage,
    )
    phase2_after_flip = _balance_phase2_by_flips(catalog, phase2_model, phase2_before_flip, pe_bundle_size)
    return LcmScenarioResult(
        case_name=case_name,
        pe_hardware=pe_hardware,
        pe_bundle_size=pe_bundle_size,
        pe_node_count=pe_node_count,
        phase1=phase1,
        phase2_before_flip=phase2_before_flip,
        phase2_after_flip=phase2_after_flip,
    )


def _build_phase(
    catalog: ProfileCatalog,
    model: ModelSelection,
    pe_hardware: str,
    pe_bundle_size: int,
    pe_node_count: int,
    case_name: str,
    fixed_dit_only_instances: int | None = None,
    fixed_dit_vae_comb_instances: int = 0,
    fixed_vae_only_stage: LcmStage | None = None,
    fixed_encoder_stage: LcmStage | None = None,
) -> LcmPhase:
    generator = canonical_generator_name(model.generator_model)
    pe_stage = _pe_stage(catalog, model, pe_hardware, pe_bundle_size, pe_node_count)
    encoder_stage = fixed_encoder_stage or _encoder_stage(catalog, generator, pe_stage.stage_throughput)
    dit_profile = catalog.generator_stage_profile(generator, "DiT", "H100", 8)
    vae_profile = catalog.generator_stage_profile(generator, "VAE", "H100", 8)

    if fixed_dit_only_instances is None:
        total_dit_instances = math.ceil(pe_stage.stage_throughput / dit_profile.throughput)
        if case_name.startswith("Case A"):
            comb_instances = min(total_dit_instances, math.ceil(pe_stage.stage_throughput / vae_profile.throughput))
            dit_only_instances = total_dit_instances - comb_instances
            vae_only_stage = None
        else:
            comb_instances = 0
            dit_only_instances = total_dit_instances
            vae_only_stage = _choose_vae_only_stage(catalog, generator, pe_stage.stage_throughput)
    else:
        dit_only_instances = fixed_dit_only_instances
        comb_instances = fixed_dit_vae_comb_instances
        vae_only_stage = fixed_vae_only_stage

    dit_only_stage = _dit_stage(dit_profile, dit_only_instances)
    comb_stage = _comb_stage(dit_profile, vae_profile, comb_instances)
    dit_throughput = dit_only_stage.stage_throughput + (comb_stage.stage_throughput if comb_stage else 0.0)
    vae_throughput = 0.0
    if comb_stage:
        vae_throughput += comb_instances * vae_profile.throughput
    if vae_only_stage:
        vae_throughput += vae_only_stage.stage_throughput
    stage_throughputs = {
        "PE": pe_stage.stage_throughput,
        "Encoder": encoder_stage.stage_throughput,
        "DiT": dit_throughput,
        "VAE": vae_throughput,
    }
    memory_feasible = _comb_memory_feasible(catalog, generator) if comb_instances else True
    return LcmPhase(
        pe_stage=pe_stage,
        encoder_stage=encoder_stage,
        dit_only_stage=dit_only_stage,
        dit_vae_comb_stage=comb_stage,
        vae_only_stage=vae_only_stage,
        throughput=min(stage_throughputs["PE"], stage_throughputs["Encoder"], stage_throughputs["DiT"], stage_throughputs["VAE"]),
        stage_throughputs=stage_throughputs,
        memory_feasible=memory_feasible,
    )


def _pe_stage(catalog: ProfileCatalog, model: ModelSelection, hardware: str, bundle_size: int, node_count: int) -> LcmStage:
    profile = catalog.pe_stage_profile(model, hardware, bundle_size)
    instances = node_count * (8 // bundle_size)
    return LcmStage("PE_Only", hardware, bundle_size, instances, node_count, profile.latency_s, profile.throughput, instances * profile.throughput)


def _encoder_stage(catalog: ProfileCatalog, generator: str, target_throughput: float) -> LcmStage:
    latency = catalog.encoder_cpu_latency_s(generator)
    throughput = 1.0 / latency
    instances = math.ceil(target_throughput / throughput)
    return LcmStage("Encoder_CPU", "CPU", "none", instances, "shared CPU pool", latency, throughput, instances * throughput)


def _dit_stage(dit_profile, instances: int) -> LcmStage:
    return LcmStage("DiT_Only", "H100", 8, instances, instances, dit_profile.latency_s, dit_profile.throughput, instances * dit_profile.throughput)


def _comb_stage(dit_profile, vae_profile, instances: int) -> LcmStage | None:
    if instances <= 0:
        return None
    return LcmStage(
        "DiT_VAE_Comb",
        "H100",
        8,
        instances,
        instances,
        max(dit_profile.latency_s, vae_profile.latency_s),
        dit_profile.throughput,
        instances * dit_profile.throughput,
    )


def _choose_vae_only_stage(catalog: ProfileCatalog, generator: str, target_throughput: float) -> LcmStage:
    candidates: list[LcmStage] = []
    for hardware in ("A800", "H100"):
        for bundle_size in (1, 2, 4, 8):
            profile = catalog.generator_stage_profile(generator, "VAE", hardware, bundle_size)
            instances_per_node = 8 // bundle_size
            node_throughput = instances_per_node * profile.throughput
            nodes = max(1, math.ceil(target_throughput / node_throughput))
            instances = nodes * instances_per_node
            candidates.append(
                LcmStage(
                    "VAE_Only",
                    hardware,
                    bundle_size,
                    instances,
                    nodes,
                    profile.latency_s,
                    profile.throughput,
                    instances * profile.throughput,
                )
            )
    return min(
        candidates,
        key=lambda stage: (
            int(stage.nodes),
            -int(stage.parallelism),
            0 if stage.hardware == "A800" else 1,
            stage.stage_throughput,
        ),
    )


def _comb_memory_feasible(catalog: ProfileCatalog, generator: str) -> bool:
    dit = catalog.generator_stage_profile(generator, "DiT", "H100", 8)
    vae = catalog.generator_stage_profile(generator, "VAE", "H100", 8)
    return dit.memory_gb + vae.memory_gb <= 8 * 80


def _balance_phase2_by_flips(
    catalog: ProfileCatalog,
    model: ModelSelection,
    phase: LcmPhase,
    pe_bundle_size: int,
) -> LcmFlipPlan:
    generator = canonical_generator_name(model.generator_model)
    pe_profile = catalog.pe_stage_profile(model, "H100", pe_bundle_size)
    pe_per_flip = 8 // pe_bundle_size
    dit_profile = catalog.generator_stage_profile(generator, "DiT", "H100", 8)
    vae_profile = catalog.generator_stage_profile(generator, "VAE", "H100", 8)

    best: LcmFlipPlan | None = None
    max_dit_only_flips = phase.dit_only_stage.instances
    max_comb_flips = phase.dit_vae_comb_stage.instances if phase.dit_vae_comb_stage else 0
    total_possible = max_dit_only_flips + max_comb_flips
    for total_flips in range(total_possible + 1):
        dit_only_flips = min(total_flips, max_dit_only_flips)
        comb_flips = max(0, total_flips - max_dit_only_flips)
        pe_throughput = phase.pe_stage.stage_throughput + total_flips * pe_per_flip * pe_profile.throughput
        dit_throughput = phase.stage_throughputs["DiT"] - total_flips * dit_profile.throughput
        vae_throughput = phase.stage_throughputs["VAE"] - comb_flips * vae_profile.throughput
        stage_throughputs = {
            "PE": pe_throughput,
            "Encoder": phase.encoder_stage.stage_throughput,
            "DiT": dit_throughput,
            "VAE": vae_throughput,
        }
        throughput = min(stage_throughputs.values())
        candidate = LcmFlipPlan(
            flipped_dit_only=dit_only_flips,
            flipped_dit_vae_comb=comb_flips,
            added_pe_instances=total_flips * pe_per_flip,
            throughput=throughput,
            stage_throughputs=stage_throughputs,
        )
        if best is None or _flip_key(candidate) < _flip_key(best):
            best = candidate
    assert best is not None
    return best


def _flip_key(plan: LcmFlipPlan) -> tuple[float, float, int]:
    return (-plan.throughput, abs(plan.stage_throughputs["PE"] - plan.stage_throughputs["DiT"]), plan.flipped_dit_only + plan.flipped_dit_vae_comb)
