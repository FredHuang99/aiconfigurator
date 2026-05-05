# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


STAGE_PE = "PE"
STAGE_ENCODER = "Encoder"
STAGE_DIT = "DiT"
STAGE_VAE = "VAE"
STAGES: tuple[str, ...] = (STAGE_PE, STAGE_DIT, STAGE_VAE)
STAGES_WITH_ENCODER: tuple[str, ...] = (STAGE_PE, STAGE_ENCODER, STAGE_DIT, STAGE_VAE)

SEARCH_KIND_PRUNED = "pruned"
SEARCH_KIND_FULL_ENCODER_CPU = "full_encoder_cpu"
SEARCH_KIND_FULL_ENCODER_GPU = "full_encoder_gpu"

TEMPLATE_EMPTY = "Empty"
TEMPLATE_PE_ONLY = "PE_Only"
TEMPLATE_ENCODER_ONLY = "Encoder_Only"
TEMPLATE_ENCODER_CPU = "Encoder_CPU"
TEMPLATE_DIT_ONLY = "DiT_Only"
TEMPLATE_VAE_ONLY = "VAE_Only"
TEMPLATE_DIT_VAE_COMB = "DiT_VAE_Comb"


@dataclass(frozen=True)
class NodeSpec:
    name: str
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: float
    dram_gb: float = 512.0

    @property
    def total_hbm_gb(self) -> float:
        return self.gpu_count * self.gpu_memory_gb


@dataclass(frozen=True)
class ResourceGroupSpec:
    name: str
    nodes: tuple[NodeSpec, ...]
    network_setup: str = "All inter-node links are treated as >= 10Gb/s; weak-network search ignores transfer latency."

    @property
    def total_dram_gb(self) -> float:
        return sum(node.dram_gb for node in self.nodes)

    @property
    def node_counts_by_gpu_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.gpu_type] = counts.get(node.gpu_type, 0) + 1
        return counts

    @property
    def nodes_by_gpu_type(self) -> dict[str, list[NodeSpec]]:
        grouped: dict[str, list[NodeSpec]] = {}
        for node in self.nodes:
            grouped.setdefault(node.gpu_type, []).append(node)
        return grouped


@dataclass(frozen=True)
class ModelSelection:
    pe_model: str = "pe7b"
    generator_model: str = "wan2.2-ti2v-5b"
    input_tokens: int = 128
    output_tokens: int = 512


@dataclass(frozen=True)
class StageProfile:
    stage: str
    gpu_type: str
    parallelism: int
    latency_s: float
    memory_gb: float
    parallelism_method: str
    profile_source: str

    @property
    def throughput(self) -> float:
        return 1.0 / self.latency_s


@dataclass(frozen=True)
class TemplateProfile:
    name: str
    bundle_size: int
    gpu_type: str
    stage_profiles: tuple[StageProfile, ...]

    @property
    def memory_gb(self) -> float:
        return sum(profile.memory_gb for profile in self.stage_profiles)

    @property
    def stage_throughputs(self) -> dict[str, float]:
        throughputs = {stage: 0.0 for stage in STAGES_WITH_ENCODER}
        for profile in self.stage_profiles:
            throughputs[profile.stage] += profile.throughput
        return throughputs

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(profile.stage for profile in self.stage_profiles)


@dataclass(frozen=True)
class BundleSpec:
    size: int
    index: int

    @property
    def label(self) -> str:
        return f"bundle-{self.index}"


@dataclass(frozen=True)
class BundleAssignment:
    bundle: BundleSpec
    template: TemplateProfile | None

    @property
    def is_empty(self) -> bool:
        return self.template is None


@dataclass(frozen=True)
class NodeConfig:
    gpu_type: str
    assignments: tuple[BundleAssignment, ...]
    signature: tuple[tuple[int, str], ...] = field(init=False)

    def __post_init__(self) -> None:
        canonical = tuple(
            sorted(
                (assignment.bundle.size, assignment.template.name if assignment.template else TEMPLATE_EMPTY)
                for assignment in self.assignments
            )
        )
        object.__setattr__(self, "signature", canonical)

    @property
    def active_assignments(self) -> tuple[BundleAssignment, ...]:
        return tuple(assignment for assignment in self.assignments if assignment.template is not None)

    @property
    def active_bundle_count(self) -> int:
        return len(self.active_assignments)

    @property
    def stage_throughputs(self) -> dict[str, float]:
        throughputs = {stage: 0.0 for stage in STAGES_WITH_ENCODER}
        for assignment in self.active_assignments:
            assert assignment.template is not None
            for stage, throughput in assignment.template.stage_throughputs.items():
                throughputs[stage] += throughput
        return throughputs

    @property
    def template_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for assignment in self.active_assignments:
            assert assignment.template is not None
            counts[assignment.template.name] = counts.get(assignment.template.name, 0) + 1
        return counts


@dataclass(frozen=True)
class ExpandedInstance:
    stage: str
    template_name: str
    node_name: str
    gpu_type: str
    parallelism: int
    latency_s: float
    throughput: float
    bundle_label: str
    bundle_size: int
    parallelism_method: str


@dataclass(frozen=True)
class SearchResult:
    group: ResourceGroupSpec
    model: ModelSelection
    rank: int
    throughput: float
    stage_throughputs: Mapping[str, float]
    config_counts: Mapping[int, int]
    node_configs: tuple[NodeConfig, ...]
    expanded_instances: tuple[ExpandedInstance, ...]
    encoder_cpu_latency_s: float
    encoder_cpu_throughput: float
    encoder_cpu_instances: int
    encoder_cpu_memory_gb: float
    encoder_cpu_memory_feasible: bool
    required_stages: tuple[str, ...] = STAGES

    @property
    def bottleneck_stage(self) -> str:
        return min(self.required_stages, key=lambda stage: self.stage_throughputs.get(stage, 0.0))
