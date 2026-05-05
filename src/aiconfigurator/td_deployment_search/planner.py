# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from aiconfigurator.td_deployment_search.models import (
    SEARCH_KIND_PRUNED,
    STAGES,
    BundleAssignment,
    BundleSpec,
    ExpandedInstance,
    ModelSelection,
    NodeConfig,
    NodeSpec,
    ResourceGroupSpec,
    SearchResult,
    TemplateProfile,
)
from aiconfigurator.td_deployment_search.profiles import ProfileCatalog, pareto_prune_templates, required_stages_for_kind


BUNDLE_SIZES = (8, 4, 2, 1)


@dataclass(frozen=True)
class _VariableConfig:
    gpu_type: str
    config: NodeConfig


@dataclass(frozen=True)
class _MatrixNodeVariable:
    gpu_type: str
    partition: tuple[int, ...]


@dataclass(frozen=True)
class _MatrixBundleVariable:
    gpu_type: str
    partition: tuple[int, ...]
    bundle_size: int
    template: TemplateProfile


def build_default_resource_groups() -> tuple[ResourceGroupSpec, ResourceGroupSpec]:
    group1_nodes = (
        _make_nodes("H100-NVLink", "H100", 4, 8, 80)
        + _make_nodes("A800-NVLink", "A800", 8, 8, 80)
        + _make_nodes("A100-NVLink", "A100", 12, 8, 40)
    )
    group2_nodes = _make_nodes("H100-NVLink", "H100", 8, 8, 80) + _make_nodes(
        "A800-NVLink", "A800", 32, 8, 80
    )
    return (
        ResourceGroupSpec(name="Group 1", nodes=group1_nodes),
        ResourceGroupSpec(name="Group 2", nodes=group2_nodes),
    )


def _make_nodes(prefix: str, gpu_type: str, count: int, gpu_count: int, gpu_memory_gb: float) -> tuple[NodeSpec, ...]:
    return tuple(
        NodeSpec(
            name=f"{prefix}-{idx}",
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
        )
        for idx in range(1, count + 1)
    )


def bundle_partitions(total_gpus: int) -> tuple[tuple[int, ...], ...]:
    results: list[tuple[int, ...]] = []

    def visit(remaining: int, max_part: int, current: list[int]) -> None:
        if remaining == 0:
            results.append(tuple(current))
            return
        for part in BUNDLE_SIZES:
            if part <= remaining and part <= max_part:
                current.append(part)
                visit(remaining - part, part, current)
                current.pop()

    visit(total_gpus, max(BUNDLE_SIZES), [])
    return tuple(results)


def enumerate_node_configs(
    catalog: ProfileCatalog,
    model: ModelSelection,
    node: NodeSpec,
    search_kind: str = SEARCH_KIND_PRUNED,
) -> tuple[NodeConfig, ...]:
    seen: dict[tuple[tuple[int, str], ...], NodeConfig] = {}
    for partition in bundle_partitions(node.gpu_count):
        choices_by_bundle: list[tuple[TemplateProfile | None, ...]] = []
        for bundle_size in partition:
            legal_templates = [
                template
                for template in catalog.template_options(model, node.gpu_type, bundle_size, search_kind=search_kind)
                if template.memory_gb <= bundle_size * node.gpu_memory_gb
            ]
            choices_by_bundle.append((None, *legal_templates))

        for choices in _product(choices_by_bundle):
            assignments = tuple(
                BundleAssignment(bundle=BundleSpec(size=size, index=idx), template=template)
                for idx, (size, template) in enumerate(zip(partition, choices), start=1)
            )
            config = NodeConfig(gpu_type=node.gpu_type, assignments=assignments)
            # Drop the all-empty config from the ILP variables. Unused nodes are represented by slack.
            if not config.active_assignments:
                continue
            seen.setdefault(config.signature, config)
    return tuple(sorted(seen.values(), key=lambda config: (config.active_bundle_count, config.signature)))


def _product(groups: list[tuple[TemplateProfile | None, ...]]) -> Iterable[tuple[TemplateProfile | None, ...]]:
    if not groups:
        yield ()
        return
    first, *rest = groups
    for item in first:
        for suffix in _product(rest):
            yield (item, *suffix)


def solve_group_enumeration(
    catalog: ProfileCatalog,
    group: ResourceGroupSpec,
    model: ModelSelection,
    top_k: int = 5,
    mip_time_limit_s: float = 120.0,
    search_kind: str = SEARCH_KIND_PRUNED,
) -> list[SearchResult]:
    variable_configs: list[_VariableConfig] = []
    representative_nodes = {
        gpu_type: nodes[0]
        for gpu_type, nodes in group.nodes_by_gpu_type.items()
    }
    for gpu_type, node in representative_nodes.items():
        for config in enumerate_node_configs(catalog, model, node, search_kind=search_kind):
            variable_configs.append(_VariableConfig(gpu_type=gpu_type, config=config))

    if not variable_configs:
        return []

    excluded: list[np.ndarray] = []
    results: list[SearchResult] = []
    seen_counts: set[tuple[int, ...]] = set()
    for rank in range(1, top_k + 1):
        solution = _solve_config_counts(
            variable_configs=variable_configs,
            group=group,
            excluded=excluded,
            mip_time_limit_s=mip_time_limit_s,
            required_stages=required_stages_for_kind(search_kind),
        )
        if solution is None:
            break
        counts, _lambda_value = solution
        signature = tuple(int(value) for value in counts)
        if signature in seen_counts:
            excluded.append(counts)
            continue
        seen_counts.add(signature)
        excluded.append(counts)
        result = _build_search_result(
            catalog=catalog,
            group=group,
            model=model,
            rank=rank,
            variable_configs=variable_configs,
            counts=counts,
            required_stages=required_stages_for_kind(search_kind),
        )
        if result.throughput <= 0:
            break
        results.append(result)
    return results


def _solve_config_counts(
    variable_configs: list[_VariableConfig],
    group: ResourceGroupSpec,
    excluded: list[np.ndarray],
    mip_time_limit_s: float,
    required_stages: tuple[str, ...] = STAGES,
) -> tuple[np.ndarray, float] | None:
    base_n = len(variable_configs)
    lambda_idx = base_n
    aux_n = len(excluded) * base_n * 2
    total_n = base_n + 1 + aux_n

    objective = np.zeros(total_n)
    for idx, var_config in enumerate(variable_configs):
        objective[idx] = 1e-10 * max(1, var_config.config.active_bundle_count)
    objective[lambda_idx] = -1.0

    lower_bounds = np.zeros(total_n)
    upper_bounds = np.full(total_n, np.inf)
    node_counts = group.node_counts_by_gpu_type
    for idx, var_config in enumerate(variable_configs):
        upper_bounds[idx] = node_counts[var_config.gpu_type]
    upper_bounds[lambda_idx] = np.inf
    if aux_n:
        upper_bounds[base_n + 1 :] = 1.0

    integrality = np.zeros(total_n)
    integrality[:base_n] = 1
    if aux_n:
        integrality[base_n + 1 :] = 1

    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    for gpu_type, node_count in node_counts.items():
        row = np.zeros(total_n)
        for idx, var_config in enumerate(variable_configs):
            if var_config.gpu_type == gpu_type:
                row[idx] = 1.0
        rows.append(row)
        lbs.append(0.0)
        ubs.append(float(node_count))

    for stage in required_stages:
        row = np.zeros(total_n)
        for idx, var_config in enumerate(variable_configs):
            row[idx] = -var_config.config.stage_throughputs.get(stage, 0.0)
        row[lambda_idx] = 1.0
        rows.append(row)
        lbs.append(-np.inf)
        ubs.append(0.0)

    aux_offset = base_n + 1
    for excluded_idx, previous in enumerate(excluded):
        cut_sum_row = np.zeros(total_n)
        for var_idx, previous_value in enumerate(previous):
            ub = upper_bounds[var_idx]
            big_m = ub + 1.0
            lt_idx = aux_offset + excluded_idx * base_n * 2 + var_idx * 2
            gt_idx = lt_idx + 1

            lt_row = np.zeros(total_n)
            lt_row[var_idx] = 1.0
            lt_row[lt_idx] = big_m
            rows.append(lt_row)
            lbs.append(-np.inf)
            ubs.append(float(previous_value) - 1.0 + big_m)

            gt_row = np.zeros(total_n)
            gt_row[var_idx] = -1.0
            gt_row[gt_idx] = big_m
            rows.append(gt_row)
            lbs.append(-np.inf)
            ubs.append(-float(previous_value) - 1.0 + big_m)

            cut_sum_row[lt_idx] = -1.0
            cut_sum_row[gt_idx] = -1.0

        rows.append(cut_sum_row)
        lbs.append(-np.inf)
        ubs.append(-1.0)

    constraints = LinearConstraint(np.vstack(rows), np.array(lbs), np.array(ubs))
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"time_limit": mip_time_limit_s, "mip_rel_gap": 1e-9},
    )
    if not result.success or result.x is None:
        return None
    counts = np.rint(result.x[:base_n]).astype(int)
    lambda_value = float(result.x[lambda_idx])
    return counts, lambda_value


def _build_search_result(
    catalog: ProfileCatalog,
    group: ResourceGroupSpec,
    model: ModelSelection,
    rank: int,
    variable_configs: list[_VariableConfig],
    counts: np.ndarray,
    required_stages: tuple[str, ...] = STAGES,
) -> SearchResult:
    stage_throughputs = {stage: 0.0 for stage in required_stages}
    config_counts: dict[int, int] = {}
    node_configs: list[NodeConfig] = []
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        config_counts[idx] = int(count)
        config = variable_configs[idx].config
        for stage in required_stages:
            stage_throughputs[stage] += config.stage_throughputs.get(stage, 0.0) * int(count)
        node_configs.extend([config] * int(count))

    throughput = min(stage_throughputs.values())
    encoder_cpu_latency_s = catalog.encoder_cpu_latency_s(model.generator_model)
    encoder_cpu_throughput = 1.0 / encoder_cpu_latency_s
    encoder_cpu_instances = math.ceil(throughput / encoder_cpu_throughput) if throughput > 0 else 0
    encoder_cpu_memory_gb = catalog.encoder_cpu_memory_gb(model.generator_model)
    encoder_cpu_memory_feasible = encoder_cpu_instances * encoder_cpu_memory_gb <= group.total_dram_gb

    expanded_instances = _expand_instances(group, variable_configs, counts)
    return SearchResult(
        group=group,
        model=model,
        rank=rank,
        throughput=throughput,
        stage_throughputs=stage_throughputs,
        config_counts=config_counts,
        node_configs=tuple(node_configs),
        expanded_instances=expanded_instances,
        encoder_cpu_latency_s=encoder_cpu_latency_s,
        encoder_cpu_throughput=encoder_cpu_throughput,
        encoder_cpu_instances=encoder_cpu_instances,
        encoder_cpu_memory_gb=encoder_cpu_memory_gb,
        encoder_cpu_memory_feasible=encoder_cpu_memory_feasible,
        required_stages=required_stages,
    )


def _expand_instances(
    group: ResourceGroupSpec,
    variable_configs: list[_VariableConfig],
    counts: np.ndarray,
) -> tuple[ExpandedInstance, ...]:
    available_nodes = {gpu_type: list(nodes) for gpu_type, nodes in group.nodes_by_gpu_type.items()}
    config_queue_by_gpu_type: dict[str, list[NodeConfig]] = {}
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        var_config = variable_configs[idx]
        config_queue_by_gpu_type.setdefault(var_config.gpu_type, []).extend([var_config.config] * int(count))

    instances: list[ExpandedInstance] = []
    for gpu_type, configs in config_queue_by_gpu_type.items():
        configs = sorted(configs, key=lambda config: config.signature)
        for node, config in zip(available_nodes[gpu_type], configs):
            for assignment in config.active_assignments:
                template = assignment.template
                assert template is not None
                for profile in template.stage_profiles:
                    instances.append(
                        ExpandedInstance(
                            stage=profile.stage,
                            template_name=template.name,
                            node_name=node.name,
                            gpu_type=gpu_type,
                            parallelism=profile.parallelism,
                            latency_s=profile.latency_s,
                            throughput=profile.throughput,
                            bundle_label=assignment.bundle.label,
                            bundle_size=assignment.bundle.size,
                            parallelism_method=profile.parallelism_method,
                        )
                    )
    return tuple(instances)


def solve_group(
    catalog: ProfileCatalog,
    group: ResourceGroupSpec,
    model: ModelSelection,
    top_k: int = 5,
    mip_time_limit_s: float = 120.0,
    search_kind: str = SEARCH_KIND_PRUNED,
    solver_backend: str = "matrix",
) -> list[SearchResult]:
    if solver_backend == "enumeration":
        return solve_group_enumeration(
            catalog=catalog,
            group=group,
            model=model,
            top_k=top_k,
            mip_time_limit_s=mip_time_limit_s,
            search_kind=search_kind,
        )
    if solver_backend != "matrix":
        raise ValueError(f"Unsupported solver backend: {solver_backend}")
    return solve_group_matrix(
        catalog=catalog,
        group=group,
        model=model,
        top_k=top_k,
        mip_time_limit_s=mip_time_limit_s,
        search_kind=search_kind,
    )


def solve_group_matrix(
    catalog: ProfileCatalog,
    group: ResourceGroupSpec,
    model: ModelSelection,
    top_k: int = 5,
    mip_time_limit_s: float = 120.0,
    search_kind: str = SEARCH_KIND_PRUNED,
) -> list[SearchResult]:
    required_stages = required_stages_for_kind(search_kind)
    node_variables: list[_MatrixNodeVariable] = []
    bundle_variables: list[_MatrixBundleVariable] = []
    representative_nodes = {gpu_type: nodes[0] for gpu_type, nodes in group.nodes_by_gpu_type.items()}
    for gpu_type, node in representative_nodes.items():
        legal_by_size: dict[int, tuple[TemplateProfile, ...]] = {}
        for bundle_size in BUNDLE_SIZES:
            templates = tuple(
                template
                for template in catalog.template_options(model, gpu_type, bundle_size, search_kind=search_kind)
                if template.memory_gb <= bundle_size * node.gpu_memory_gb
            )
            if search_kind != SEARCH_KIND_PRUNED:
                templates = pareto_prune_templates(templates, required_stages)
            legal_by_size[bundle_size] = templates
        for partition in bundle_partitions(node.gpu_count):
            node_variables.append(_MatrixNodeVariable(gpu_type=gpu_type, partition=partition))
            for bundle_size in sorted(set(partition), reverse=True):
                for template in legal_by_size[bundle_size]:
                    bundle_variables.append(
                        _MatrixBundleVariable(
                            gpu_type=gpu_type,
                            partition=partition,
                            bundle_size=bundle_size,
                            template=template,
                        )
                    )

    excluded: list[np.ndarray] = []
    results: list[SearchResult] = []
    for rank in range(1, top_k + 1):
        solution = _solve_matrix_counts(
            node_variables=node_variables,
            bundle_variables=bundle_variables,
            group=group,
            excluded=excluded,
            mip_time_limit_s=mip_time_limit_s,
            required_stages=required_stages,
        )
        if solution is None:
            break
        node_counts, bundle_counts = solution
        excluded.append(bundle_counts)
        result = _build_matrix_result(
            catalog=catalog,
            group=group,
            model=model,
            rank=rank,
            node_variables=node_variables,
            bundle_variables=bundle_variables,
            node_counts=node_counts,
            bundle_counts=bundle_counts,
            required_stages=required_stages,
        )
        if result.throughput <= 0:
            break
        results.append(result)
    return results


def _solve_matrix_counts(
    node_variables: list[_MatrixNodeVariable],
    bundle_variables: list[_MatrixBundleVariable],
    group: ResourceGroupSpec,
    excluded: list[np.ndarray],
    mip_time_limit_s: float,
    required_stages: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    node_n = len(node_variables)
    bundle_n = len(bundle_variables)
    lambda_idx = node_n + bundle_n
    aux_n = len(excluded) * bundle_n * 2
    total_n = node_n + bundle_n + 1 + aux_n

    objective = np.zeros(total_n)
    objective[lambda_idx] = -1.0
    objective[: node_n + bundle_n] = 1e-10

    lower_bounds = np.zeros(total_n)
    upper_bounds = np.full(total_n, np.inf)
    node_counts_by_type = group.node_counts_by_gpu_type
    for idx, node_var in enumerate(node_variables):
        upper_bounds[idx] = node_counts_by_type[node_var.gpu_type]
    for idx, bundle_var in enumerate(bundle_variables, start=node_n):
        upper_bounds[idx] = node_counts_by_type[bundle_var.gpu_type] * bundle_var.partition.count(bundle_var.bundle_size)
    if aux_n:
        upper_bounds[node_n + bundle_n + 1 :] = 1.0

    integrality = np.zeros(total_n)
    integrality[: node_n + bundle_n] = 1
    if aux_n:
        integrality[node_n + bundle_n + 1 :] = 1

    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    for gpu_type, available_count in node_counts_by_type.items():
        row = np.zeros(total_n)
        for idx, node_var in enumerate(node_variables):
            if node_var.gpu_type == gpu_type:
                row[idx] = 1.0
        rows.append(row)
        lbs.append(0.0)
        ubs.append(float(available_count))

    for node_idx, node_var in enumerate(node_variables):
        for bundle_size in sorted(set(node_var.partition), reverse=True):
            row = np.zeros(total_n)
            for bundle_idx, bundle_var in enumerate(bundle_variables, start=node_n):
                if (
                    bundle_var.gpu_type == node_var.gpu_type
                    and bundle_var.partition == node_var.partition
                    and bundle_var.bundle_size == bundle_size
                ):
                    row[bundle_idx] = 1.0
            row[node_idx] = -node_var.partition.count(bundle_size)
            rows.append(row)
            lbs.append(-np.inf)
            ubs.append(0.0)

    for stage in required_stages:
        row = np.zeros(total_n)
        row[lambda_idx] = 1.0
        for bundle_idx, bundle_var in enumerate(bundle_variables, start=node_n):
            row[bundle_idx] = -bundle_var.template.stage_throughputs.get(stage, 0.0)
        rows.append(row)
        lbs.append(-np.inf)
        ubs.append(0.0)

    aux_offset = node_n + bundle_n + 1
    for excluded_idx, previous_bundle_counts in enumerate(excluded):
        cut_sum_row = np.zeros(total_n)
        for local_idx, previous_value in enumerate(previous_bundle_counts):
            var_idx = node_n + local_idx
            big_m = upper_bounds[var_idx] + 1.0
            lt_idx = aux_offset + excluded_idx * bundle_n * 2 + local_idx * 2
            gt_idx = lt_idx + 1

            lt_row = np.zeros(total_n)
            lt_row[var_idx] = 1.0
            lt_row[lt_idx] = big_m
            rows.append(lt_row)
            lbs.append(-np.inf)
            ubs.append(float(previous_value) - 1.0 + big_m)

            gt_row = np.zeros(total_n)
            gt_row[var_idx] = -1.0
            gt_row[gt_idx] = big_m
            rows.append(gt_row)
            lbs.append(-np.inf)
            ubs.append(-float(previous_value) - 1.0 + big_m)

            cut_sum_row[lt_idx] = -1.0
            cut_sum_row[gt_idx] = -1.0
        rows.append(cut_sum_row)
        lbs.append(-np.inf)
        ubs.append(-1.0)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), np.array(lbs), np.array(ubs)),
        options={"time_limit": mip_time_limit_s, "mip_rel_gap": 1e-9},
    )
    if not result.success or result.x is None:
        return None
    node_counts = np.rint(result.x[:node_n]).astype(int)
    bundle_counts = np.rint(result.x[node_n : node_n + bundle_n]).astype(int)
    return node_counts, bundle_counts


def _build_matrix_result(
    catalog: ProfileCatalog,
    group: ResourceGroupSpec,
    model: ModelSelection,
    rank: int,
    node_variables: list[_MatrixNodeVariable],
    bundle_variables: list[_MatrixBundleVariable],
    node_counts: np.ndarray,
    bundle_counts: np.ndarray,
    required_stages: tuple[str, ...],
) -> SearchResult:
    stage_throughputs = {stage: 0.0 for stage in required_stages}
    for count, bundle_var in zip(bundle_counts, bundle_variables):
        if count <= 0:
            continue
        for stage in required_stages:
            stage_throughputs[stage] += bundle_var.template.stage_throughputs.get(stage, 0.0) * int(count)
    throughput = min(stage_throughputs.values())

    encoder_cpu_latency_s = catalog.encoder_cpu_latency_s(model.generator_model)
    encoder_cpu_throughput = 1.0 / encoder_cpu_latency_s
    encoder_cpu_instances = math.ceil(throughput / encoder_cpu_throughput) if throughput > 0 else 0
    encoder_cpu_memory_gb = catalog.encoder_cpu_memory_gb(model.generator_model)
    encoder_cpu_memory_feasible = encoder_cpu_instances * encoder_cpu_memory_gb <= group.total_dram_gb

    expanded_instances = _expand_matrix_instances(group, node_variables, bundle_variables, node_counts, bundle_counts)
    return SearchResult(
        group=group,
        model=model,
        rank=rank,
        throughput=throughput,
        stage_throughputs=stage_throughputs,
        config_counts={idx: int(count) for idx, count in enumerate(bundle_counts) if count > 0},
        node_configs=(),
        expanded_instances=expanded_instances,
        encoder_cpu_latency_s=encoder_cpu_latency_s,
        encoder_cpu_throughput=encoder_cpu_throughput,
        encoder_cpu_instances=encoder_cpu_instances,
        encoder_cpu_memory_gb=encoder_cpu_memory_gb,
        encoder_cpu_memory_feasible=encoder_cpu_memory_feasible,
        required_stages=required_stages,
    )


def _expand_matrix_instances(
    group: ResourceGroupSpec,
    node_variables: list[_MatrixNodeVariable],
    bundle_variables: list[_MatrixBundleVariable],
    node_counts: np.ndarray,
    bundle_counts: np.ndarray,
) -> tuple[ExpandedInstance, ...]:
    slots: dict[tuple[str, tuple[int, ...], int], list[tuple[str, BundleSpec]]] = {}
    available_nodes = {gpu_type: list(nodes) for gpu_type, nodes in group.nodes_by_gpu_type.items()}
    node_cursor = {gpu_type: 0 for gpu_type in available_nodes}
    for count, node_var in zip(node_counts, node_variables):
        for _ in range(int(count)):
            nodes = available_nodes[node_var.gpu_type]
            node = nodes[node_cursor[node_var.gpu_type]]
            node_cursor[node_var.gpu_type] += 1
            local_index_by_size: dict[int, int] = {}
            for bundle_size in node_var.partition:
                local_index_by_size[bundle_size] = local_index_by_size.get(bundle_size, 0) + 1
                bundle = BundleSpec(size=bundle_size, index=local_index_by_size[bundle_size])
                slots.setdefault((node_var.gpu_type, node_var.partition, bundle_size), []).append((node.name, bundle))

    instances: list[ExpandedInstance] = []
    for count, bundle_var in zip(bundle_counts, bundle_variables):
        if count <= 0:
            continue
        slot_key = (bundle_var.gpu_type, bundle_var.partition, bundle_var.bundle_size)
        for _ in range(int(count)):
            node_name, bundle = slots[slot_key].pop(0)
            for profile in bundle_var.template.stage_profiles:
                instances.append(
                    ExpandedInstance(
                        stage=profile.stage,
                        template_name=bundle_var.template.name,
                        node_name=node_name,
                        gpu_type=bundle_var.gpu_type,
                        parallelism=profile.parallelism,
                        latency_s=profile.latency_s,
                        throughput=profile.throughput,
                        bundle_label=bundle.label,
                        bundle_size=bundle.size,
                        parallelism_method=profile.parallelism_method,
                    )
                )
    return tuple(instances)


def result_template_counter(result: SearchResult) -> Counter[tuple[str, int, str]]:
    counter: Counter[tuple[str, int, str]] = Counter()
    for instance in result.expanded_instances:
        # Count once per template instance, not once per stage inside a compound template.
        if instance.stage == "VAE" and instance.template_name == "DiT_VAE_Comb":
            continue
        counter[(instance.template_name, instance.parallelism, instance.gpu_type)] += 1
    return counter
