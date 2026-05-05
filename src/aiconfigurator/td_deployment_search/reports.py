# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from aiconfigurator.td_deployment_search.lcm import LcmScenarioResult
from aiconfigurator.td_deployment_search.models import (
    SEARCH_KIND_PRUNED,
    TEMPLATE_DIT_ONLY,
    TEMPLATE_DIT_VAE_COMB,
    TEMPLATE_PE_ONLY,
    TEMPLATE_VAE_ONLY,
    ModelSelection,
    NodeSpec,
    SearchResult,
    StageProfile,
    TemplateProfile,
)
from aiconfigurator.td_deployment_search.planner import bundle_partitions, result_template_counter
from aiconfigurator.td_deployment_search.profiles import ProfileCatalog, canonical_generator_name


MAIN_REPORT_PATH = Path(r"C:\aiconfigurator\td_main_search_report.txt")
LCM_REPORT_PATH = Path(r"C:\aiconfigurator\td_lcm_report.txt")
FULL_REPORT_PATH = Path(r"C:\aiconfigurator\td_full_search_report.txt")


TemplateKey = tuple[str, int, str]


@dataclass(frozen=True)
class FragmentCompletion:
    name: str
    counter: Counter[TemplateKey]
    actions: tuple[str, ...]
    stage_throughputs: dict[str, float]
    throughput: float
    utilization: dict[str, tuple[int, int]]


def render_main_report(
    catalog: ProfileCatalog,
    grouped_results: dict[str, dict[int, list[SearchResult]]],
    output_path: Path = MAIN_REPORT_PATH,
) -> str:
    lines: list[str] = []
    lines.append("TD Deployment Search Planner Report")
    lines.append("=" * 43)
    lines.append("")
    lines.append("Setup")
    lines.append("- Resource groups: Group 1 = 4 H100 nodes + 8 A800 nodes + 12 A100 nodes; Group 2 = 8 H100 nodes + 32 A800 nodes.")
    lines.append("- Each node has 8 GPUs. H100/A800 GPU memory is 80GB; A100 GPU memory is 40GB. Node DRAM is 512GB.")
    lines.append("- Network: all inter-node links are treated as excellent and non-bottleneck, so the weak-network max-min ILP is used.")
    lines.append("- Memory convention: template memory is the sum of profiled components and must be <= bundle total HBM.")
    lines.append("- Main templates considered: PE_Only, DiT_Only, VAE_Only, DiT_VAE_Comb; Encoder_CPU is derived after solving.")
    lines.append("- General vocabulary reserved for future runs: PE_Only, Encoder_Only, Encoder_CPU, DiT_Only, VAE_Only, PE_Encoder_Comb, DiT_VAE_Comb.")
    lines.append("")
    lines.append("Profile Completeness")
    for note in catalog.completeness_notes():
        lines.append(f"- {note}")
    lines.append("")
    lines.append("Unpruned Planning-Space Estimate")
    lines.append("- For an 8-GPU node, bundle partitions over {1,2,4,8} have node space sum_p (T+1)^|p| = 7,413,245,658 when T=16 template families.")
    lines.append("- Group 1 has 24 nodes, so naive unpruned search is about O(7,413,245,658^24) ~= 10^236 configurations.")
    lines.append("- Group 2 has 40 nodes, so naive unpruned search is about O(7,413,245,658^40) ~= 10^395 configurations.")
    lines.append("- On the current 14-logical-processor machine, exhaustive enumeration would be effectively intractable; even at 10M configs/s it would exceed astronomical time scales.")
    lines.append("")

    for group_name in ("Group 1", "Group 2"):
        results_by_output = grouped_results[group_name]
        lines.append(group_name)
        lines.append("-" * len(group_name))
        for output_tokens in (512, 2048):
            results = results_by_output.get(output_tokens, [])
            if not results:
                lines.append(f"Output tokens == {output_tokens}: no feasible config found.")
                continue
            top = results[0]
            lines.extend(
                _render_result(
                    catalog,
                    top,
                    title=f"Output tokens == {output_tokens}: top config",
                    include_fragment_completion=True,
                )
            )
            if len(results) > 1:
                lines.append(f"Output tokens == {output_tokens}: top 2-5 structurally distinct configs")
                for result in results[1:5]:
                    lines.append(
                        f"- Rank {result.rank}: throughput={result.throughput:.4f} req/s; "
                        f"PE={result.stage_throughputs['PE']:.4f}, DiT={result.stage_throughputs['DiT']:.4f}, "
                        f"VAE={result.stage_throughputs['VAE']:.4f}; bottleneck={result.bottleneck_stage}"
                    )
                    lines.append(f"  Node summary: {_summarize_templates(result)}")
                lines.append("")
        if 512 in results_by_output and 2048 in results_by_output:
            lines.extend(_render_flip_plan(catalog, results_by_output[512][0], results_by_output[2048][0]))
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


def render_full_report(
    catalog: ProfileCatalog,
    grouped_results: dict[str, dict[str, dict[int, list[SearchResult]]]],
    output_path: Path = FULL_REPORT_PATH,
) -> str:
    lines: list[str] = []
    lines.append("TD Full-Template Search Report")
    lines.append("=" * 31)
    lines.append("")
    lines.append("Full search variants:")
    lines.append("- Encoder_CPU: searches all non-empty templates over {PE, DiT, VAE}; Encoder_CPU is derived after solving.")
    lines.append("- Encoder_GPU: searches all 15 non-empty templates over {PE, Encoder, DiT, VAE}.")
    lines.append("- Legal templates are memory-filtered and Pareto-pruned per hardware/bundle size before matrix ILP.")
    lines.append("")
    for group_name in ("Group 1", "Group 2"):
        lines.append(group_name)
        lines.append("-" * len(group_name))
        for variant_name, results_by_output in grouped_results[group_name].items():
            lines.append(variant_name)
            for output_tokens in (512, 2048):
                results = results_by_output.get(output_tokens, [])
                if not results:
                    lines.append(f"- Output tokens == {output_tokens}: no feasible config found.")
                    continue
                top = results[0]
                lines.extend(_render_result(catalog, top, title=f"Output tokens == {output_tokens}: top config"))
                if len(results) > 1:
                    lines.append(f"Output tokens == {output_tokens}: top 2-5 structurally distinct configs")
                    for result in results[1:5]:
                        stage_text = ", ".join(
                            f"{stage}={result.stage_throughputs.get(stage, 0.0):.4f}" for stage in result.required_stages
                        )
                        lines.append(f"- Rank {result.rank}: throughput={result.throughput:.4f} req/s; {stage_text}; bottleneck={result.bottleneck_stage}")
                        lines.append(f"  Node summary: {_summarize_templates(result)}")
                    lines.append("")
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


def _render_result(
    catalog: ProfileCatalog,
    result: SearchResult,
    title: str,
    include_fragment_completion: bool = False,
) -> list[str]:
    lines = [title]
    lines.append(
        f"- Model: {result.model.pe_model} + {canonical_generator_name(result.model.generator_model)}, "
        f"input tokens={result.model.input_tokens}, output tokens={result.model.output_tokens}"
    )
    lines.append(
        f"- Overall throughput: {result.throughput:.4f} req/s; bottleneck stage: {result.bottleneck_stage}"
    )
    for stage in result.required_stages:
        stage_instances = [instance for instance in result.expanded_instances if instance.stage == stage]
        lines.append(
            f"- {stage} stage: {len(stage_instances)} profiled stage entries, "
            f"stage throughput={result.stage_throughputs[stage]:.4f} req/s"
        )
        for instance in stage_instances:
            lines.append(
                f"  * {instance.template_name} on {instance.node_name}/{instance.bundle_label}: "
                f"{instance.parallelism_method}{instance.parallelism}, {instance.bundle_size}x{instance.gpu_type}, "
                f"latency={instance.latency_s:.4f}s, throughput={instance.throughput:.4f} req/s"
            )
    lines.append(
        f"- Encoder_CPU stage: {result.encoder_cpu_instances} instances, "
        f"per-instance latency={result.encoder_cpu_latency_s:.4f}s, "
        f"stage throughput={result.encoder_cpu_instances * result.encoder_cpu_throughput:.4f} req/s, "
        f"DRAM feasible={result.encoder_cpu_memory_feasible}"
    )
    lines.append(f"- Throughput arithmetic: {_result_throughput_arithmetic(catalog, result)}")
    lines.append(f"- Active GPU utilization (base ILP core): {_format_utilization(_base_utilization(result))}")
    lines.append(f"- Node summary (base ILP core): {_summarize_templates(result)}")
    if include_fragment_completion:
        completions = _fragment_completion_options(catalog, result)
        for completion in completions:
            lines.append(
                f"- Fragment completion ({completion.name}): throughput={completion.throughput:.4f} req/s; "
                f"PE={completion.stage_throughputs.get('PE', 0.0):.4f}, "
                f"DiT={completion.stage_throughputs.get('DiT', 0.0):.4f}, "
                f"VAE={completion.stage_throughputs.get('VAE', 0.0):.4f}; "
                f"active GPU utilization={_format_utilization(completion.utilization)}"
            )
            if completion.actions:
                lines.append(f"  Actions: {'; '.join(completion.actions)}")
            else:
                lines.append("  Actions: no active-node fragments to fill.")
            lines.append(f"  Node summary: {_summarize_counter(completion.counter)}")
    lines.append("")
    return lines


def _summarize_templates(result: SearchResult) -> str:
    counter = result_template_counter(result)
    return _summarize_counter(counter)


def _summarize_counter(counter) -> str:
    if not counter:
        return "no active GPU templates"
    chunks = []
    for (template, parallelism, hardware), count in sorted(counter.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])):
        chunks.append(f"{count}x {template}_{hardware}_P{parallelism}")
    return "; ".join(chunks)


def _render_flip_plan(catalog: ProfileCatalog, source: SearchResult, target: SearchResult) -> list[str]:
    lines = [f"Flip plan: output {source.model.output_tokens} -> {target.model.output_tokens}"]
    source_base_counter = result_template_counter(source)
    target_base_counter = result_template_counter(target)
    source_completed = _fragment_completion(catalog, source, "same_model")
    target_completed = _fragment_completion(catalog, target, "same_model")
    source_counter = source_completed.counter
    target_counter = target_completed.counter
    restricted_counter, restricted_actions = _restricted_flip_counter(catalog, target.model, source_counter, target_counter)
    restricted_caps = _counter_stage_caps(catalog, target.model, restricted_counter)
    restricted_throughput = min(restricted_caps[stage] for stage in target.required_stages)
    target_completed_throughput = min(target_completed.stage_throughputs[stage] for stage in target.required_stages)
    gap = target_completed_throughput - restricted_throughput

    lines.append("Result 1: restricted flip plan")
    if restricted_actions:
        for action in restricted_actions:
            lines.append(f"- {action}")
    else:
        lines.append("- No restricted DiT/VAE -> PE_Only flip is available.")
    lines.append(
        f"- Restricted final throughput: {restricted_throughput:.4f} req/s; "
        f"fragment-filled target throughput: {target_completed_throughput:.4f} req/s; gap={gap:.4f} req/s."
    )
    lines.append(
        f"- Restricted stage throughput: PE={restricted_caps.get('PE', 0.0):.4f}, "
        f"DiT={restricted_caps.get('DiT', 0.0):.4f}, VAE={restricted_caps.get('VAE', 0.0):.4f}."
    )
    lines.append(f"- Restricted final summary: {_summarize_counter(restricted_counter)}")
    lines.append(f"- Throughput arithmetic: {_counter_throughput_arithmetic(catalog, target.model, restricted_counter, target.required_stages)}")
    lines.append("Result 2: standard source -> target minimum-difference plan")
    for action in _standard_diff_actions(source_counter, target_counter):
        lines.append(f"- {action}")
    lines.append("Base ILP core diff before fragment completion")
    for action in _standard_diff_actions(source_base_counter, target_base_counter):
        lines.append(f"- {action}")
    lines.append(f"- Source summary (fragment-filled): {_summarize_counter(source_counter)}")
    lines.append(f"- Target summary (fragment-filled): {_summarize_counter(target_counter)}")
    lines.append(f"- Source summary (base ILP core): {_summarize_counter(source_base_counter)}")
    lines.append(f"- Target summary (base ILP core): {_summarize_counter(target_base_counter)}")
    lines.append("")
    return lines


def _restricted_flip_counter(
    catalog: ProfileCatalog,
    model: ModelSelection,
    source_counter,
    target_counter,
):
    source_extra = source_counter - target_counter
    target_extra = target_counter - source_counter
    pe_deficits = [
        (key, count)
        for key, count in sorted(target_extra.items(), key=lambda item: (item[0][2], item[0][1]))
        if key[0] == TEMPLATE_PE_ONLY
    ]
    convertible = [
        (key, count)
        for key, count in sorted(source_extra.items(), key=lambda item: (item[0][2], item[0][0], item[0][1]))
        if key[0] in {TEMPLATE_DIT_ONLY, TEMPLATE_DIT_VAE_COMB, TEMPLATE_VAE_ONLY}
    ]
    flip_options: list[tuple[TemplateKey, int, TemplateKey, int]] = []
    for source_key, source_count in convertible:
        source_template, source_parallelism, source_hardware = source_key
        for pe_key, _pe_needed in pe_deficits:
            _pe_template, pe_parallelism, pe_hardware = pe_key
            if source_hardware != pe_hardware or source_parallelism < pe_parallelism:
                continue
            if source_parallelism % pe_parallelism != 0:
                continue
            pe_per_source = source_parallelism // pe_parallelism
            if pe_per_source > 0:
                flip_options.append((source_key, source_count, pe_key, pe_per_source))
                break

    if not flip_options:
        return +Counter(source_counter), []

    best_counter: Counter[TemplateKey] | None = None
    best_actions: list[str] = []
    best_score: tuple[float, float, int, int] | None = None

    ranges = [range(source_count + 1) for _source_key, source_count, _pe_key, _pe_per_source in flip_options]
    for counts in product(*ranges):
        final_counter: Counter[TemplateKey] = Counter(source_counter)
        actions: list[str] = []
        total_flips = 0
        for flip_count, (source_key, _source_count, pe_key, pe_per_source) in zip(counts, flip_options):
            if flip_count <= 0:
                continue
            source_template, source_parallelism, source_hardware = source_key
            _pe_template, pe_parallelism, pe_hardware = pe_key
            pe_created = flip_count * pe_per_source
            final_counter[source_key] -= flip_count
            final_counter[pe_key] += pe_created
            total_flips += flip_count
            actions.append(
                f"Flip {flip_count}x {source_template}_{source_hardware}_P{source_parallelism} "
                f"-> {pe_created}x PE_Only_{pe_hardware}_P{pe_parallelism}."
            )
        final_counter = +final_counter
        caps = _counter_stage_caps(catalog, model, final_counter)
        throughput = min(caps[stage] for stage in model_required_stages(model))
        pe_dit_gap = abs(caps.get("PE", 0.0) - caps.get("DiT", 0.0))
        pe_over_target = sum(
            max(0, final_counter[key] - target_counter.get(key, 0))
            for key in final_counter
            if key[0] == TEMPLATE_PE_ONLY
        )
        score = (throughput, -pe_dit_gap, -total_flips, -pe_over_target)
        if best_score is None or score > best_score:
            best_score = score
            best_counter = final_counter
            best_actions = actions

    assert best_counter is not None
    if not best_actions:
        best_actions = ["No restricted flip improves the stage balance."]
    return best_counter, best_actions


def _standard_diff_actions(source_counter, target_counter) -> list[str]:
    source_extra = source_counter - target_counter
    target_extra = target_counter - source_counter
    actions: list[str] = []
    hardware = sorted({key[2] for key in source_extra} | {key[2] for key in target_extra})
    for gpu_type in hardware:
        lhs = [(key, count) for key, count in source_extra.items() if key[2] == gpu_type]
        rhs = [(key, count) for key, count in target_extra.items() if key[2] == gpu_type]
        if not lhs and not rhs:
            continue
        lhs_text = " + ".join(f"{count}x {key[0]}_{gpu_type}_P{key[1]}" for key, count in sorted(lhs))
        rhs_text = " + ".join(f"{count}x {key[0]}_{gpu_type}_P{key[1]}" for key, count in sorted(rhs))
        lhs_gpus = sum(count * key[1] for key, count in lhs)
        rhs_gpus = sum(count * key[1] for key, count in rhs)
        gpu_text = f"GPU slots {lhs_gpus} -> {rhs_gpus}"
        if lhs and rhs:
            actions.append(f"Repartition {gpu_type}: {lhs_text} -> {rhs_text}. ({gpu_text})")
        elif lhs:
            actions.append(f"Remove from {gpu_type}: {lhs_text}. ({gpu_text})")
        else:
            actions.append(f"Add on {gpu_type}: {rhs_text}. ({gpu_text})")
    if not actions:
        actions.append("Source and target template counters are identical.")
    return actions


def _counter_stage_caps(catalog: ProfileCatalog, model: ModelSelection, counter) -> dict[str, float]:
    caps = {stage: 0.0 for stage in model_required_stages(model)}
    for (template_name, parallelism, hardware), count in counter.items():
        if count <= 0:
            continue
        template = _lookup_template(catalog, model, hardware, parallelism, template_name)
        for stage, throughput in template.stage_throughputs.items():
            caps[stage] = caps.get(stage, 0.0) + throughput * count
    return caps


def model_required_stages(_model: ModelSelection) -> tuple[str, ...]:
    return ("PE", "DiT", "VAE")


def _lookup_template(
    catalog: ProfileCatalog,
    model: ModelSelection,
    hardware: str,
    parallelism: int,
    template_name: str,
):
    for template in catalog.template_options(model, hardware, parallelism, search_kind=SEARCH_KIND_PRUNED):
        if template.name == template_name:
            return template
    raise KeyError(f"Cannot find template {template_name}_{hardware}_P{parallelism}")


def _fragment_completion_options(catalog: ProfileCatalog, result: SearchResult) -> tuple[FragmentCompletion, FragmentCompletion]:
    return (
        _fragment_completion(catalog, result, "same_model"),
        _fragment_completion(catalog, result, "minimum_waste"),
    )


def _fragment_completion(catalog: ProfileCatalog, result: SearchResult, strategy: str) -> FragmentCompletion:
    counter: Counter[TemplateKey] = Counter(result_template_counter(result))
    base_capacity = _active_capacity_by_hardware(result)
    current_caps = _counter_stage_caps(catalog, result.model, counter)
    actions: list[str] = []
    for node, used_bundles in _used_bundles_by_node(result).items():
        used_gpus = sum(bundle_size for _template_name, bundle_size, _hardware in used_bundles)
        remaining = node.gpu_count - used_gpus
        if remaining <= 0:
            continue
        if strategy == "same_model":
            fill_counter = _same_model_fill_for_node(catalog, result.model, node, remaining, used_bundles, current_caps)
        elif strategy == "minimum_waste":
            fill_counter = _minimum_waste_fill_for_node(catalog, result.model, node, remaining, current_caps)
        else:
            raise ValueError(f"Unsupported fragment completion strategy: {strategy}")
        if not fill_counter:
            actions.append(f"{node.name}: leave {remaining} GPU fragment unused; no legal filler template.")
            continue
        counter.update(fill_counter)
        current_caps = _counter_stage_caps(catalog, result.model, counter)
        actions.append(f"{node.name}: fill {remaining} GPU fragment with {_summarize_counter(fill_counter)}.")

    stage_caps = _counter_stage_caps(catalog, result.model, counter)
    throughput = min(stage_caps[stage] for stage in result.required_stages)
    return FragmentCompletion(
        name="node-local same-model first" if strategy == "same_model" else "minimum wasted throughput first",
        counter=+counter,
        actions=tuple(actions),
        stage_throughputs=stage_caps,
        throughput=throughput,
        utilization=_completion_utilization(counter, base_capacity),
    )


def _used_bundles_by_node(result: SearchResult) -> dict[NodeSpec, list[tuple[str, int, str]]]:
    node_by_name = {node.name: node for node in result.group.nodes}
    seen: set[tuple[str, str, int, str]] = set()
    grouped: dict[NodeSpec, list[tuple[str, int, str]]] = defaultdict(list)
    for instance in result.expanded_instances:
        key = (instance.node_name, instance.bundle_label, instance.bundle_size, instance.template_name)
        if key in seen:
            continue
        seen.add(key)
        node = node_by_name[instance.node_name]
        grouped[node].append((instance.template_name, instance.bundle_size, instance.gpu_type))
    return grouped


def _same_model_fill_for_node(
    catalog: ProfileCatalog,
    model: ModelSelection,
    node: NodeSpec,
    remaining: int,
    used_bundles: list[tuple[str, int, str]],
    current_caps: dict[str, float],
) -> Counter[TemplateKey]:
    fill_counter: Counter[TemplateKey] = Counter()
    existing = sorted(set(used_bundles), key=lambda item: (-item[1], item[0]))
    while remaining > 0:
        placed = False
        for template_name, bundle_size, hardware in existing:
            if bundle_size > remaining:
                continue
            template = _lookup_legal_template(catalog, model, node, bundle_size, template_name)
            if template is None:
                continue
            fill_counter[(template.name, bundle_size, hardware)] += 1
            remaining -= bundle_size
            placed = True
            break
        if not placed:
            fill_counter.update(_minimum_waste_fill_for_node(catalog, model, node, remaining, current_caps))
            break
    return +fill_counter


def _minimum_waste_fill_for_node(
    catalog: ProfileCatalog,
    model: ModelSelection,
    node: NodeSpec,
    remaining: int,
    current_caps: dict[str, float],
) -> Counter[TemplateKey]:
    candidates = _exact_fill_candidates(catalog, model, node, remaining)
    if not candidates:
        return Counter()
    current_throughput = min(current_caps[stage] for stage in model_required_stages(model))
    best_counter: Counter[TemplateKey] | None = None
    best_score: tuple[float, float, int] | None = None
    for candidate in candidates:
        increments = _counter_stage_caps(catalog, model, candidate)
        final_caps = {
            stage: current_caps.get(stage, 0.0) + increments.get(stage, 0.0)
            for stage in model_required_stages(model)
        }
        final_throughput = min(final_caps.values())
        added_capacity = sum(increments.get(stage, 0.0) for stage in model_required_stages(model))
        instance_count = sum(candidate.values())
        score = (final_throughput - current_throughput, -added_capacity, -instance_count)
        if best_score is None or score > best_score:
            best_score = score
            best_counter = candidate
    assert best_counter is not None
    return +best_counter


def _exact_fill_candidates(
    catalog: ProfileCatalog,
    model: ModelSelection,
    node: NodeSpec,
    remaining: int,
) -> tuple[Counter[TemplateKey], ...]:
    seen: set[tuple[tuple[TemplateKey, int], ...]] = set()
    candidates: list[Counter[TemplateKey]] = []
    for partition in bundle_partitions(remaining):
        choices_by_bundle: list[tuple[TemplateProfile, ...]] = []
        for bundle_size in partition:
            choices = tuple(
                template
                for template in catalog.template_options(model, node.gpu_type, bundle_size, search_kind=SEARCH_KIND_PRUNED)
                if template.memory_gb <= bundle_size * node.gpu_memory_gb
            )
            if not choices:
                break
            choices_by_bundle.append(choices)
        if len(choices_by_bundle) != len(partition):
            continue
        for templates in product(*choices_by_bundle):
            counter: Counter[TemplateKey] = Counter((template.name, template.bundle_size, node.gpu_type) for template in templates)
            signature = tuple(sorted(counter.items()))
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(counter)
    return tuple(candidates)


def _lookup_legal_template(
    catalog: ProfileCatalog,
    model: ModelSelection,
    node: NodeSpec,
    bundle_size: int,
    template_name: str,
) -> TemplateProfile | None:
    for template in catalog.template_options(model, node.gpu_type, bundle_size, search_kind=SEARCH_KIND_PRUNED):
        if template.name == template_name and template.memory_gb <= bundle_size * node.gpu_memory_gb:
            return template
    return None


def _base_utilization(result: SearchResult) -> dict[str, tuple[int, int]]:
    used: defaultdict[str, int] = defaultdict(int)
    for bundles in _used_bundles_by_node(result).values():
        for _template_name, bundle_size, hardware in bundles:
            used[hardware] += bundle_size
    capacity = _active_capacity_by_hardware(result)
    return {hardware: (used.get(hardware, 0), capacity[hardware]) for hardware in sorted(capacity)}


def _active_capacity_by_hardware(result: SearchResult) -> dict[str, int]:
    capacity: defaultdict[str, int] = defaultdict(int)
    for node in _used_bundles_by_node(result):
        capacity[node.gpu_type] += node.gpu_count
    return dict(capacity)


def _completion_utilization(counter, base_capacity: dict[str, int]) -> dict[str, tuple[int, int]]:
    used: defaultdict[str, int] = defaultdict(int)
    for (_template_name, parallelism, hardware), count in counter.items():
        used[hardware] += parallelism * count
    return {hardware: (used.get(hardware, 0), base_capacity[hardware]) for hardware in sorted(base_capacity)}


def _format_utilization(utilization: dict[str, tuple[int, int]]) -> str:
    if not utilization:
        return "no active GPU nodes"
    return "; ".join(f"{hardware}={used}/{capacity}" for hardware, (used, capacity) in sorted(utilization.items()))


def _result_throughput_arithmetic(catalog: ProfileCatalog, result: SearchResult) -> str:
    profile_counts: Counter[tuple[str, str, str, int, float, float]] = Counter()
    for instance in result.expanded_instances:
        key = (
            instance.stage,
            instance.template_name,
            instance.gpu_type,
            instance.parallelism,
            instance.latency_s,
            instance.throughput,
        )
        profile_counts[key] += 1
    profiles = [
        (count, StageProfile(stage, hardware, parallelism, latency_s, 0.0, "", ""), template_name)
        for (stage, template_name, hardware, parallelism, latency_s, _throughput), count in profile_counts.items()
    ]
    return _throughput_arithmetic(catalog, result.model, profiles, result.required_stages)


def _counter_throughput_arithmetic(
    catalog: ProfileCatalog,
    model: ModelSelection,
    counter,
    required_stages: tuple[str, ...],
) -> str:
    profiles: list[tuple[int, StageProfile, str]] = []
    for (template_name, parallelism, hardware), count in sorted(counter.items()):
        template = _lookup_template(catalog, model, hardware, parallelism, template_name)
        for profile in template.stage_profiles:
            profiles.append((count, profile, template.name))
    return _throughput_arithmetic(catalog, model, profiles, required_stages)


def _throughput_arithmetic(
    catalog: ProfileCatalog,
    model: ModelSelection,
    profiles: list[tuple[int, StageProfile, str]],
    required_stages: tuple[str, ...],
) -> str:
    instance_terms = []
    stage_terms: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for count, profile, template_name in sorted(
        profiles,
        key=lambda item: (item[1].stage, item[2], item[1].gpu_type, item[1].parallelism),
    ):
        instance_terms.append(_profile_formula(catalog, model, template_name, profile, count))
        stage_terms[profile.stage].append((count, profile.throughput))
    stage_sums = []
    stage_totals = {}
    for stage in required_stages:
        terms = stage_terms.get(stage, [])
        total = sum(count * throughput for count, throughput in terms)
        stage_totals[stage] = total
        expression = " + ".join(f"{count}*{throughput:.4f}" for count, throughput in terms) or "0"
        stage_sums.append(f"{stage}={expression}={total:.4f}")
    overall = min(stage_totals.values()) if stage_totals else 0.0
    return "; ".join(instance_terms + [", ".join(stage_sums), f"overall=min(...)={overall:.4f} req/s"])


def _profile_formula(
    catalog: ProfileCatalog,
    model: ModelSelection,
    template_name: str,
    profile: StageProfile,
    count: int,
) -> str:
    label = f"{count}x {template_name}_{profile.gpu_type}_P{profile.parallelism} {profile.stage}"
    if profile.stage == "PE":
        pe_model = catalog.profile_data.pe_models[model.pe_model.lower().replace("-", "")]
        hardware = pe_model.hardware[profile.gpu_type]
        ttft = hardware.ttft_ms[model.input_tokens][model.output_tokens][profile.parallelism]
        tpot = hardware.tpot_ms[model.input_tokens][model.output_tokens][profile.parallelism]
        return (
            f"{label}: latency=({ttft:.5f}+({model.output_tokens}-1)*{tpot:.6f})/1000"
            f"={profile.latency_s:.4f}s, thpt=1/{profile.latency_s:.4f}={profile.throughput:.4f}"
        )
    return f"{label}: latency={profile.latency_s:.4f}s, thpt=1/{profile.latency_s:.4f}={profile.throughput:.4f}"


def render_lcm_report(
    catalog: ProfileCatalog,
    base_model: ModelSelection,
    lcm_results: dict[tuple[str, int, int], list[LcmScenarioResult]],
    output_path: Path = LCM_REPORT_PATH,
) -> str:
    lines: list[str] = []
    lines.append("TD Extra LCM Planner Report")
    lines.append("=" * 31)
    lines.append("")
    lines.append(
        f"Model: {base_model.pe_model} + {canonical_generator_name(base_model.generator_model)}; "
        f"input tokens={base_model.input_tokens}"
    )
    lines.append("Templates use PE_Only, Encoder_CPU, DiT_Only, DiT_VAE_Comb, and VAE_Only according to Case A/B.")
    lines.append("This report is an independent resource-demand estimate and is not capped by Group 1/2 capacity.")
    lines.append("")

    for key in sorted(lcm_results):
        pe_hardware, pe_bundle_size, pe_node_count = key
        lines.append(f"Scenario: PE={pe_hardware}, PE bundle={pe_bundle_size}, PE nodes={pe_node_count}")
        for result in lcm_results[key]:
            lines.append(f"- {result.case_name}")
            lines.extend(_render_lcm_phase("Phase 1 output tokens == 512", result.phase1))
            lines.extend(_render_lcm_phase("Phase 2 before flip, output tokens == 2048", result.phase2_before_flip))
            flip = result.phase2_after_flip
            lines.append(
                f"  * Phase 2 balanced flip: flip {flip.flipped_dit_only} DiT_Only and "
                f"{flip.flipped_dit_vae_comb} DiT_VAE_Comb into {flip.added_pe_instances} H100 PE_Only instances."
            )
            lines.append(
                f"  * After flip throughput={flip.throughput:.4f} req/s; "
                f"PE={flip.stage_throughputs['PE']:.4f}, DiT={flip.stage_throughputs['DiT']:.4f}, "
                f"VAE={flip.stage_throughputs['VAE']:.4f}, Encoder={flip.stage_throughputs['Encoder']:.4f}."
            )
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


def _render_lcm_phase(title: str, phase) -> list[str]:
    lines = [f"  * {title}: throughput={phase.throughput:.4f} req/s; memory feasible={phase.memory_feasible}."]
    for stage in (phase.pe_stage, phase.encoder_stage, phase.dit_only_stage, phase.dit_vae_comb_stage, phase.vae_only_stage):
        if stage is None:
            continue
        lines.append(
            f"    - {stage.name}: hardware={stage.hardware}, parallelism={stage.parallelism}, "
            f"instances={stage.instances}, nodes={stage.nodes}, latency={stage.per_instance_latency_s:.4f}s, "
            f"stage throughput={stage.stage_throughput:.4f} req/s"
        )
    return lines


def summarize_stage_instances(result: SearchResult) -> dict[str, Counter[str]]:
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for instance in result.expanded_instances:
        summary[instance.stage][f"{instance.template_name}_{instance.gpu_type}_P{instance.parallelism}"] += 1
    return summary
