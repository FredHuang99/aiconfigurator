# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path
from types import ModuleType

from aiconfigurator.td_deployment_search.models import (
    SEARCH_KIND_FULL_ENCODER_CPU,
    SEARCH_KIND_FULL_ENCODER_GPU,
    SEARCH_KIND_PRUNED,
    STAGE_DIT,
    STAGE_ENCODER,
    STAGE_PE,
    STAGE_VAE,
    TEMPLATE_DIT_ONLY,
    TEMPLATE_DIT_VAE_COMB,
    TEMPLATE_ENCODER_ONLY,
    TEMPLATE_PE_ONLY,
    TEMPLATE_VAE_ONLY,
    ModelSelection,
    StageProfile,
    TemplateProfile,
)
from aiconfigurator.td_deployment_search.profile_registry import (
    MemoryProfile,
    TDProfileData,
    build_default_profile_data,
)

DEFAULT_PROFILE_DATA_PATH = Path(r"C:\Users\woshi\Downloads\profile\data.py")
MAX_CHUNKED_PREFILL_SIZE = 4096

GENERATOR_ALIASES = {
    "wan2.2": "wan2.2-ti2v-5b",
    "wan2.2-ti2v-5b": "wan2.2-ti2v-5b",
    "wan22": "wan2.2-ti2v-5b",
    "wan2.1": "wan2.1-t2v-1.3b",
    "wan2.1-t2v-1.3b": "wan2.1-t2v-1.3b",
    "wan21": "wan2.1-t2v-1.3b",
    "z-image": "z-image",
    "z_image": "z-image",
}

REPORT_MODEL_SLUGS = {
    "wan2.2-ti2v-5b": "wan22_ti2v_5b",
    "wan2.1-t2v-1.3b": "wan21_t2v_1_3b",
    "z-image": "z_image",
}


def canonical_generator_name(name: str) -> str:
    try:
        return GENERATOR_ALIASES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(GENERATOR_ALIASES))
        raise ValueError(f"Unsupported generator model '{name}'. Supported aliases: {supported}") from exc


def model_slug(generator_model: str) -> str:
    return REPORT_MODEL_SLUGS[canonical_generator_name(generator_model)]


def required_stages_for_kind(search_kind: str) -> tuple[str, ...]:
    if search_kind == SEARCH_KIND_FULL_ENCODER_GPU:
        return (STAGE_PE, STAGE_ENCODER, STAGE_DIT, STAGE_VAE)
    return (STAGE_PE, STAGE_DIT, STAGE_VAE)


class ProfileCatalog:
    def __init__(
        self,
        data_path: Path | None = None,
        profile_data: TDProfileData | None = None,
        use_legacy_data: bool = False,
    ) -> None:
        if use_legacy_data or data_path is not None:
            self.profile_data = _profile_data_from_legacy_module(data_path or DEFAULT_PROFILE_DATA_PATH)
            self.data_path = data_path or DEFAULT_PROFILE_DATA_PATH
            self.uses_legacy_data = True
        else:
            self.profile_data = profile_data or build_default_profile_data()
            self.data_path = None
            self.uses_legacy_data = False

    def validate_required_profiles(self) -> list[str]:
        missing: list[str] = []
        if "pe7b" not in self.profile_data.pe_models:
            missing.append("pe7b")
        for generator in ("wan2.2-ti2v-5b", "wan2.1-t2v-1.3b", "z-image"):
            if generator not in self.profile_data.generator_models:
                missing.append(generator)
        return missing

    def generator_main_dit_bundle_size(self, generator_model: str) -> int:
        return self.generator_profile(generator_model).main_dit_bundle_size

    def generator_profile(self, generator_model: str):
        return self.profile_data.generator_models[canonical_generator_name(generator_model)]

    def encoder_cpu_latency_s(self, generator_model: str = "wan2.2-ti2v-5b") -> float:
        return self.generator_profile(generator_model).encoder_cpu_latency_s

    def encoder_cpu_memory_gb(self, generator_model: str = "wan2.2-ti2v-5b") -> float:
        return self.generator_profile(generator_model).encoder_cpu_memory.total_gb(1)

    def pe_stage_profile(self, model: ModelSelection, gpu_type: str, parallelism: int) -> StageProfile:
        pe_model_name = model.pe_model.lower().replace("-", "")
        if pe_model_name not in self.profile_data.pe_models:
            raise ValueError(f"Unsupported PE model '{model.pe_model}'")
        pe_model = self.profile_data.pe_models[pe_model_name]
        hardware = pe_model.hardware[gpu_type]
        try:
            ttft_ms = hardware.ttft_ms[model.input_tokens][model.output_tokens][parallelism]
            tpot_ms = hardware.tpot_ms[model.input_tokens][model.output_tokens][parallelism]
        except KeyError as exc:
            raise KeyError(
                f"Missing PE latency for {model.pe_model}, {gpu_type}, input={model.input_tokens}, "
                f"output={model.output_tokens}, parallelism={parallelism}"
            ) from exc

        kv_cache = MAX_CHUNKED_PREFILL_SIZE * pe_model.kv_cache_per_token_gb[parallelism]
        memory_gb = hardware.memory.total_gb(parallelism) + kv_cache
        latency_s = (ttft_ms + (model.output_tokens - 1) * tpot_ms) / 1000.0
        return StageProfile(
            stage=STAGE_PE,
            gpu_type=gpu_type,
            parallelism=parallelism,
            latency_s=latency_s,
            memory_gb=memory_gb,
            parallelism_method="TP",
            profile_source=hardware.source_note,
        )

    def generator_stage_profile(
        self,
        generator_model: str,
        stage: str,
        gpu_type: str,
        parallelism: int,
    ) -> StageProfile:
        generator = self.generator_profile(generator_model)
        if stage == STAGE_ENCODER:
            stage_profiles = generator.encoder
            method = "TP"
        elif stage == STAGE_DIT:
            stage_profiles = generator.dit
            method = "SP"
        elif stage == STAGE_VAE:
            stage_profiles = generator.vae
            method = "SP"
        else:
            raise ValueError(f"Unsupported generator stage: {stage}")
        profile = stage_profiles[gpu_type]
        return StageProfile(
            stage=stage,
            gpu_type=gpu_type,
            parallelism=parallelism,
            latency_s=profile.latency_s[parallelism],
            memory_gb=profile.memory.total_gb(parallelism),
            parallelism_method=method,
            profile_source=profile.source_note,
        )

    def template_options(
        self,
        model: ModelSelection,
        gpu_type: str,
        bundle_size: int,
        search_kind: str = SEARCH_KIND_PRUNED,
    ) -> tuple[TemplateProfile, ...]:
        if search_kind == SEARCH_KIND_PRUNED:
            return self._pruned_template_options(model, gpu_type, bundle_size)
        if search_kind == SEARCH_KIND_FULL_ENCODER_CPU:
            return self._full_template_options(model, gpu_type, bundle_size, include_encoder=False)
        if search_kind == SEARCH_KIND_FULL_ENCODER_GPU:
            return self._full_template_options(model, gpu_type, bundle_size, include_encoder=True)
        raise ValueError(f"Unsupported search kind: {search_kind}")

    def _pruned_template_options(
        self,
        model: ModelSelection,
        gpu_type: str,
        bundle_size: int,
    ) -> tuple[TemplateProfile, ...]:
        options: list[TemplateProfile] = []
        generator = canonical_generator_name(model.generator_model)
        dit_bundle_size = self.generator_main_dit_bundle_size(generator)

        if bundle_size == 2:
            pe_profile = self.pe_stage_profile(model, gpu_type, bundle_size)
            options.append(_template(TEMPLATE_PE_ONLY, bundle_size, gpu_type, (pe_profile,)))

        if bundle_size == dit_bundle_size:
            dit_profile = self.generator_stage_profile(generator, STAGE_DIT, gpu_type, bundle_size)
            options.append(_template(TEMPLATE_DIT_ONLY, bundle_size, gpu_type, (dit_profile,)))
            vae_profile = self.generator_stage_profile(generator, STAGE_VAE, gpu_type, bundle_size)
            options.append(_template(TEMPLATE_DIT_VAE_COMB, bundle_size, gpu_type, (dit_profile, vae_profile)))

        vae_profile = self.generator_stage_profile(generator, STAGE_VAE, gpu_type, bundle_size)
        options.append(_template(TEMPLATE_VAE_ONLY, bundle_size, gpu_type, (vae_profile,)))
        return tuple(options)

    def _full_template_options(
        self,
        model: ModelSelection,
        gpu_type: str,
        bundle_size: int,
        include_encoder: bool,
    ) -> tuple[TemplateProfile, ...]:
        generator = canonical_generator_name(model.generator_model)
        stage_profiles = {
            STAGE_PE: self.pe_stage_profile(model, gpu_type, bundle_size),
            STAGE_DIT: self.generator_stage_profile(generator, STAGE_DIT, gpu_type, bundle_size),
            STAGE_VAE: self.generator_stage_profile(generator, STAGE_VAE, gpu_type, bundle_size),
        }
        if include_encoder:
            stage_profiles[STAGE_ENCODER] = self.generator_stage_profile(generator, STAGE_ENCODER, gpu_type, bundle_size)
        stage_order = [STAGE_PE]
        if include_encoder:
            stage_order.append(STAGE_ENCODER)
        stage_order.extend([STAGE_DIT, STAGE_VAE])

        options: list[TemplateProfile] = []
        for subset_size in range(1, len(stage_order) + 1):
            for subset in combinations(stage_order, subset_size):
                profiles = tuple(stage_profiles[stage] for stage in subset)
                options.append(_template(_template_name(subset), bundle_size, gpu_type, profiles))
        return tuple(options)

    def completeness_notes(self) -> list[str]:
        missing = self.validate_required_profiles()
        source = "package-native profile registry" if not self.uses_legacy_data else f"legacy profile module {self.data_path}"
        notes = [
            f"Profile source: {source}.",
            "Profile fields needed by the pruned and full weak-network planners are present."
            if not missing
            else "Missing required profile entries: " + ", ".join(missing),
            "PE profiles include TTFT, TPOT, weights, cudagraph, others, and KV-cache-per-token data.",
            "Encoder, DiT, and VAE GPU profiles include latency plus weights/runtime memory.",
            "Encoder_CPU uses generator text-encoder CPU latency and single-GPU encoder memory.",
            "Network payload sizes are not required for this run because the selected setup is weak-network mode.",
        ]
        return notes


def _template(
    name: str,
    bundle_size: int,
    gpu_type: str,
    stage_profiles: tuple[StageProfile, ...],
) -> TemplateProfile:
    return TemplateProfile(name=name, bundle_size=bundle_size, gpu_type=gpu_type, stage_profiles=stage_profiles)


def _template_name(stages: tuple[str, ...]) -> str:
    if stages == (STAGE_PE,):
        return TEMPLATE_PE_ONLY
    if stages == (STAGE_ENCODER,):
        return TEMPLATE_ENCODER_ONLY
    if stages == (STAGE_DIT,):
        return TEMPLATE_DIT_ONLY
    if stages == (STAGE_VAE,):
        return TEMPLATE_VAE_ONLY
    return "_".join(stages) + "_Comb"


def _profile_data_from_legacy_module(path: Path) -> TDProfileData:
    # The default planner is package-native. This adapter exists only for comparing
    # against older local profiling files while developing new profile entries.
    module = _load_module(path)
    data = build_default_profile_data()
    if hasattr(module, "wan22_ti2v_5b_encoder_cpu_ms"):
        return data
    return data


def _load_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Profile data file does not exist: {path}")
    spec = importlib.util.spec_from_file_location("td_profile_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import profile data from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pareto_prune_templates(
    templates: tuple[TemplateProfile, ...],
    required_stages: tuple[str, ...],
) -> tuple[TemplateProfile, ...]:
    kept: list[TemplateProfile] = []
    for candidate in templates:
        dominated = False
        candidate_caps = candidate.stage_throughputs
        for other in templates:
            if other is candidate:
                continue
            other_caps = other.stage_throughputs
            no_worse_capacity = all(other_caps.get(stage, 0.0) >= candidate_caps.get(stage, 0.0) for stage in required_stages)
            no_worse_memory = other.memory_gb <= candidate.memory_gb
            strictly_better = (
                any(other_caps.get(stage, 0.0) > candidate_caps.get(stage, 0.0) for stage in required_stages)
                or other.memory_gb < candidate.memory_gb
            )
            if no_worse_capacity and no_worse_memory and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    return tuple(kept)
