# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profiler-backed TD deployment search helpers."""

from aiconfigurator.td_deployment_search.models import (
    BundleAssignment,
    BundleSpec,
    ModelSelection,
    NodeConfig,
    NodeSpec,
    ResourceGroupSpec,
    SearchResult,
    StageProfile,
    TemplateProfile,
)
try:
    from aiconfigurator.td_deployment_search.planner import (
        build_default_resource_groups,
        solve_group,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scipy":
        raise
    _planner_import_error = exc

    def build_default_resource_groups(*args, **kwargs):
        raise ModuleNotFoundError("scipy is required for TD deployment planning") from _planner_import_error

    def solve_group(*args, **kwargs):
        raise ModuleNotFoundError("scipy is required for TD deployment planning") from _planner_import_error
from aiconfigurator.td_deployment_search.profiles import (
    DEFAULT_PROFILE_DATA_PATH,
    ProfileCatalog,
)

__all__ = [
    "DEFAULT_PROFILE_DATA_PATH",
    "BundleAssignment",
    "BundleSpec",
    "ModelSelection",
    "NodeConfig",
    "NodeSpec",
    "ProfileCatalog",
    "ResourceGroupSpec",
    "SearchResult",
    "StageProfile",
    "TemplateProfile",
    "build_default_resource_groups",
    "solve_group",
]
