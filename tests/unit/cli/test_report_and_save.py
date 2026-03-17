# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from aiconfigurator.cli.report_and_save import _format_model_path_for_result_prefix

pytestmark = pytest.mark.unit


def test_format_model_path_for_result_prefix_uses_basename_for_absolute_paths():
    label = _format_model_path_for_result_prefix("/home/heyang/models/promptenhancer-7b")
    assert label == "promptenhancer-7b"


def test_format_model_path_for_result_prefix_keeps_hf_namespace_readable():
    label = _format_model_path_for_result_prefix("Qwen/Qwen3-32B")
    assert label == "Qwen_Qwen3-32B"


def test_format_model_path_for_result_prefix_never_contains_separators():
    label = _format_model_path_for_result_prefix(r"C:\models\promptenhancer-7b")
    assert "/" not in label
    assert "\\" not in label


def test_safe_label_keeps_join_under_save_dir():
    save_dir = "/tmp/aic-output"
    label = _format_model_path_for_result_prefix("/home/heyang/models/promptenhancer-7b")
    result_dir = os.path.join(save_dir, f"{label}_123")
    assert result_dir.startswith(save_dir)
