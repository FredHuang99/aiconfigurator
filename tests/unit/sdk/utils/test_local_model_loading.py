# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for local model loading functionality.

Tests loading model configs from local directories, specifically for the
enhancer model at enhancer_models/tencent--HunyuanImage-2.1--reprompt.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aiconfigurator.sdk.utils import (
    _load_local_config,
    _load_local_quant_config,
    _load_model_config_from_model_path,
    _parse_hf_config_json,
    get_model_config_from_model_path,
)

pytestmark = pytest.mark.unit

# Path to the actual enhancer model
ENHANCER_MODEL_PATH = "/data/projects/myproject/aiconfigurator/enhancer_models/tencent--HunyuanImage-2.1--reprompt"


class TestLoadLocalConfig:
    """Test loading local config.json files."""

    def test_load_local_config_success(self):
        """Test successfully loading a local config.json."""
        config = _load_local_config(ENHANCER_MODEL_PATH)

        assert config is not None
        assert config["architectures"] == ["HunYuanDenseV1ForCausalLM"]
        assert config["num_hidden_layers"] == 32
        assert config["hidden_size"] == 4096
        assert config["num_attention_heads"] == 32
        assert config["num_key_value_heads"] == 8
        assert config["head_dim"] == 128
        assert config["intermediate_size"] == 14336
        assert config["vocab_size"] == 128167
        assert config["max_position_embeddings"] == 32768

    def test_load_local_config_file_not_found(self):
        """Test error handling when config.json doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="config.json not found"):
                _load_local_config(tmpdir)

    def test_load_local_config_invalid_json(self):
        """Test error handling for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("invalid json {")

            with pytest.raises(Exception):
                _load_local_config(tmpdir)


class TestLoadLocalQuantConfig:
    """Test loading local quantization config (hf_quant_config.json)."""

    def test_load_local_quant_config_exists(self):
        """Test loading hf_quant_config.json when it exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            quant_config = {"quantization": {"quant_algo": "fp8"}}
            quant_path = Path(tmpdir) / "hf_quant_config.json"
            quant_path.write_text(json.dumps(quant_config))

            result = _load_local_quant_config(tmpdir)

            assert result == quant_config

    def test_load_local_quant_config_not_exists(self):
        """Test loading hf_quant_config.json when it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _load_local_quant_config(tmpdir)

            assert result is None


class TestParseHFConfigForEnhancerModel:
    """Test parsing HuggingFace config for the enhancer model."""

    def test_parse_actual_enhancer_config(self):
        """Test parsing the actual enhancer model config.json."""
        raw_config = _load_local_config(ENHANCER_MODEL_PATH)
        parsed = _parse_hf_config_json(raw_config)

        # Verify all expected fields are parsed correctly
        assert parsed["architecture"] == "HunYuanDenseV1ForCausalLM"
        assert parsed["layers"] == 32
        assert parsed["n"] == 32  # num_attention_heads
        assert parsed["n_kv"] == 8  # num_key_value_heads (GQA)
        assert parsed["d"] == 128  # head_dim
        assert parsed["hidden_size"] == 4096
        assert parsed["inter_size"] == 14336  # intermediate_size
        assert parsed["vocab"] == 128167  # vocab_size
        assert parsed["context"] == 32768  # max_position_embeddings

        # This is a dense model, not MoE - but moe_inter_size falls back to inter_size
        assert parsed["topk"] == 0
        assert parsed["num_experts"] == 0
        # Note: For dense models, moe_inter_size falls back to intermediate_size
        assert parsed["moe_inter_size"] == 14336

    def test_parse_enhancer_config_layers_match(self):
        """Verify layer count matches expected for this model."""
        raw_config = _load_local_config(ENHANCER_MODEL_PATH)
        parsed = _parse_hf_config_json(raw_config)

        # HunYuanDenseV1ForCausalLM should have 32 layers
        assert parsed["layers"] == 32
        # This is a 7B class model (approximately)
        assert parsed["hidden_size"] == 4096


class TestLoadModelConfigFromModelPath:
    """Test the main entry point for loading model configs from local paths."""

    def test_load_from_enhancer_model_path(self):
        """Test loading raw config from the actual enhancer model local path."""
        result = _load_model_config_from_model_path(ENHANCER_MODEL_PATH)

        # Verify raw config fields (not parsed)
        assert result["architectures"] == ["HunYuanDenseV1ForCausalLM"]
        assert result["num_hidden_layers"] == 32
        assert result["num_attention_heads"] == 32
        assert result["num_key_value_heads"] == 8
        assert result["head_dim"] == 128
        assert result["hidden_size"] == 4096
        assert result["intermediate_size"] == 14336
        assert result["vocab_size"] == 128167
        assert result["max_position_embeddings"] == 32768

    def test_load_from_nonexistent_path_raises_error(self):
        """Test that loading from a non-existent path raises an error."""
        from aiconfigurator.sdk.utils import HuggingFaceDownloadError

        # A path starting with / that doesn't exist as a directory
        # will be treated as a HF ID and raise HuggingFaceDownloadError
        with pytest.raises(HuggingFaceDownloadError):
            _load_model_config_from_model_path("/nonexistent/path/to/model")

    def test_load_from_path_without_config_raises_error(self):
        """Test that loading from a directory without config.json raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="config.json not found"):
                _load_model_config_from_model_path(tmpdir)

    def test_load_from_hf_id_still_works(self):
        """Verify that HuggingFace ID loading still works alongside local paths."""
        # Mock the _download_hf_config to avoid actual network call
        mock_config = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
        }

        with patch(
            "aiconfigurator.sdk.utils._download_hf_config"
        ) as mock_download, patch(
            "aiconfigurator.sdk.utils._download_hf_json"
        ) as mock_download_quant:
            mock_download.return_value = mock_config
            mock_download_quant.return_value = None

            # This looks like an HF ID (no leading slash, no local path indicators)
            result = _load_model_config_from_model_path("organization/model-name")

            assert result["architectures"] == ["LlamaForCausalLM"]
            mock_download.assert_called_once_with("organization/model-name")


class TestGetModelConfigFromModelPath:
    """Test get_model_config_from_model_path which returns parsed config."""

    def test_get_parsed_config_from_enhancer_model(self):
        """Test getting parsed config from the enhancer model path."""
        result = get_model_config_from_model_path(ENHANCER_MODEL_PATH)

        # This should return parsed fields
        assert result["architecture"] == "HunYuanDenseV1ForCausalLM"
        assert result["layers"] == 32
        assert result["n"] == 32
        assert result["n_kv"] == 8
        assert result["d"] == 128
        assert result["hidden_size"] == 4096
        assert result["inter_size"] == 14336
        assert result["vocab"] == 128167
        assert result["context"] == 32768
        assert result["topk"] == 0
        assert result["num_experts"] == 0

        # Should also include raw_config
        assert "raw_config" in result
        assert result["raw_config"]["architectures"] == ["HunYuanDenseV1ForCausalLM"]


class TestEnhancerModelArchitectureMapping:
    """Test that the enhancer model architecture is properly mapped."""

    def test_hunyuan_dense_maps_to_llama_family(self):
        """Verify HunYuanDenseV1ForCausalLM maps to LLAMA model family."""
        from aiconfigurator.sdk.common import ARCHITECTURE_TO_MODEL_FAMILY

        assert (
            ARCHITECTURE_TO_MODEL_FAMILY["HunYuanDenseV1ForCausalLM"] == "LLAMA"
        )

    def test_hunyuan_dense_not_in_unknown_architectures(self):
        """Verify HunYuanDenseV1ForCausalLM is a known architecture."""
        from aiconfigurator.sdk.common import ARCHITECTURE_TO_MODEL_FAMILY

        assert "HunYuanDenseV1ForCausalLM" in ARCHITECTURE_TO_MODEL_FAMILY
        # Should not map to UNKNOWN
        assert ARCHITECTURE_TO_MODEL_FAMILY["HunYuanDenseV1ForCausalLM"] != "UNKNOWN"

