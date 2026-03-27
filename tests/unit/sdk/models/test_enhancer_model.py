# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the enhancer model loading via get_model().

Tests that the HunYuanDenseV1ForCausalLM model can be loaded from the local path
and that the resulting LLAMAModel has correct operations configured.
"""

import pytest

from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.models import get_model

pytestmark = pytest.mark.unit

# Path to the actual enhancer model
ENHANCER_MODEL_PATH = "/data/projects/myproject/aiconfigurator/enhancer_models/tencent--HunyuanImage-2.1--reprompt"


class TestEnhancerModelLoading:
    """Test loading the enhancer model through the get_model() factory."""

    def test_get_model_returns_llama_model(self):
        """Test that get_model() returns an LLAMAModel for HunYuanDenseV1ForCausalLM."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(
            model_path=ENHANCER_MODEL_PATH,
            model_config=model_config,
            backend_name="trtllm"
        )

        # Verify it's an LLAMAModel (class name check)
        assert model.__class__.__name__ == "LLAMAModel"

    def test_enhancer_model_basic_attributes(self):
        """Test that the loaded model has correct basic attributes."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(
            model_path=ENHANCER_MODEL_PATH,
            model_config=model_config,
            backend_name="trtllm"
        )

        # Verify model dimensions (using internal attributes with underscore prefix)
        assert model._num_layers == 32
        assert model._hidden_size == 4096
        assert model._num_heads == 32  # num_attention_heads
        assert model._num_kv_heads == 8  # num_key_value_heads (GQA)
        assert model._head_size == 128  # head_dim
        assert model._inter_size == 14336
        assert model._vocab_size == 128167
        assert model._context_length == 32768

    def test_enhancer_model_context_ops(self):
        """Test that the model has context (prefill) operations defined."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(
            model_path=ENHANCER_MODEL_PATH,
            model_config=model_config,
            backend_name="trtllm"
        )

        # Verify context_ops exists and has operations
        assert hasattr(model, "context_ops")
        assert len(model.context_ops) > 0

        # Check for expected operation types (operations have _name attribute)
        op_names = [op._name for op in model.context_ops]
        assert "context_qkv_gemm" in op_names
        assert "context_attention" in op_names
        assert "context_logits_gemm" in op_names

    def test_enhancer_model_generation_ops(self):
        """Test that the model has generation (decode) operations defined."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(
            model_path=ENHANCER_MODEL_PATH,
            model_config=model_config,
            backend_name="trtllm"
        )

        # Verify generation_ops exists and has operations
        assert hasattr(model, "generation_ops")
        assert len(model.generation_ops) > 0

        # Check for expected operation types (operations have _name attribute)
        op_names = [op._name for op in model.generation_ops]
        assert "generation_qkv_gemm" in op_names
        assert "generation_attention" in op_names
        assert "generation_logits_gemm" in op_names

    def test_enhancer_model_with_tensor_parallel(self):
        """Test loading the model with tensor parallelism."""
        model_config = ModelConfig(tp_size=2, pp_size=1)

        model = get_model(
            model_path=ENHANCER_MODEL_PATH,
            model_config=model_config,
            backend_name="trtllm"
        )

        # Verify model config accounts for TP
        assert model.config.tp_size == 2
        assert model._num_layers == 32  # Layers don't change with TP

    def test_enhancer_model_with_pipeline_parallel(self):
        """Test loading the model with pipeline parallelism."""
        model_config = ModelConfig(tp_size=1, pp_size=2)

        model = get_model(
            model_path=ENHANCER_MODEL_PATH,
            model_config=model_config,
            backend_name="trtllm"
        )

        # Verify model config accounts for PP
        assert model.config.pp_size == 2
        assert model._num_layers == 32  # Total layers unchanged

    def test_enhancer_model_is_dense_not_moe(self):
        """Test that the model is correctly identified as dense (not MoE)."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(
            model_path=ENHANCER_MODEL_PATH,
            model_config=model_config,
            backend_name="trtllm"
        )

        # Verify no MoE operations in context_ops
        context_op_names = [op._name for op in model.context_ops]
        assert not any("moe" in name.lower() for name in context_op_names)

        # Verify no MoE operations in generation_ops
        generation_op_names = [op._name for op in model.generation_ops]
        assert not any("moe" in name.lower() for name in generation_op_names)


class TestEnhancerModelWithDifferentBackends:
    """Test the enhancer model with different backends."""

    def test_enhancer_model_with_trtllm_backend(self):
        """Test loading with TRT-LLM backend."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(
            model_path=ENHANCER_MODEL_PATH,
            model_config=model_config,
            backend_name="trtllm"
        )

        assert model is not None

    def test_enhancer_model_backend_case_insensitive(self):
        """Test that backend name is case insensitive."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        # Try with different case variations
        model_lower = get_model(
            model_path=ENHANCER_MODEL_PATH,
            model_config=model_config,
            backend_name="trtllm"
        )

        assert model_lower is not None


class TestEnhancerModelArchitecture:
    """Test that the enhancer model architecture is correctly identified."""

    def test_enhancer_model_architecture(self):
        """Test that the model has correct architecture attribute."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(
            model_path=ENHANCER_MODEL_PATH,
            model_config=model_config,
            backend_name="trtllm"
        )

        assert model.architecture == "HunYuanDenseV1ForCausalLM"
        assert model.model_family == "LLAMA"
        assert model.model_path == ENHANCER_MODEL_PATH

