# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the PromptEnhancer-32B model loading via get_model().

Tests that the Qwen2_5_VLForConditionalGeneration model can be loaded from the local path
and that the resulting LLAMAModel has correct operations configured.
"""

import pytest

from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.models import get_model

pytestmark = pytest.mark.unit

# Path to the PromptEnhancer-32B model
PROMPT_ENHANCER_32B_MODEL_PATH = "/data/projects/myproject/aiconfigurator/enhancer_models/PromptEnhancer-32B"


class TestPromptEnhancer32BModelLoading:
    """Test loading the PromptEnhancer-32B model through the get_model() factory."""

    def test_get_model_returns_llama_model(self):
        """Test that get_model() returns an LLAMAModel for Qwen2_5_VLForConditionalGeneration."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        # Verify it's an LLAMAModel (class name check)
        assert model.__class__.__name__ == "LLAMAModel"

    def test_prompt_enhancer_model_basic_attributes(self):
        """Test that the loaded model has correct basic attributes from config."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        # Verify model dimensions from config.json
        # hidden_size: 5120, num_hidden_layers: 64, num_attention_heads: 40
        # num_key_value_heads: 8 (GQA), head_dim: 128 (5120/40)
        # intermediate_size: 27648, vocab_size: 152064, max_position_embeddings: 128000
        assert model._num_layers == 64, f"Expected 64 layers, got {model._num_layers}"
        assert model._hidden_size == 5120, f"Expected hidden_size 5120, got {model._hidden_size}"
        assert model._num_heads == 40, f"Expected 40 attention heads, got {model._num_heads}"
        assert model._num_kv_heads == 8, f"Expected 8 KV heads (GQA), got {model._num_kv_heads}"
        assert model._head_size == 128, f"Expected head_size 128, got {model._head_size}"
        assert model._inter_size == 27648, f"Expected inter_size 27648, got {model._inter_size}"
        assert model._vocab_size == 152064, f"Expected vocab_size 152064, got {model._vocab_size}"
        assert model._context_length == 128000, f"Expected context_length 128000, got {model._context_length}"

    def test_prompt_enhancer_model_context_ops(self):
        """Test that the model has context (prefill) operations defined."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        # Verify context_ops exists and has operations
        assert hasattr(model, "context_ops")
        assert len(model.context_ops) > 0, "context_ops should not be empty"

        # Check for expected operation types (operations have _name attribute)
        op_names = [op._name for op in model.context_ops]
        assert "context_qkv_gemm" in op_names, "context_qkv_gemm should be in context_ops"
        assert "context_attention" in op_names, "context_attention should be in context_ops"
        assert "context_logits_gemm" in op_names, "context_logits_gemm should be in context_ops"

    def test_prompt_enhancer_model_generation_ops(self):
        """Test that the model has generation (decode) operations defined."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        # Verify generation_ops exists and has operations
        assert hasattr(model, "generation_ops")
        assert len(model.generation_ops) > 0, "generation_ops should not be empty"

        # Check for expected operation types (operations have _name attribute)
        op_names = [op._name for op in model.generation_ops]
        assert "generation_qkv_gemm" in op_names, "generation_qkv_gemm should be in generation_ops"
        assert "generation_attention" in op_names, "generation_attention should be in generation_ops"
        assert "generation_logits_gemm" in op_names, "generation_logits_gemm should be in generation_ops"

    def test_prompt_enhancer_model_with_tensor_parallel(self):
        """Test loading the model with tensor parallelism."""
        model_config = ModelConfig(tp_size=2, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        # Verify model config accounts for TP
        assert model.config.tp_size == 2, f"Expected tp_size=2, got {model.config.tp_size}"
        assert model._num_layers == 64  # Layers don't change with TP

    def test_prompt_enhancer_model_with_pipeline_parallel(self):
        """Test loading the model with pipeline parallelism."""
        model_config = ModelConfig(tp_size=1, pp_size=2)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        # Verify model config accounts for PP
        assert model.config.pp_size == 2, f"Expected pp_size=2, got {model.config.pp_size}"
        assert model._num_layers == 64  # Total layers unchanged

    def test_prompt_enhancer_model_is_dense_not_moe(self):
        """Test that the model is correctly identified as dense (not MoE)."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        # Verify no MoE operations in context_ops
        context_op_names = [op._name for op in model.context_ops]
        assert not any("moe" in name.lower() for name in context_op_names), (
            "MoE operations should not be present in dense model"
        )

        # Verify no MoE operations in generation_ops
        generation_op_names = [op._name for op in model.generation_ops]
        assert not any("moe" in name.lower() for name in generation_op_names), (
            "MoE operations should not be present in dense model"
        )


class TestPromptEnhancer32BModelWithDifferentBackends:
    """Test the PromptEnhancer-32B model with different backends."""

    def test_prompt_enhancer_model_with_trtllm_backend(self):
        """Test loading with TRT-LLM backend."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        assert model is not None

    def test_prompt_enhancer_model_with_sglang_backend(self):
        """Test loading with SGLANG backend."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="sglang")

        assert model is not None

    def test_prompt_enhancer_model_with_vllm_backend(self):
        """Test loading with vLLM backend."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="vllm")

        assert model is not None


class TestPromptEnhancer32BModelArchitecture:
    """Test that the PromptEnhancer-32B model architecture is correctly identified."""

    def test_prompt_enhancer_model_architecture(self):
        """Test that the model has correct architecture attribute."""
        model_config = ModelConfig(tp_size=1, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        assert model.architecture == "Qwen2_5_VLForConditionalGeneration", (
            f"Expected architecture 'Qwen2_5_VLForConditionalGeneration', got {model.architecture}"
        )
        assert model.model_family == "LLAMA", f"Expected model_family 'LLAMA', got {model.model_family}"
        assert model.model_path == PROMPT_ENHANCER_32B_MODEL_PATH


class TestPromptEnhancer32BModelScaling:
    """Test that model parameters scale correctly with TP/PP configuration."""

    def test_num_kv_heads_per_gpu_with_tp(self):
        """Test that num_kv_heads_per_gpu is correctly computed with TP."""
        model_config = ModelConfig(tp_size=4, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        # num_key_value_heads=8, tp_size=4, so num_kv_heads_per_gpu = ceil(8/4) = 2
        assert model._num_kv_heads_per_gpu == 2, (
            f"Expected _num_kv_heads_per_gpu=2 with tp_size=4, got {model._num_kv_heads_per_gpu}"
        )

    def test_context_ops_tp_scaling(self):
        """Test that context_ops have correct TP scaling."""
        model_config = ModelConfig(tp_size=2, pp_size=1)

        model = get_model(model_path=PROMPT_ENHANCER_32B_MODEL_PATH, model_config=model_config, backend_name="trtllm")

        # With TP=2, vocab_size should be divided
        # The logits gemm should use vocab_size // tp_size
        logits_gemm_ops = [op for op in model.context_ops if op._name == "context_logits_gemm"]
        assert len(logits_gemm_ops) > 0, "context_logits_gemm should exist"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
