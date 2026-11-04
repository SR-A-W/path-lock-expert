"""Static weight-consistency checks for PLE.

Compares tensors one by one directly from the safetensors files, without loading
the full model into memory. Verifies the output of the weight-copy script
(copy_weights_to_ple.py).

Usage:
    cd ./
    python -m pytest src/tests/test_ple_weights_lite.py -v -s

Expected runtime: ~1-2 minutes (tensors are streamed and compared one at a time, so memory use is very low)
"""

import json
import os
import sys

import pytest
import torch

# Path configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "source", "Qwen2.5-7B-Instruct")
MLP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "source", "DeepSeek-R1-Distill-Qwen-7B")
PLE_PATH = os.path.join(PROJECT_ROOT, "models", "ple_initialized")

# Model parameters
NUM_LAYERS = 28
MLP_PROJS = ["gate_proj", "up_proj", "down_proj"]


# ==================== Helper functions ====================

def _load_tensor(model_path: str, key: str) -> torch.Tensor:
    """Load a single tensor from safetensors on demand (memory friendly).

    Reads index.json to find the shard file, then opens that shard with safe_open.

    Args:
        model_path: Path to the model directory
        key: Weight key name

    Returns:
        The weight tensor on CPU
    """
    from safetensors import safe_open

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        weight_map = json.load(f)["weight_map"]

    shard_file = weight_map[key]
    shard_path = os.path.join(model_path, shard_file)

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _get_all_keys(model_path: str) -> list:
    """Return all weight key names of the model."""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        return sorted(json.load(f)["weight_map"].keys())


# ==================== Preconditions ====================

@pytest.fixture(scope="module", autouse=True)
def check_models_exist():
    """Make sure all required model directories exist."""
    for path, name in [
        (MAIN_MODEL_PATH, "Qwen2.5-7B-Instruct"),
        (MLP_MODEL_PATH, "DeepSeek-R1-Distill-Qwen-7B"),
        (PLE_PATH, "PLE initialized"),
    ]:
        index = os.path.join(path, "model.safetensors.index.json")
        assert os.path.exists(index), (
            f"Model {name} does not exist or is not fully initialized: {path}\n"
            f"Run first: python -m src.utils.copy_weights_to_ple ..."
        )


# ==================== Expert 0 MLP weights == Qwen2.5-7B MLP ====================

class TestExpert0MLP:
    """Expert 0 (no_think) MLP weights are identical to Qwen2.5-7B-Instruct."""

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    @pytest.mark.parametrize("proj", MLP_PROJS)
    def test_expert0_mlp_weight(self, layer_idx, proj):
        """Compare Expert 0 MLP weights with the Qwen2.5-7B MLP weights, per layer and per projection."""
        src_key = f"model.layers.{layer_idx}.mlp.{proj}.weight"
        tgt_key = f"model.layers.{layer_idx}.mlp.experts.0.{proj}.weight"

        src_tensor = _load_tensor(MAIN_MODEL_PATH, src_key)
        tgt_tensor = _load_tensor(PLE_PATH, tgt_key)

        assert src_tensor.shape == tgt_tensor.shape, (
            f"Shape mismatch: {src_key} {src_tensor.shape} vs {tgt_key} {tgt_tensor.shape}"
        )
        assert torch.equal(src_tensor, tgt_tensor), (
            f"Weight mismatch: layer {layer_idx} {proj}\n"
            f"  max diff = {(src_tensor - tgt_tensor).abs().max().item()}"
        )


# ==================== Expert 1 MLP weights == DeepSeek-R1 MLP ====================

class TestExpert1MLP:
    """Expert 1 (think) MLP weights are identical to DeepSeek-R1-Distill-Qwen-7B."""

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    @pytest.mark.parametrize("proj", MLP_PROJS)
    def test_expert1_mlp_weight(self, layer_idx, proj):
        """Compare Expert 1 MLP weights with the DeepSeek-R1 MLP weights, per layer and per projection."""
        src_key = f"model.layers.{layer_idx}.mlp.{proj}.weight"
        tgt_key = f"model.layers.{layer_idx}.mlp.experts.1.{proj}.weight"

        src_tensor = _load_tensor(MLP_MODEL_PATH, src_key)
        tgt_tensor = _load_tensor(PLE_PATH, tgt_key)

        assert src_tensor.shape == tgt_tensor.shape, (
            f"Shape mismatch: {src_key} {src_tensor.shape} vs {tgt_key} {tgt_tensor.shape}"
        )
        assert torch.equal(src_tensor, tgt_tensor), (
            f"Weight mismatch: layer {layer_idx} {proj}\n"
            f"  max diff = {(src_tensor - tgt_tensor).abs().max().item()}"
        )


# ==================== Non-MLP weights == Qwen2.5-7B ====================

class TestSharedWeights:
    """Non-MLP weights (embedding, attention, layernorm, lm_head) are identical to Qwen2.5-7B."""

    @pytest.fixture(scope="class")
    def shared_keys(self):
        """Return the key names of all shared (non-MLP) weights.

        Shared weights are the PLE weights whose key does not contain '.experts.';
        they should have identical key names in the source model and in PLE.
        """
        all_keys = _get_all_keys(PLE_PATH)
        return [k for k in all_keys if ".experts." not in k]

    def test_shared_weight_count(self, shared_keys):
        """The number of shared weights equals the Qwen2.5-7B total minus the MLP weights.

        Qwen2.5-7B: 339 weights
        of which 3 MLP weights per layer x 28 layers = 84 MLP weights
        shared weights = 339 - 84 = 255
        """
        expected = 339 - NUM_LAYERS * len(MLP_PROJS)  # 339 - 84 = 255
        assert len(shared_keys) == expected, (
            f"Shared weight count: {len(shared_keys)}, expected: {expected}"
        )

    def test_embed_tokens(self):
        """embed_tokens weight consistency."""
        key = "model.embed_tokens.weight"
        src = _load_tensor(MAIN_MODEL_PATH, key)
        tgt = _load_tensor(PLE_PATH, key)
        assert torch.equal(src, tgt), f"{key} mismatch"

    def test_final_norm(self):
        """Final LayerNorm weight consistency."""
        key = "model.norm.weight"
        src = _load_tensor(MAIN_MODEL_PATH, key)
        tgt = _load_tensor(PLE_PATH, key)
        assert torch.equal(src, tgt), f"{key} mismatch"

    def test_lm_head(self):
        """lm_head weight consistency."""
        key = "lm_head.weight"
        src = _load_tensor(MAIN_MODEL_PATH, key)
        tgt = _load_tensor(PLE_PATH, key)
        assert torch.equal(src, tgt), f"{key} mismatch"

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    def test_attention_weights(self, layer_idx):
        """Per-layer attention weight consistency (weight + bias of q, k, v, o)."""
        attn_keys = [
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.q_proj.bias",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.bias",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.bias",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        ]
        for key in attn_keys:
            src = _load_tensor(MAIN_MODEL_PATH, key)
            tgt = _load_tensor(PLE_PATH, key)
            assert torch.equal(src, tgt), (
                f"{key} mismatch, max diff = {(src - tgt).abs().max().item()}"
            )

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    def test_layernorm_weights(self, layer_idx):
        """Per-layer LayerNorm weight consistency."""
        norm_keys = [
            f"model.layers.{layer_idx}.input_layernorm.weight",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        ]
        for key in norm_keys:
            src = _load_tensor(MAIN_MODEL_PATH, key)
            tgt = _load_tensor(PLE_PATH, key)
            assert torch.equal(src, tgt), f"{key} mismatch"


# ==================== No router/gate weights ====================

class TestNoGateWeights:
    """The PLE model has no gate/router weights; routing is decided purely by control tokens."""

    def test_no_gate_keys(self):
        """The weight index must not contain any gate weights (gate_proj excluded: it is part of the MLP)."""
        all_keys = _get_all_keys(PLE_PATH)
        gate_keys = [k for k in all_keys if "gate.weight" in k and "gate_proj" not in k]
        assert len(gate_keys) == 0, (
            f"Unexpected gate/router weights found: {gate_keys}\n"
            f"PLE should have no gate weights; routing is decided by control tokens"
        )

    def test_no_router_keys(self):
        """The weight index must not contain router-related key names."""
        all_keys = _get_all_keys(PLE_PATH)
        router_keys = [k for k in all_keys if "router" in k.lower()]
        assert len(router_keys) == 0, f"Unexpected router weights found: {router_keys}"

    def test_total_weight_count(self):
        """Total weight count: 255 shared + 84 Expert 0 + 84 Expert 1 = 423."""
        all_keys = _get_all_keys(PLE_PATH)
        assert len(all_keys) == 423, f"Total weight count: {len(all_keys)}, expected: 423"


# ==================== Config file validation ====================

class TestConfig:
    """The generated config.json is correct."""

    @pytest.fixture(scope="class")
    def config(self):
        with open(os.path.join(PLE_PATH, "config.json"), "r") as f:
            return json.load(f)

    def test_model_type(self, config):
        assert config["model_type"] == "qwen2_ple"

    def test_architectures(self, config):
        assert config["architectures"] == ["Qwen2PleForCausalLM"]

    def test_hidden_size(self, config):
        assert config["hidden_size"] == 3584

    def test_intermediate_size(self, config):
        assert config["intermediate_size"] == 18944

    def test_num_layers(self, config):
        assert config["num_hidden_layers"] == 28

    def test_default_routing_index(self, config):
        assert config["default_routing_index"] == 1

    def test_auto_map(self, config):
        assert "auto_map" in config
        assert "AutoConfig" in config["auto_map"]
        assert "AutoModelForCausalLM" in config["auto_map"]

    def test_no_moe_gate_params(self, config):
        """The PLE config must not contain legacy learnable-router (gate) parameters."""
        for key in ["num_experts", "num_experts_per_tok", "decoder_sparse_step",
                     "router_aux_loss_coef", "norm_topk_prob"]:
            assert key not in config, f"PLE config must not contain: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
