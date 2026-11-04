"""Basic PLE model tests: config, construction, forward pass, routing logic, routing cache in generate.

Usage:
    cd ./
    python -m pytest src/tests/test_ple_basic.py -v -s
"""

import sys
import os
import torch
import pytest

# Put the transformers submodule's src on sys.path so it takes precedence over the pip-installed transformers
TRANSFORMERS_SRC = os.path.join(
    os.path.dirname(__file__), "..", "transformers", "src"
)
sys.path.insert(0, os.path.abspath(TRANSFORMERS_SRC))

from transformers.models.qwen2_ple.configuration_qwen2_ple import Qwen2PleConfig
from transformers.models.qwen2_ple.modeling_qwen2_ple import (
    Qwen2PleExpertMLP,
    Qwen2PleMLP,
    Qwen2PleDecoderLayer,
    Qwen2PleModel,
    Qwen2PleForCausalLM,
    _determine_routing_index,
)


# ---------- Small test config (no large GPU memory needed) ----------
THINK_ID = 42
NO_THINK_ID = 43

def _tiny_config(think_token_id=THINK_ID, no_think_token_id=NO_THINK_ID, default_routing_index=1):
    """Build a tiny PLE config for fast CPU tests."""
    return Qwen2PleConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        think_token_id=think_token_id,
        no_think_token_id=no_think_token_id,
        default_routing_index=default_routing_index,
    )


# ===================== Config tests =====================
class TestConfig:
    def test_model_type(self):
        """model_type is qwen2_ple."""
        config = _tiny_config()
        assert config.model_type == "qwen2_ple"

    def test_think_token_id(self):
        config = _tiny_config(think_token_id=42)
        assert config.think_token_id == 42

    def test_no_think_token_id(self):
        config = _tiny_config(no_think_token_id=43)
        assert config.no_think_token_id == 43

    def test_default_routing_index(self):
        """The default routing is 1 (think mode)."""
        config = _tiny_config()
        assert config.default_routing_index == 1

    def test_default_routing_index_custom(self):
        config = _tiny_config(default_routing_index=0)
        assert config.default_routing_index == 0


# ===================== Routing-function tests =====================
class TestDetermineRouting:
    def test_only_think_token(self):
        """Only a think token -> routing_index = 1."""
        config = _tiny_config()
        ids = torch.tensor([[1, 2, THINK_ID, 3, 4]])
        assert _determine_routing_index(ids, config) == 1

    def test_only_no_think_token(self):
        """Only a no_think token -> routing_index = 0."""
        config = _tiny_config()
        ids = torch.tensor([[1, 2, NO_THINK_ID, 3, 4]])
        assert _determine_routing_index(ids, config) == 0

    def test_no_control_token(self):
        """No control token -> use default_routing_index (default 1)."""
        config = _tiny_config()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        assert _determine_routing_index(ids, config) == 1  # default=1 (think)

    def test_no_control_token_custom_default(self):
        """Custom default=0 and no control token -> returns 0."""
        config = _tiny_config(default_routing_index=0)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        assert _determine_routing_index(ids, config) == 0

    def test_both_tokens_think_last(self):
        """Both tokens present, think last -> routing_index = 1."""
        config = _tiny_config()
        ids = torch.tensor([[1, NO_THINK_ID, 2, THINK_ID, 3]])
        assert _determine_routing_index(ids, config) == 1

    def test_both_tokens_no_think_last(self):
        """Both tokens present, no_think last -> routing_index = 0 (injection defense)."""
        config = _tiny_config()
        ids = torch.tensor([[1, THINK_ID, 2, NO_THINK_ID, 3]])
        assert _determine_routing_index(ids, config) == 0

    def test_injection_defense(self):
        """Injection defense: the user inserted a think token early, but the system placed a no_think token later."""
        config = _tiny_config()
        # Sequence: [bos, user_think_injection, ..., system_no_think_control]
        ids = torch.tensor([[1, THINK_ID, 10, 11, 12, NO_THINK_ID]])
        assert _determine_routing_index(ids, config) == 0  # the final no_think overrides the injected think

    def test_no_token_ids_configured(self):
        """No token IDs configured -> returns the default directly."""
        config = _tiny_config(think_token_id=None, no_think_token_id=None, default_routing_index=1)
        ids = torch.tensor([[1, 2, 3]])
        assert _determine_routing_index(ids, config) == 1

    def test_only_think_id_configured_not_found(self):
        """Only think_token_id configured and absent from input_ids -> default."""
        config = _tiny_config(think_token_id=THINK_ID, no_think_token_id=None, default_routing_index=1)
        ids = torch.tensor([[1, 2, 3]])
        assert _determine_routing_index(ids, config) == 1  # not found -> default

    def test_only_think_id_configured_found(self):
        """Only think_token_id configured and present -> 1."""
        config = _tiny_config(think_token_id=THINK_ID, no_think_token_id=None)
        ids = torch.tensor([[1, THINK_ID, 3]])
        assert _determine_routing_index(ids, config) == 1


# ===================== MLP tests =====================
class TestMLP:
    def test_expert_mlp_output_shape(self):
        """A single expert MLP produces the correct output shape."""
        config = _tiny_config()
        mlp = Qwen2PleExpertMLP(config)
        x = torch.randn(2, 8, 64)
        out = mlp(x)
        assert out.shape == (2, 8, 64)

    def test_ple_mlp_routing(self):
        """The two experts in the PLE MLP container have different weights, so routing to different experts gives different outputs."""
        config = _tiny_config()
        mlp = Qwen2PleMLP(config)
        x = torch.randn(2, 8, 64)
        out0 = mlp(x, routing_index=0)
        out1 = mlp(x, routing_index=1)
        assert out0.shape == out1.shape == (2, 8, 64)
        assert not torch.allclose(out0, out1), "The two experts' outputs should not be identical"

    def test_ple_mlp_has_two_experts(self):
        config = _tiny_config()
        mlp = Qwen2PleMLP(config)
        assert len(mlp.experts) == 2


# ===================== Decoder layer tests =====================
class TestDecoderLayer:
    def _make_layer(self):
        config = _tiny_config()
        config._attn_implementation = "eager"
        return config, Qwen2PleDecoderLayer(config, layer_idx=0)

    def test_forward_shape(self):
        config, layer = self._make_layer()
        x = torch.randn(1, 8, 64)
        cos = torch.ones(1, 8, 16)
        sin = torch.zeros(1, 8, 16)
        out = layer(x, position_embeddings=(cos, sin), routing_index=0)
        assert out.shape == (1, 8, 64)

    def test_routing_index_passed(self):
        config, layer = self._make_layer()
        x = torch.randn(1, 8, 64)
        cos = torch.ones(1, 8, 16)
        sin = torch.zeros(1, 8, 16)
        out0 = layer(x, position_embeddings=(cos, sin), routing_index=0)
        out1 = layer(x, position_embeddings=(cos, sin), routing_index=1)
        assert not torch.allclose(out0, out1), "Routing to different experts should produce different outputs"


# ===================== Model tests =====================
class TestModel:
    def test_model_creation(self):
        config = _tiny_config()
        model = Qwen2PleModel(config)
        assert len(model.layers) == 2
        for layer in model.layers:
            assert isinstance(layer.mlp, Qwen2PleMLP)

    def test_forward_with_think(self):
        """Contains a think token -> routing_index=1."""
        config = _tiny_config()
        model = Qwen2PleModel(config).eval()
        input_ids = torch.tensor([[1, THINK_ID, 3, 4, 5]])
        output = model(input_ids=input_ids)
        assert output.last_hidden_state.shape == (1, 5, 64)

    def test_forward_with_no_think(self):
        """Contains a no_think token -> routing_index=0."""
        config = _tiny_config()
        model = Qwen2PleModel(config).eval()
        input_ids = torch.tensor([[1, NO_THINK_ID, 3, 4, 5]])
        output = model(input_ids=input_ids)
        assert output.last_hidden_state.shape == (1, 5, 64)

    def test_forward_no_control_token_defaults_to_think(self):
        """No control token -> defaults to think (expert 1); verified by passing routing_index explicitly."""
        config = _tiny_config()
        model = Qwen2PleModel(config).eval()
        # Same input_ids; only the way routing is chosen changes
        ids = torch.tensor([[1, 2, 3, 4, 5]])  # contains no control token
        with torch.no_grad():
            # No routing_index -> default_routing_index=1 (think) should be used automatically
            out_default = model(input_ids=ids).last_hidden_state
            # Explicit routing_index=1 -> think selected manually
            out_explicit_think = model(input_ids=ids, routing_index=1).last_hidden_state
        # The two must be identical: default routing == explicit think
        assert torch.allclose(out_default, out_explicit_think), "default should match explicit routing_index=1"

    def test_routing_differs(self):
        """think and no_think routing give different outputs."""
        config = _tiny_config()
        model = Qwen2PleModel(config).eval()
        ids_think = torch.tensor([[1, THINK_ID, 3, 4, 5]])
        ids_no_think = torch.tensor([[1, NO_THINK_ID, 3, 4, 5]])
        with torch.no_grad():
            out_think = model(input_ids=ids_think).last_hidden_state
            out_no_think = model(input_ids=ids_no_think).last_hidden_state
        assert not torch.allclose(out_think, out_no_think), "think and no_think outputs should differ"

    def test_explicit_routing_index_override(self):
        """An explicitly passed routing_index overrides automatic detection."""
        config = _tiny_config()
        model = Qwen2PleModel(config).eval()
        # Same input_ids (containing a think token); only the routing_index argument differs
        ids = torch.tensor([[1, THINK_ID, 3, 4, 5]])
        with torch.no_grad():
            # Auto-detection -> routing_index=1 (because of the think token)
            out_auto = model(input_ids=ids).last_hidden_state
            # Explicit routing_index=0 -> should override auto-detection and use no_think
            out_override = model(input_ids=ids, routing_index=0).last_hidden_state
            # Explicit routing_index=1 -> explicit think
            out_explicit = model(input_ids=ids, routing_index=1).last_hidden_state
        # auto == explicit think
        assert torch.allclose(out_auto, out_explicit), "Auto-detection should match explicit routing_index=1"
        # override != auto (different experts)
        assert not torch.allclose(out_override, out_auto), "Explicit routing_index=0 should differ from auto think"


# ===================== ForCausalLM tests =====================
class TestForCausalLM:
    def test_creation(self):
        config = _tiny_config()
        model = Qwen2PleForCausalLM(config)
        assert model is not None
        assert model._cached_routing_index is None  # initial state

    def test_forward_logits(self):
        config = _tiny_config()
        model = Qwen2PleForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (1, 5, 256)

    def test_forward_with_labels(self):
        config = _tiny_config()
        model = Qwen2PleForCausalLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[2, 3, 4, 5, 0]])
        output = model(input_ids=input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.ndim == 0

    def test_generate(self):
        """generate runs correctly."""
        config = _tiny_config()
        config.pad_token_id = 0
        config.eos_token_id = 255
        model = Qwen2PleForCausalLM(config).eval()
        input_ids = torch.tensor([[1, THINK_ID, 3]])
        with torch.no_grad():
            gen_ids = model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False)
        assert gen_ids.shape[0] == 1
        assert gen_ids.shape[1] >= 3

    def test_generate_caches_routing(self):
        """routing_index is cached during generate and not lost on later tokens."""
        config = _tiny_config()
        config.pad_token_id = 0
        config.eos_token_id = 255
        model = Qwen2PleForCausalLM(config).eval()

        # generate with an input containing a think token
        input_ids_think = torch.tensor([[1, THINK_ID, 3]])
        with torch.no_grad():
            model.generate(input_ids=input_ids_think, max_new_tokens=3, do_sample=False)
        assert model._cached_routing_index == 1, "routing_index=1 (think) should be cached after generation"

        # generate with an input containing a no_think token
        input_ids_no_think = torch.tensor([[1, NO_THINK_ID, 3]])
        with torch.no_grad():
            model.generate(input_ids=input_ids_no_think, max_new_tokens=3, do_sample=False)
        assert model._cached_routing_index == 0, "routing_index=0 (no_think) should be cached after generation"

    def test_generate_different_routes_differ(self):
        """generate outputs for think and no_think routing should differ."""
        config = _tiny_config()
        config.pad_token_id = 0
        config.eos_token_id = 255
        model = Qwen2PleForCausalLM(config).eval()

        input_ids_think = torch.tensor([[1, THINK_ID, 3]])
        input_ids_no_think = torch.tensor([[1, NO_THINK_ID, 3]])
        with torch.no_grad():
            gen_think = model.generate(input_ids=input_ids_think, max_new_tokens=5, do_sample=False)
            gen_no_think = model.generate(input_ids=input_ids_no_think, max_new_tokens=5, do_sample=False)
        # The two routes should not generate identical results (the experts have different weights)
        assert not torch.equal(gen_think, gen_no_think), "generate outputs for think and no_think should differ"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
