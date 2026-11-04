#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weight-consistency tests for LLaMA PLE (requires GPU).

Covers:
1. Static weight checks (shard count, expert symmetry, parameter-name completeness)
2. 4-bit quantized loading + routing tests
3. Output comparison between PLE (expert 0 / default routing) and the original LLaMA 3.1-8B-Instruct

Usage:
    cd .
    python -m pytest src/tests/test_llama_ple_weights.py -v -s

Note: requires a GPU and the following model files:
    - models/source/Llama-3.1-8B-Instruct
    - models/llama_ple_initialized
"""

import json
import os
import sys

import pytest
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# Model paths
SOURCE_MODEL_PATH = os.path.join(project_root, "models", "source", "Llama-3.1-8B-Instruct")
PLE_MODEL_PATH = os.path.join(project_root, "models", "llama_ple_initialized")

# Skip conditions
skip_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)
skip_no_source = pytest.mark.skipif(
    not os.path.exists(SOURCE_MODEL_PATH), reason=f"Source model not found: {SOURCE_MODEL_PATH}"
)
skip_no_ple = pytest.mark.skipif(
    not os.path.exists(PLE_MODEL_PATH), reason=f"PLE model not found: {PLE_MODEL_PATH}"
)


# ==================== Static weight checks (no GPU needed) ====================

class TestStaticWeights:
    """Static weight checks: no model is loaded, only the safetensors index is inspected."""

    @skip_no_ple
    def test_weight_count(self):
        """Total weight count = 32 layers x 12 per layer + 3 global = 387."""
        index_path = os.path.join(PLE_MODEL_PATH, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_keys = list(index["weight_map"].keys())
        assert len(weight_keys) == 387, f"Expected 387, got {len(weight_keys)}"

    @skip_no_ple
    def test_expert_symmetry(self):
        """Expert 0 and Expert 1 weight names should correspond one-to-one."""
        index_path = os.path.join(PLE_MODEL_PATH, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_keys = list(index["weight_map"].keys())

        e0_keys = sorted([k for k in weight_keys if ".experts.0." in k])
        e1_keys = sorted([k for k in weight_keys if ".experts.1." in k])

        assert len(e0_keys) == 96, f"Expert 0: expected 96, got {len(e0_keys)}"
        assert len(e1_keys) == 96, f"Expert 1: expected 96, got {len(e1_keys)}"

        # Names should correspond one-to-one (differing only in .0. vs .1.)
        for k0, k1 in zip(e0_keys, e1_keys):
            assert k0.replace(".experts.0.", ".experts.1.") == k1

    @skip_no_ple
    def test_no_gate_weights(self):
        """PLE must not have gate/router weights."""
        index_path = os.path.join(PLE_MODEL_PATH, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        gate_keys = [k for k in index["weight_map"] if "gate.weight" in k and "gate_proj" not in k]
        assert len(gate_keys) == 0, f"Unexpected gate weights found: {gate_keys}"

    @skip_no_ple
    def test_config_fields(self):
        """config.json contains the required PLE fields."""
        config_path = os.path.join(PLE_MODEL_PATH, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        assert config["model_type"] == "llama_ple"
        assert config["think_token_id"] == 128256
        assert config["no_think_token_id"] == 128257
        assert config["default_routing_index"] == 0
        assert config["vocab_size"] == 128258

    @skip_no_ple
    def test_embedding_shapes(self):
        """Embedding and lm_head have been resized to 128258."""
        from safetensors import safe_open

        index_path = os.path.join(PLE_MODEL_PATH, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)

        for key in ["model.embed_tokens.weight", "lm_head.weight"]:
            shard = index["weight_map"][key]
            shard_path = os.path.join(PLE_MODEL_PATH, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                t = f.get_tensor(key)
                assert t.shape[0] == 128258, f"{key}: expected [128258, ...], got {list(t.shape)}"
                assert t.shape[1] == 4096


# ==================== GPU loading and routing tests ====================

@skip_no_gpu
@skip_no_ple
class TestGPURouting:
    """4-bit quantized loading + routing tests."""

    @pytest.fixture(scope="class")
    def ple_model(self):
        """Load the 4-bit quantized PLE model (cached at class scope)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"\nLoading PLE model (4-bit): {PLE_MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            PLE_MODEL_PATH,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(PLE_MODEL_PATH, trust_remote_code=True)
        model.eval()
        return model, tokenizer

    def test_model_loads(self, ple_model):
        """The model loads correctly."""
        model, tokenizer = ple_model
        assert model is not None
        assert len(tokenizer) == 128258

    def test_generate_expert_0(self, ple_model):
        """Expert 0 (no_think) can generate."""
        model, tokenizer = ple_model
        prompt = "/no_think What is 2+2?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False, routing_index=0)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  Expert 0 output: {text[:100]}")
        assert len(output[0]) > len(inputs["input_ids"][0])

    def test_generate_expert_1(self, ple_model):
        """Expert 1 (think) can generate."""
        model, tokenizer = ple_model
        prompt = "/think What is 2+2?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False, routing_index=1)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  Expert 1 output: {text[:100]}")
        assert len(output[0]) > len(inputs["input_ids"][0])

    def test_routing_cached(self, ple_model):
        """routing_index is cached correctly after generate."""
        model, tokenizer = ple_model
        inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5, do_sample=False, routing_index=0)
        assert model._cached_routing_index == 0

        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5, do_sample=False, routing_index=1)
        assert model._cached_routing_index == 1

    def test_auto_routing_from_control_token(self, ple_model):
        """Control tokens are detected in the input and routed automatically."""
        model, tokenizer = ple_model

        # /think → expert 1
        inputs_think = tokenizer("/think Hello", return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs_think, max_new_tokens=5, do_sample=False)
        assert model._cached_routing_index == 1

        # /no_think → expert 0
        inputs_no_think = tokenizer("/no_think Hello", return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs_no_think, max_new_tokens=5, do_sample=False)
        assert model._cached_routing_index == 0


# ==================== Output comparison with the original model ====================

@skip_no_gpu
@skip_no_source
@skip_no_ple
class TestOutputConsistency:
    """PLE (expert 0) output should match the original LLaMA 3.1."""

    @pytest.fixture(scope="class")
    def both_models(self):
        """Load the original model and PLE together (4-bit, sharing the GPU)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print(f"\nLoading original model (4-bit): {SOURCE_MODEL_PATH}")
        source_model = AutoModelForCausalLM.from_pretrained(
            SOURCE_MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
        )
        source_tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_PATH)
        source_model.eval()

        print(f"Loading PLE model (4-bit): {PLE_MODEL_PATH}")
        ple_model = AutoModelForCausalLM.from_pretrained(
            PLE_MODEL_PATH,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        ple_tokenizer = AutoTokenizer.from_pretrained(PLE_MODEL_PATH, trust_remote_code=True)
        ple_model.eval()

        return source_model, source_tokenizer, ple_model, ple_tokenizer

    def test_greedy_output_matches(self, both_models):
        """Expert 0 greedy-decode output should match the original model token by token."""
        source_model, source_tok, ple_model, ple_tok = both_models

        # Use a prompt without control tokens (default routing = expert 0)
        prompt = "The capital of France is"
        source_inputs = source_tok(prompt, return_tensors="pt").to(source_model.device)
        ple_inputs = ple_tok(prompt, return_tensors="pt").to(ple_model.device)

        max_new = 30
        with torch.no_grad():
            source_out = source_model.generate(
                **source_inputs, max_new_tokens=max_new, do_sample=False
            )
            ple_out = ple_model.generate(
                **ple_inputs, max_new_tokens=max_new, do_sample=False, routing_index=0
            )

        source_text = source_tok.decode(source_out[0], skip_special_tokens=True)
        ple_text = ple_tok.decode(ple_out[0], skip_special_tokens=True)

        print(f"\n  Source:  {source_text[:120]}")
        print(f"  PLE:  {ple_text[:120]}")

        # Token-by-token comparison
        source_ids = source_out[0].tolist()
        ple_ids = ple_out[0].tolist()

        # Compare the newly generated tokens (skip the prompt part)
        prompt_len = source_inputs["input_ids"].shape[1]
        source_gen = source_ids[prompt_len:]
        ple_gen = ple_ids[prompt_len:]

        match_count = sum(1 for a, b in zip(source_gen, ple_gen) if a == b)
        total = min(len(source_gen), len(ple_gen))
        match_rate = match_count / total if total > 0 else 0

        print(f"  Token match: {match_count}/{total} ({match_rate:.1%})")
        assert match_rate == 1.0, (
            f"PLE expert 0 output differs from the original model ({match_rate:.1%} match).\n"
            f"Source tokens: {source_gen[:10]}\n"
            f"PLE tokens: {ple_gen[:10]}"
        )

    def test_logits_close(self, both_models):
        """Single-step forward logits should be very close."""
        source_model, source_tok, ple_model, ple_tok = both_models

        prompt = "Hello world"
        source_inputs = source_tok(prompt, return_tensors="pt").to(source_model.device)
        ple_inputs = ple_tok(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            source_logits = source_model(**source_inputs).logits
            ple_logits = ple_model(**ple_inputs, routing_index=0).logits

        # Compare only logits within the original vocab (first 128256)
        source_logits_trimmed = source_logits[:, :, :128256]
        ple_logits_trimmed = ple_logits[:, :, :128256]

        # Check whether the argmax agrees
        source_argmax = source_logits_trimmed.argmax(dim=-1)
        ple_argmax = ple_logits_trimmed.argmax(dim=-1)
        argmax_match = (source_argmax == ple_argmax).float().mean().item()
        print(f"\n  Argmax match rate: {argmax_match:.1%}")
        assert argmax_match == 1.0, f"Argmax mismatch: {argmax_match:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
