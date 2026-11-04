"""PLE routing and forward-pass behaviour tests (4-bit quantized).

Loads the PLE model (4-bit quantized) and checks the routing logic and that the
generated output is sensible. The reference model is also loaded in 4-bit so the
comparison is fair.

Test strategy:
- All models use NF4 4-bit quantization to fit in GPU memory
- Expert 0 (no_think, Qwen2.5-7B MLP) should match the original Qwen2.5-7B output
- Expert 1 (think, DeepSeek-R1 MLP) producing unreadable output before fine-tuning is expected
- The focus is routing correctness: whether routing_index is passed through correctly in forward/generate

Usage:
    cd ./
    python -m pytest src/tests/test_ple_weights.py -v -s

Note: requires bitsandbytes and a CUDA GPU.
"""

import gc
import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

# Path configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "source", "Qwen2.5-7B-Instruct")
MLP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "source", "DeepSeek-R1-Distill-Qwen-7B")
PLE_PATH = os.path.join(PROJECT_ROOT, "models", "ple_initialized")


# ==================== Helper functions ====================

def _load_tensor(model_path: str, key: str) -> torch.Tensor:
    """Load a single tensor from safetensors on demand (without loading the full model)."""
    from safetensors import safe_open

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        weight_map = json.load(f)["weight_map"]

    shard_file = weight_map[key]
    shard_path = os.path.join(model_path, shard_file)

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _load_model_4bit(model_path, trust_remote_code=False):
    """Load a model onto the GPU with 4-bit NF4 quantization."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


# ==================== Preconditions ====================

@pytest.fixture(scope="module", autouse=True)
def check_models_exist():
    """Make sure all required model directories exist and CUDA is available."""
    for path, name in [
        (MAIN_MODEL_PATH, "Qwen2.5-7B-Instruct"),
        (PLE_PATH, "PLE initialized"),
    ]:
        assert os.path.exists(os.path.join(path, "config.json")), (
            f"Model {name} not found: {path}"
        )
    assert torch.cuda.is_available(), "This test requires a CUDA GPU"


# ==================== Model fixtures ====================

@pytest.fixture(scope="module")
def ple_model():
    """Load the PLE model (4-bit quantized)."""
    print("\n[Loading PLE model 4-bit...]")
    model = _load_model_4bit(PLE_PATH, trust_remote_code=True)
    print(f"  PLE loaded")
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MAIN_MODEL_PATH)


# ==================== Forward routing tests ====================

class TestForwardRouting:
    """routing_index is passed through correctly in forward, and the two paths produce different logits."""

    def test_routing_index_0_valid(self, ple_model):
        """routing_index=0 produces logits normally."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(ple_model.device)
        with torch.no_grad():
            out = ple_model(input_ids=input_ids, routing_index=0)
        assert out.logits is not None
        assert out.logits.shape[-1] == 152064

    def test_routing_index_1_valid(self, ple_model):
        """routing_index=1 produces logits normally."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(ple_model.device)
        with torch.no_grad():
            out = ple_model(input_ids=input_ids, routing_index=1)
        assert out.logits is not None
        assert out.logits.shape[-1] == 152064

    def test_different_routes_produce_different_logits(self, ple_model):
        """Logits for routing_index=0 and =1 should differ (the two experts have different MLP weights)."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(ple_model.device)
        with torch.no_grad():
            logits_r0 = ple_model(input_ids=input_ids, routing_index=0).logits.cpu()
            logits_r1 = ple_model(input_ids=input_ids, routing_index=1).logits.cpu()
        assert not torch.equal(logits_r0, logits_r1), \
            "Logits for routing_index=0 and =1 are identical: MLP routing has no effect"

    def test_default_routing_uses_config(self, ple_model):
        """Without routing_index, config.default_routing_index should be used."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(ple_model.device)
        default_idx = ple_model.config.default_routing_index

        with torch.no_grad():
            logits_default = ple_model(input_ids=input_ids).logits.cpu()
            logits_explicit = ple_model(
                input_ids=input_ids, routing_index=default_idx
            ).logits.cpu()

        assert torch.equal(logits_default, logits_explicit), (
            f"Default routing (no routing_index) should be equivalent to routing_index={default_idx}"
        )


# ==================== Generate routing tests ====================

class TestGenerateRouting:
    """routing_index is passed through correctly in generate.

    prepare_inputs_for_generation must honour a user-supplied routing_index; otherwise
    every route in generate would fall back to the default.
    """

    def test_generate_with_explicit_routing_index(self, ple_model, tokenizer):
        """Explicitly specifying the route via generate(routing_index=N) takes effect."""
        prompt = "What is 2 + 3? Answer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            gen_r0 = ple_model.generate(
                **inputs, max_new_tokens=30, do_sample=False, routing_index=0
            )
            gen_r1 = ple_model.generate(
                **inputs, max_new_tokens=30, do_sample=False, routing_index=1
            )

        text_r0 = tokenizer.decode(gen_r0[0], skip_special_tokens=True)
        text_r1 = tokenizer.decode(gen_r1[0], skip_special_tokens=True)

        print(f"\n  [generate r=0] {text_r0[:200]}")
        print(f"  [generate r=1] {text_r1[:200]}")

        assert text_r0 != text_r1, \
            "generate outputs for routing_index=0 and =1 are identical: the routing argument was not passed through"

    def test_generate_default_matches_expert0(self, ple_model, tokenizer):
        """Without routing_index, generate should use config.default_routing_index=0."""
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            gen_default = ple_model.generate(
                **inputs, max_new_tokens=30, do_sample=False,
            )
            gen_r0 = ple_model.generate(
                **inputs, max_new_tokens=30, do_sample=False, routing_index=0
            )

        text_default = tokenizer.decode(gen_default[0], skip_special_tokens=True)
        text_r0 = tokenizer.decode(gen_r0[0], skip_special_tokens=True)

        print(f"\n  [generate default] {text_default[:200]}")
        print(f"  [generate r=0]     {text_r0[:200]}")

        assert text_default == text_r0, \
            "Default generation should match routing_index=0 (config.default_routing_index=0)"

    def test_cached_routing_index_correct(self, ple_model, tokenizer):
        """After generate, _cached_routing_index reflects the route actually used."""
        prompt = "Test prompt"
        inputs = tokenizer(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            ple_model.generate(**inputs, max_new_tokens=5, do_sample=False, routing_index=1)
        assert ple_model._cached_routing_index == 1

        with torch.no_grad():
            ple_model.generate(**inputs, max_new_tokens=5, do_sample=False, routing_index=0)
        assert ple_model._cached_routing_index == 0


# ==================== Expert 0 output consistency ====================

class TestExpert0Consistency:
    """PLE (expert 0) generate output matches Qwen2.5-7B-Instruct.

    Both use the same 4-bit quantization config, so the results should be identical.
    The PLE expert 0 weights and the shared layers both come directly from Qwen2.5-7B,
    so the model with routing_index=0 should behave like the original Qwen2.5-7B.
    """

    @pytest.fixture(scope="class")
    def qwen_model(self):
        """Load Qwen2.5-7B-Instruct (4-bit quantized) for comparison."""
        print("\n[Loading Qwen2.5-7B-Instruct 4-bit...]")
        model = _load_model_4bit(MAIN_MODEL_PATH)
        print(f"  Qwen2.5-7B loaded")
        yield model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("prompt", [
        "What is 2 + 3? Answer:",
        "Hello, how are you?",
        "Explain gravity in one sentence.",
    ])
    def test_generate_matches_qwen(self, ple_model, qwen_model, tokenizer, prompt):
        """PLE expert 0 generated output should be identical to Qwen2.5-7B."""
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            gen_pl = ple_model.generate(
                input_ids=inputs["input_ids"].to(ple_model.device),
                attention_mask=inputs["attention_mask"].to(ple_model.device),
                max_new_tokens=50,
                do_sample=False,
                routing_index=0,
            )
            gen_qwen = qwen_model.generate(
                input_ids=inputs["input_ids"].to(qwen_model.device),
                attention_mask=inputs["attention_mask"].to(qwen_model.device),
                max_new_tokens=50,
                do_sample=False,
            )

        text_pl = tokenizer.decode(gen_pl[0], skip_special_tokens=True)
        text_qwen = tokenizer.decode(gen_qwen[0], skip_special_tokens=True)

        print(f"\n  Prompt: {prompt}")
        print(f"  [PLE r=0] {text_pl[:200]}")
        print(f"  [Qwen2.5-7B] {text_qwen[:200]}")

        assert text_pl == text_qwen, (
            f"PLE expert 0 output should match Qwen2.5-7B\n"
            f"  PLE: {text_pl[:100]}\n"
            f"  Qwen:   {text_qwen[:100]}"
        )

    def test_forward_logits_match(self, ple_model, qwen_model, tokenizer):
        """PLE expert 0 forward logits should be identical to Qwen2.5-7B."""
        prompt = "The answer to life is"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            logits_pl = ple_model(
                input_ids=inputs["input_ids"].to(ple_model.device),
                attention_mask=inputs["attention_mask"].to(ple_model.device),
                routing_index=0,
            ).logits.cpu().float()

            logits_qwen = qwen_model(
                input_ids=inputs["input_ids"].to(qwen_model.device),
                attention_mask=inputs["attention_mask"].to(qwen_model.device),
            ).logits.cpu().float()

        # The top-10 predictions should agree
        top_pl = logits_pl[0, -1].topk(10).indices.tolist()
        top_qwen = logits_qwen[0, -1].topk(10).indices.tolist()
        print(f"\n  PLE r=0 top10: {top_pl}")
        print(f"  Qwen2.5-7B top10: {top_qwen}")

        assert top_pl == top_qwen, (
            f"PLE expert 0 top-10 predictions should match Qwen2.5-7B\n"
            f"  PLE: {top_pl}\n"
            f"  Qwen:   {top_qwen}"
        )


# ==================== Embedding consistency ====================

class TestEmbeddingConsistency:
    """PLE embedding matches Qwen2.5-7B (shared layer, compared against safetensors)."""

    def test_embedding_output(self, ple_model):
        """Same input_ids -> same embedding output."""
        input_ids = torch.tensor([[1, 100, 500, 1000, 2000]])

        # PLE embedding
        pl_embed = ple_model.model.embed_tokens
        pl_embed_device = next(pl_embed.parameters()).device
        ids_pl = input_ids.to(pl_embed_device)
        with torch.no_grad():
            pl_out = pl_embed(ids_pl).float().cpu()

        # Reference: load the Qwen embedding weight from safetensors and compute manually
        ref_embed_weight = _load_tensor(MAIN_MODEL_PATH, "model.embed_tokens.weight")
        ref_out = F.embedding(input_ids, ref_embed_weight).float()

        assert torch.allclose(pl_out, ref_out, atol=1e-4), (
            f"Embedding output mismatch, max diff = {(pl_out - ref_out).abs().max().item()}"
        )


# ==================== Expert 1 behaviour tests ====================

class TestExpert1Behavior:
    """Behavioural properties of expert 1.

    Note: before fine-tuning, Expert 1 (DeepSeek-R1 MLP) is incompatible with the
    Qwen2.5 shared layers (attention, norm, embedding), so unreadable output is expected.
    These tests only verify that:
    1. Expert 1 does not cause errors or crashes
    2. Expert 1 and Expert 0 produce different outputs
    """

    def test_expert1_no_crash(self, ple_model, tokenizer):
        """Routing to Expert 1 can generate without raising exceptions."""
        prompt = "What is AI?"
        inputs = tokenizer(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            gen = ple_model.generate(
                **inputs, max_new_tokens=20, do_sample=False, routing_index=1
            )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"\n  [Expert 1] {text[:200]}")
        # Output quality is not checked; only that nothing crashes
        assert len(text) > 0

    def test_expert1_differs_from_expert0(self, ple_model, tokenizer):
        """Generated outputs of Expert 0 and Expert 1 should differ."""
        prompt = "2 + 2 ="
        inputs = tokenizer(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            gen_r0 = ple_model.generate(
                **inputs, max_new_tokens=20, do_sample=False, routing_index=0
            )
            gen_r1 = ple_model.generate(
                **inputs, max_new_tokens=20, do_sample=False, routing_index=1
            )

        text_r0 = tokenizer.decode(gen_r0[0], skip_special_tokens=True)
        text_r1 = tokenizer.decode(gen_r1[0], skip_special_tokens=True)

        print(f"\n  [Expert 0] {text_r0[:200]}")
        print(f"  [Expert 1] {text_r1[:200]}")

        assert text_r0 != text_r1, "Both experts produce identical output: routing has no effect"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
