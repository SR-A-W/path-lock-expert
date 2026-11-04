# coding=utf-8
# Copyright 2026 Hybrid-Expert-Thinking Project.
"""Unit tests for the Qwen3 PLE routing-detection rules.

Covers:
- Two scan paths: Branch 1 (anchor present) and Branch 2 (no anchor, bare prompt)
- Last-occurrence-wins semantics
- Regex negative lookbehind excludes the closing `</think>` tag
- Regex negative lookahead excludes substrings such as `/think_hard`
- Strict mode: on NO_MATCH, strict raises vs lenient falls back to the default
- KV cache: generate smoke test, run when a GPU is available
- Config backward compatibility: a missing `strict_routing_match` defaults to True via getattr
- All imports go through `src.models.qwen3_ple`

Usage: python -m pytest src/tests/test_qwen3_ple_routing.py -v
"""

import os
import sys

import pytest
import torch

# Make sure src.models.qwen3_ple is imported from the repository root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoTokenizer  # noqa: E402

from src.models.qwen3_ple.configuration_qwen3_ple import Qwen3PLEConfig  # noqa: E402
from src.models.qwen3_ple.modeling_qwen3_ple import (  # noqa: E402
    _ANCHOR_BIGRAM,
    _detect_routing_for_row,
    _determine_routing_index,
    _find_last_bigram,
)


CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "models/qwen3_4b_ple_initialized")


# ---------------------- fixtures ----------------------

@pytest.fixture(scope="module")
def tokenizer():
    """Tokenizer of the checkpoint (Qwen3 native BPE, no registered control tokens)."""
    return AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)


@pytest.fixture
def strict_config():
    """Strict mode on (default): raise on NO_MATCH."""
    cfg = Qwen3PLEConfig(strict_routing_match=True, default_routing_index=0)
    cfg._name_or_path = CHECKPOINT_DIR  # needed by _get_routing_tokenizer
    return cfg


@pytest.fixture
def lenient_config():
    """Strict mode off: on NO_MATCH, fall back to the default with a warning."""
    cfg = Qwen3PLEConfig(strict_routing_match=False, default_routing_index=0)
    cfg._name_or_path = CHECKPOINT_DIR
    return cfg


def _ids_batch(tokenizer, text: str) -> torch.LongTensor:
    """Encode a single text into a [1, S] LongTensor."""
    return torch.tensor([tokenizer.encode(text, add_special_tokens=False)], dtype=torch.long)


# ---------------------- Basic helper ----------------------

class TestFindLastBigram:
    """Unit tests for `_find_last_bigram`."""

    def test_finds_last_occurrence(self):
        # The sequence has three (1,2) bigrams; the last one must be returned
        ids = torch.tensor([1, 2, 3, 1, 2, 4, 1, 2, 5], dtype=torch.long)
        assert _find_last_bigram(ids, (1, 2)) == 6

    def test_not_found(self):
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        assert _find_last_bigram(ids, (5, 6)) is None

    def test_too_short(self):
        ids = torch.tensor([1], dtype=torch.long)
        assert _find_last_bigram(ids, (1, 2)) is None

    def test_anchor_bigram_constant(self):
        # The anchor is (151644=<|im_start|>, 77091=assistant)
        assert _ANCHOR_BIGRAM == (151644, 77091)


# ---------------------- Routing detection tests ----------------------

class TestRoutingBranch1WithAnchor:
    """Branch 1: a chat template is applied and the anchor is present."""

    def test_think_prompt(self, tokenizer, strict_config):
        prompt = "<|im_start|>user\nWhat is 2+2? /think<|im_end|>\n<|im_start|>assistant\n"
        result = _determine_routing_index(_ids_batch(tokenizer, prompt), strict_config, tokenizer)
        assert result.shape == (1,)
        assert result.dtype == torch.long
        assert int(result[0]) == 1, f"/think should route to 1, got {int(result[0])}"

    def test_no_think_prompt(self, tokenizer, strict_config):
        prompt = "<|im_start|>user\nWhat is 2+2? /no_think<|im_end|>\n<|im_start|>assistant\n"
        assert int(_determine_routing_index(_ids_batch(tokenizer, prompt), strict_config, tokenizer)[0]) == 0

    def test_no_think_with_close_tag_in_response(self, tokenizer, strict_config):
        """Full no_think training sequence. The response contains an empty `<think></think>`
        placeholder whose decoded text includes a `/think` substring. The anchor plus the
        regex negative lookbehind must exclude it, so routing stays no_think (0)."""
        full = (
            "<|im_start|>user\nQuestion /no_think<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n</think>\nAnswer<|im_end|>"
        )
        result = _determine_routing_index(_ids_batch(tokenizer, full), strict_config, tokenizer)
        assert int(result[0]) == 0, "A no_think training sample must not be mis-matched by </think>"

    def test_think_with_full_response(self, tokenizer, strict_config):
        full = (
            "<|im_start|>user\nQ /think<|im_end|>\n"
            "<|im_start|>assistant\n<think>reasoning here</think>Final answer<|im_end|>"
        )
        assert int(_determine_routing_index(_ids_batch(tokenizer, full), strict_config, tokenizer)[0]) == 1

    def test_last_wins_in_prompt(self, tokenizer, strict_config):
        """Several control words in the prompt -> the last one wins."""
        prompt = (
            "<|im_start|>user\nFirst I said /no_think but actually /think<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        assert int(_determine_routing_index(_ids_batch(tokenizer, prompt), strict_config, tokenizer)[0]) == 1

    def test_substring_not_matched(self, tokenizer, strict_config):
        """`/think_hard` contains the substring `/think`, but the regex `(?!\\w)` must prevent a false match.
        The last real control word is /no_think -> routing 0."""
        prompt = (
            "<|im_start|>user\nExplain /think_hard mode briefly. /no_think<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        assert int(_determine_routing_index(_ids_batch(tokenizer, prompt), strict_config, tokenizer)[0]) == 0


class TestRoutingBranch2NoAnchor:
    """Branch 2: bare prompt without a chat template."""

    def test_bare_think(self, tokenizer, strict_config):
        text = "What is 2+2? /think"
        assert int(_determine_routing_index(_ids_batch(tokenizer, text), strict_config, tokenizer)[0]) == 1

    def test_bare_no_think(self, tokenizer, strict_config):
        text = "Solve: 5*7 /no_think"
        assert int(_determine_routing_index(_ids_batch(tokenizer, text), strict_config, tokenizer)[0]) == 0

    def test_bare_with_close_tag_injection(self, tokenizer, strict_config):
        """A bare prompt where the user literally injects `</think>`. The regex must exclude it and use the real trailing control word."""
        text = "Now consider </think> closing tag patterns. Final: /think"
        assert int(_determine_routing_index(_ids_batch(tokenizer, text), strict_config, tokenizer)[0]) == 1


class TestNoMatchBehavior:
    """NO_MATCH (neither /think nor /no_think found) behaviour is controlled by `strict_routing_match`."""

    def test_strict_raises(self, tokenizer, strict_config):
        prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        with pytest.raises(ValueError, match="not found"):
            _determine_routing_index(_ids_batch(tokenizer, prompt), strict_config, tokenizer)

    def test_lenient_falls_back_to_default(self, tokenizer, lenient_config, caplog):
        prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        result = _determine_routing_index(_ids_batch(tokenizer, prompt), lenient_config, tokenizer)
        assert int(result[0]) == 0  # default_routing_index

    def test_lenient_respects_custom_default(self, tokenizer):
        # With default=1, falls back to think
        cfg = Qwen3PLEConfig(strict_routing_match=False, default_routing_index=1)
        cfg._name_or_path = CHECKPOINT_DIR
        prompt = "<|im_start|>user\nNo control word here<|im_end|>\n<|im_start|>assistant\n"
        result = _determine_routing_index(_ids_batch(tokenizer, prompt), cfg, tokenizer)
        assert int(result[0]) == 1


class TestConfigBackwardCompat:
    """An older checkpoint config.json lacking strict_routing_match defaults to True (raise) via getattr."""

    def test_missing_field_defaults_to_strict_true(self, tokenizer):
        # Simulate an old config without this field
        cfg = Qwen3PLEConfig()
        cfg._name_or_path = CHECKPOINT_DIR
        # Delete the field explicitly to mimic an old checkpoint
        if hasattr(cfg, "strict_routing_match"):
            del cfg.strict_routing_match
        prompt = "<|im_start|>user\nNo control word<|im_end|>\n<|im_start|>assistant\n"
        # getattr defaults to True -> should raise
        with pytest.raises(ValueError, match="not found"):
            _determine_routing_index(_ids_batch(tokenizer, prompt), cfg, tokenizer)


class TestTokenizerDecodeRoundtrip:
    """tokenizer.decode must restore the control-word token sequence to the literal /think or /no_think.
    If a tokenizer version change makes decode insert extra characters (e.g. ` /think`), the regex would fail to match -> NO_MATCH."""

    def test_think_encoding_round_trips(self, tokenizer):
        # /think must survive at least one encoding in common contexts
        for variant_text in ["/think", " /think", "\\boxed{}./think"]:
            ids = tokenizer.encode(variant_text, add_special_tokens=False)
            decoded = tokenizer.decode(ids)
            assert "/think" in decoded, (
                f"decode round-trip distorted: encode({variant_text!r}) -> {ids} -> decode = {decoded!r}; "
                "literal /think not in the result. The regex would fail to match."
            )

    def test_no_think_encoding_round_trips(self, tokenizer):
        for variant_text in ["/no_think", " /no_think", "\\boxed{}./no_think"]:
            ids = tokenizer.encode(variant_text, add_special_tokens=False)
            decoded = tokenizer.decode(ids)
            assert "/no_think" in decoded, (
                f"decode round-trip distorted: encode({variant_text!r}) -> {ids} -> decode = {decoded!r}; "
                "literal /no_think not in the result. The regex would fail to match."
            )


class TestLazyTokenizerLoad:
    """Regression guard: the training entry point passes no tokenizer and relies on the lazy
    load in `_get_routing_tokenizer`. This failed with a NameError when the modular
    converter dropped the module-level `_TOKENIZER_CACHE` definition.
    The tests deliberately pass no tokenizer to exercise the lazy-load path."""

    def test_lazy_load_via_get_routing_tokenizer(self, strict_config):
        """No tokenizer argument; goes through _get_routing_tokenizer(config) -> AutoTokenizer.from_pretrained.
        config._name_or_path is already set to the checkpoint in the strict_config fixture."""
        # Build input_ids by hand, independent of the tokenizer fixture:
        # a short sequence containing /think, using its known tokenization [20439, 766]
        ids = torch.tensor([[20439, 766]], dtype=torch.long)
        # Pass no tokenizer, to check that the lazy load does not raise NameError
        result = _determine_routing_index(ids, strict_config)  # tokenizer=None
        assert result.shape == (1,)
        assert int(result[0]) == 1

    def test_tokenizer_cache_module_variable_exists(self):
        """`_TOKENIZER_CACHE` must be a module-level accessible dict.
        Minimal canary for the converter-dropped-definition regression."""
        from src.models.qwen3_ple import modeling_qwen3_ple
        assert hasattr(modeling_qwen3_ple, "_TOKENIZER_CACHE"), \
            "_TOKENIZER_CACHE module-level variable missing — modular converter regression?"
        assert isinstance(modeling_qwen3_ple._TOKENIZER_CACHE, dict)


class TestBatchHandling:
    """Batch sizes > 1 are supported.
    _determine_routing_index computes each row independently and returns a LongTensor[B];
    Qwen3PLEMLP.forward ("Scheme A") computes both experts and selects per sample with a mask."""

    def test_returns_longtensor_shape_B(self, tokenizer, strict_config):
        prompt = "<|im_start|>user\nQ /think<|im_end|>\n<|im_start|>assistant\n"
        ids = _ids_batch(tokenizer, prompt)
        result = _determine_routing_index(ids, strict_config, tokenizer)
        assert result.shape == (1,)
        assert result.dtype == torch.long

    def test_batched_input_B2_routing(self, tokenizer, strict_config):
        """B=2 batch routing detection: a mix of think and no_think is routed correctly per row."""
        p1 = "<|im_start|>user\nQ /think<|im_end|>\n<|im_start|>assistant\n"
        p2 = "<|im_start|>user\nA /no_think<|im_end|>\n<|im_start|>assistant\n"
        ids1 = _ids_batch(tokenizer, p1)
        ids2 = _ids_batch(tokenizer, p2)
        max_len = max(ids1.shape[1], ids2.shape[1])
        pad_id = tokenizer.pad_token_id or 151643
        if ids1.shape[1] < max_len:
            ids1 = torch.cat([ids1, torch.full((1, max_len - ids1.shape[1]), pad_id, dtype=torch.long)], dim=1)
        if ids2.shape[1] < max_len:
            ids2 = torch.cat([ids2, torch.full((1, max_len - ids2.shape[1]), pad_id, dtype=torch.long)], dim=1)
        batched = torch.cat([ids1, ids2], dim=0)
        result = _determine_routing_index(batched, strict_config, tokenizer)
        assert result.shape == (2,)
        assert int(result[0]) == 1
        assert int(result[1]) == 0


class TestPhase2BatchDispatch:
    """Correctness of Qwen3PLEMLP per-sample mask dispatch (Scheme A).
    Builds a small MLP instance (CPU, no real weights needed) and verifies:
    - batch=1 behaves as plain single-expert routing (regression guard)
    - with mixed routing in batch>1, each sample goes to the right expert
    - gradients accumulate only on the selected expert (an expert with mask=0 gets zero gradient)
    """

    @pytest.fixture
    def tiny_mlp(self):
        """Build a small MLP; experts[0] outputs all zeros (act_fn(0)*x = 0) and experts[1] outputs non-zero values,
        so the two experts' outputs are distinguishable."""
        from src.models.qwen3_ple.modeling_qwen3_ple import Qwen3PLEMLP

        # A tiny config is enough to initialize the MLP
        cfg = Qwen3PLEConfig(
            hidden_size=8,
            intermediate_size=16,
            hidden_act="silu",
        )
        mlp = Qwen3PLEMLP(cfg)
        # expert 0: gate/up/down all zero -> output all zero
        with torch.no_grad():
            for p in mlp.experts[0].parameters():
                p.zero_()
            # expert 1: keeps the default random init; distinguishable as long as its output is not all zero
        mlp.eval()
        return mlp

    def test_dispatch_batch_1_think(self, tiny_mlp):
        """B=1, routing=1 -> expert 1 output, non-zero."""
        h = torch.randn(1, 4, 8)  # [B=1, S=4, H=8]
        routing = torch.tensor([1], dtype=torch.long)
        out = tiny_mlp(h, routing)
        assert out.shape == h.shape
        # expert 1 has random weights -> output should not be all zero
        assert out.abs().sum().item() > 0

    def test_dispatch_batch_1_no_think(self, tiny_mlp):
        """B=1, routing=0 -> expert 0 (zero weights) output, all zero."""
        h = torch.randn(1, 4, 8)
        routing = torch.tensor([0], dtype=torch.long)
        out = tiny_mlp(h, routing)
        assert out.shape == h.shape
        # expert 0 weights are all zero -> output all zero
        assert out.abs().sum().item() == 0.0

    def test_dispatch_mixed_batch_B2(self, tiny_mlp):
        """B=2 mixed: routing=[0, 1] -> sample 0 uses expert 0 (all zero), sample 1 uses expert 1 (non-zero)."""
        h = torch.randn(2, 4, 8)
        routing = torch.tensor([0, 1], dtype=torch.long)
        out = tiny_mlp(h, routing)
        assert out.shape == h.shape
        # sample 0 uses expert 0 -> all zero
        assert out[0].abs().sum().item() == 0.0, "sample 0 should route to expert 0 (zero weights)"
        # sample 1 uses expert 1 -> non-zero
        assert out[1].abs().sum().item() > 0, "sample 1 should route to expert 1 (random weights)"

    def test_dispatch_mixed_batch_B2_reverse(self, tiny_mlp):
        """B=2 mixed: routing=[1, 0] (reversed) -> sample 0 non-zero, sample 1 all zero."""
        h = torch.randn(2, 4, 8)
        routing = torch.tensor([1, 0], dtype=torch.long)
        out = tiny_mlp(h, routing)
        assert out[0].abs().sum().item() > 0
        assert out[1].abs().sum().item() == 0.0

    def test_dispatch_batch_4_mixed(self, tiny_mlp):
        """B=4 mixed routing=[1, 0, 1, 0]."""
        h = torch.randn(4, 4, 8)
        routing = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        out = tiny_mlp(h, routing)
        # Samples with routing=1 (0, 2) should have non-zero output; samples with routing=0 (1, 3) should be all zero
        assert out[0].abs().sum().item() > 0
        assert out[1].abs().sum().item() == 0.0
        assert out[2].abs().sum().item() > 0
        assert out[3].abs().sum().item() == 0.0

    def test_dispatch_gradient_isolation(self, tiny_mlp):
        """When a sample goes to expert 1, expert 0's .grad must not be polluted by that sample.
        The Scheme A mask should make the gradient of an unselected expert exactly 0 (per sample)."""
        tiny_mlp.train()
        h = torch.randn(2, 4, 8, requires_grad=False)
        routing = torch.tensor([1, 1], dtype=torch.long)  # both samples go to expert 1
        out = tiny_mlp(h, routing)
        loss = out.sum()
        loss.backward()
        # expert 0 took no part (mask=0), so its .grad should be all zero
        for name, p in tiny_mlp.experts[0].named_parameters():
            assert p.grad is not None, f"expert 0 {name}.grad missing"
            assert p.grad.abs().sum().item() == 0.0, (
                f"expert 0 {name}.grad should be 0 when no sample routes to it; "
                f"got sum={p.grad.abs().sum().item()}"
            )
        # expert 1 should have non-zero gradients
        any_nonzero = False
        for name, p in tiny_mlp.experts[1].named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                any_nonzero = True
                break
        assert any_nonzero, "expert 1 should have non-zero gradients"

    def test_dispatch_mask_dtype_bf16_compatible(self, tiny_mlp):
        """The mask is cast to out0.dtype; checks that the bf16 inference path does not break."""
        tiny_mlp.eval()
        # Cast the MLP to bf16 to mimic the training/inference dtype
        tiny_mlp.bfloat16()
        h = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        routing = torch.tensor([0, 1], dtype=torch.long)
        out = tiny_mlp(h, routing)
        assert out.dtype == torch.bfloat16
        assert out.shape == h.shape


# ---------------------- KV cache + generate smoke test ----------------------
# Runs a real-model forward + generate. Works on CPU but is slow; takes seconds on a GPU.

@pytest.mark.slow
class TestGenerateSmoke:
    """KV cache argument naming, and consistency between generate and prefill decoding."""

    def test_load_and_generate_5_tokens(self, tokenizer):
        """Load the checkpoint and generate 5 tokens, checking that the KV cache does not silently fail."""
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT_DIR, trust_remote_code=True, torch_dtype=dtype
        ).to(device).eval()

        prompt = "<|im_start|>user\nWhat is 2+2? /think<|im_end|>\n<|im_start|>assistant\n"
        ids = _ids_batch(tokenizer, prompt).to(device)

        with torch.no_grad():
            output = model.generate(ids, max_new_tokens=5, do_sample=False)

        # At least 5 new tokens were generated
        assert output.shape[1] >= ids.shape[1] + 5

        # The routing index is cached with the right value (not overwritten by a wrong detection inside the generate loop)
        assert model._cached_routing_index is not None
        assert model._cached_routing_index.shape == (1,)
        assert int(model._cached_routing_index[0]) == 1, "/think → expert 1"


class TestSchemeBInference:
    """Scheme B (inference path, triggered under torch.no_grad): computes only the selected expert and is bit-identical to Scheme A."""

    @pytest.fixture
    def tiny_mlp(self):
        from src.models.qwen3_ple.modeling_qwen3_ple import Qwen3PLEMLP
        cfg = Qwen3PLEConfig(hidden_size=8, intermediate_size=16, hidden_act="silu")
        mlp = Qwen3PLEMLP(cfg)
        # Give the two experts different weights so we can tell which one was used
        with torch.no_grad():
            for p in mlp.experts[0].parameters():
                p.zero_()  # expert 0 outputs all zero
            # expert 1 stays random
        mlp.eval()
        return mlp

    def test_scheme_b_homogeneous_no_think(self, tiny_mlp):
        """no_grad + all routing=0 -> only expert 0 (zero weights) is called -> output all zero."""
        h = torch.randn(3, 4, 8)
        routing = torch.tensor([0, 0, 0], dtype=torch.long)
        with torch.no_grad():
            out = tiny_mlp(h, routing)
        assert out.shape == h.shape
        assert out.abs().sum().item() == 0.0

    def test_scheme_b_homogeneous_think(self, tiny_mlp):
        """no_grad + all routing=1 -> only expert 1 (random weights) is called -> output non-zero."""
        h = torch.randn(3, 4, 8)
        routing = torch.tensor([1, 1, 1], dtype=torch.long)
        with torch.no_grad():
            out = tiny_mlp(h, routing)
        assert out.abs().sum().item() > 0

    def test_scheme_b_mixed_batch(self, tiny_mlp):
        """no_grad + routing=[0,1,0,1] -> segmented dispatch; samples 0/2 use expert 0 (zero), 1/3 use expert 1 (non-zero)."""
        h = torch.randn(4, 4, 8)
        routing = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        with torch.no_grad():
            out = tiny_mlp(h, routing)
        assert out[0].abs().sum().item() == 0.0
        assert out[1].abs().sum().item() > 0
        assert out[2].abs().sum().item() == 0.0
        assert out[3].abs().sum().item() > 0

    def test_scheme_a_equals_scheme_b(self, tiny_mlp):
        """With the same input, Scheme A (grad on) and Scheme B (no_grad) outputs are bit-identical.
        This shows that switching to Scheme B at inference does not change results, only saves compute."""
        h = torch.randn(4, 4, 8)
        routing = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        # Scheme A: grad enabled (default)
        out_a = tiny_mlp(h, routing)
        # Scheme B: no_grad
        with torch.no_grad():
            out_b = tiny_mlp(h, routing)
        assert torch.equal(out_a, out_b), (
            f"Scheme A vs B mismatch: max diff = {(out_a - out_b).abs().max().item()}"
        )

    def test_scheme_b_bf16(self, tiny_mlp):
        """The no_grad + bf16 inference path does not break."""
        tiny_mlp.bfloat16()
        h = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        routing = torch.tensor([0, 1], dtype=torch.long)
        with torch.no_grad():
            out = tiny_mlp(h, routing)
        assert out.dtype == torch.bfloat16
        assert out.shape == h.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
