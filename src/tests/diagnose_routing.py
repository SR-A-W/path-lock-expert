#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLE routing diagnostic script (4-bit quantization, everything loaded on GPU).

Diagnostic goals:
1. Verify that the /think and /no_think control tokens are registered and encoded correctly
2. Verify that _determine_routing_index detects control tokens in input_ids
3. Verify that routing switches automatically in generate when control tokens are present
4. Verify that routing_index is passed through correctly in forward and generate
5. Confirm that expert 0 and expert 1 produce different outputs

Usage:
    cd ./
    python src/tests/diagnose_routing.py
"""

import gc
import sys
import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLE_PATH = os.path.join(PROJECT_ROOT, "models", "ple_initialized")
QWEN_PATH = os.path.join(PROJECT_ROOT, "models", "source", "Qwen2.5-7B-Instruct")
DEEPSEEK_PATH = os.path.join(PROJECT_ROOT, "models", "source", "DeepSeek-R1-Distill-Qwen-7B")


def load_model_4bit(model_path, trust_remote_code=False):
    """Load a model onto the GPU with 4-bit quantization."""
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


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def generate_text(model, tokenizer, prompt, max_new_tokens=80, routing_index=None):
    """Generate text, optionally with an explicit routing_index."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    if routing_index is not None:
        gen_kwargs["routing_index"] = routing_index

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Keep only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    # ===== Diagnostic 1: control-token registration =====
    section("Diagnostic 1: control-token registration")

    tokenizer = load_tokenizer(PLE_PATH)

    # Check whether /think and /no_think are special tokens
    print(f"  additional_special_tokens in tokenizer:")
    for t in tokenizer.special_tokens_map.get("additional_special_tokens", []):
        print(f"    {repr(str(t))}")

    # Encoding check: each must be a single token
    think_ids = tokenizer.encode("/think", add_special_tokens=False)
    no_think_ids = tokenizer.encode("/no_think", add_special_tokens=False)
    print(f"\n  /think    encode → {think_ids} (single token={'✓' if len(think_ids)==1 else '✗'})")
    print(f"  /no_think encode → {no_think_ids} (single token={'✓' if len(no_think_ids)==1 else '✗'})")

    think_token_id = think_ids[0] if len(think_ids) == 1 else None
    no_think_token_id = no_think_ids[0] if len(no_think_ids) == 1 else None
    print(f"\n  think_token_id    = {think_token_id}")
    print(f"  no_think_token_id = {no_think_token_id}")

    # Check that the tokens are not split inside a sentence
    test_text = "Hello /think world /no_think end"
    test_enc = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"\n  Sentence: {repr(test_text)}")
    print(f"  Encoding: {test_enc}")
    print(f"  /think  in encoding: {'✓' if think_token_id in test_enc else '✗'}")
    print(f"  /no_think in encoding: {'✓' if no_think_token_id in test_enc else '✗'}")

    # ===== Diagnostic 2: _determine_routing_index control-token detection =====
    section("Diagnostic 2: _determine_routing_index control-token detection")

    # Inlined routing function (same logic as modeling_qwen2_ple.py)
    class FakeConfig:
        pass

    def _determine_routing_index_test(input_ids, config):
        think_id = config.think_token_id
        no_think_id = config.no_think_token_id
        default_idx = getattr(config, "default_routing_index", 0)
        if think_id is None and no_think_id is None:
            return default_idx
        last_think_pos = -1
        last_no_think_pos = -1
        flat = input_ids.view(-1)
        if think_id is not None:
            positions = (flat == think_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                last_think_pos = positions[-1].item()
        if no_think_id is not None:
            positions = (flat == no_think_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                last_no_think_pos = positions[-1].item()
        if last_think_pos == -1 and last_no_think_pos == -1:
            return default_idx
        if last_think_pos > last_no_think_pos:
            return 1
        else:
            return 0

    cfg = FakeConfig()
    cfg.think_token_id = think_token_id
    cfg.no_think_token_id = no_think_token_id
    cfg.default_routing_index = 0

    # Test various scenarios
    cases = [
        ("no control token",      "What is 2+3?"),
        ("contains /think",         "What is 2+3? /think"),
        ("contains /no_think",      "What is 2+3? /no_think"),
        ("/think after /no_think","What /no_think is /think 2+3?"),
        ("/no_think after /think","What /think is /no_think 2+3?"),
    ]

    for name, text in cases:
        ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)])
        result = _determine_routing_index_test(ids, cfg)
        print(f"  {name:30s} → routing_index = {result}")

    # ===== Diagnostic 3: load the PLE model and test forward routing =====
    section("Diagnostic 3: PLE forward routing test (4-bit)")

    print("  Loading PLE model (4-bit)...")
    pl_model = load_model_4bit(PLE_PATH, trust_remote_code=True)

    # Check the routing settings in the config
    print(f"  config.think_token_id     = {pl_model.config.think_token_id}")
    print(f"  config.no_think_token_id  = {pl_model.config.no_think_token_id}")
    print(f"  config.default_routing_index = {pl_model.config.default_routing_index}")

    # Test routing_index passing in forward
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(pl_model.device)
    with torch.no_grad():
        logits_r0 = pl_model(input_ids=input_ids, routing_index=0).logits.cpu()
        logits_r1 = pl_model(input_ids=input_ids, routing_index=1).logits.cpu()
        # No routing_index -> goes through _determine_routing_index -> returns the default
        logits_default = pl_model(input_ids=input_ids).logits.cpu()

    print(f"\n  forward results:")
    print(f"  routing_index=0  top5 token ids: {logits_r0[0, -1].topk(5).indices.tolist()}")
    print(f"  routing_index=1  top5 token ids: {logits_r1[0, -1].topk(5).indices.tolist()}")
    print(f"  no routing_index top5 token ids: {logits_default[0, -1].topk(5).indices.tolist()}")
    print(f"  routing_index=0 == default?  {torch.equal(logits_r0, logits_default)}")
    print(f"  routing_index=1 == default?  {torch.equal(logits_r1, logits_default)}")
    print(f"  routing_index=0 == routing_index=1?  {torch.equal(logits_r0, logits_r1)}")

    # ===== Diagnostic 4: explicit routing in generate =====
    section("Diagnostic 4: PLE generate explicit routing test")

    prompt = "What is 2 + 3? Answer:"

    print(f"\n  Prompt: '{prompt}'")

    try:
        text_r0 = generate_text(pl_model, tokenizer, prompt, routing_index=0)
        print(f"\n  [generate routing_index=0] {text_r0[:200]}")
    except Exception as e:
        print(f"\n  [generate routing_index=0] ERROR: {e}")

    try:
        text_r1 = generate_text(pl_model, tokenizer, prompt, routing_index=1)
        print(f"\n  [generate routing_index=1] {text_r1[:200]}")
    except Exception as e:
        print(f"\n  [generate routing_index=1] ERROR: {e}")

    try:
        text_default = generate_text(pl_model, tokenizer, prompt)
        print(f"\n  [generate no routing_index] {text_default[:200]}")
    except Exception as e:
        print(f"\n  [generate no routing_index] ERROR: {e}")

    if text_r0 == text_r1:
        print("\n  ✗ BUG: generate outputs for routing_index=0 and =1 are identical! Routing has no effect!")
    else:
        print("\n  ✓ generate outputs for routing_index=0 and =1 differ; explicit routing works.")

    print(f"  Last cached _cached_routing_index = {pl_model._cached_routing_index}")

    # ===== Diagnostic 5: automatic routing from control tokens in generate =====
    section("Diagnostic 5: PLE generate control-token auto-routing test")

    prompt_think = "What is 2 + 3? /think Answer:"
    prompt_no_think = "What is 2 + 3? /no_think Answer:"
    prompt_plain = "What is 2 + 3? Answer:"

    print(f"\n  Prompt with /think:      '{prompt_think}'")
    print(f"  Prompt with /no_think:   '{prompt_no_think}'")
    print(f"  Prompt, no control token: '{prompt_plain}'")

    try:
        text_think = generate_text(pl_model, tokenizer, prompt_think)
        cached_think = pl_model._cached_routing_index
        print(f"\n  [/think]    routing={cached_think} → {text_think[:200]}")
    except Exception as e:
        print(f"\n  [/think]    ERROR: {e}")
        cached_think = None

    try:
        text_no_think = generate_text(pl_model, tokenizer, prompt_no_think)
        cached_no_think = pl_model._cached_routing_index
        print(f"\n  [/no_think] routing={cached_no_think} → {text_no_think[:200]}")
    except Exception as e:
        print(f"\n  [/no_think] ERROR: {e}")
        cached_no_think = None

    try:
        text_plain = generate_text(pl_model, tokenizer, prompt_plain)
        cached_plain = pl_model._cached_routing_index
        print(f"\n  [no ctrl]   routing={cached_plain} → {text_plain[:200]}")
    except Exception as e:
        print(f"\n  [no ctrl]   ERROR: {e}")
        cached_plain = None

    # Check the routing detection results
    print(f"\n  Routing detection results:")
    print(f"    /think    → _cached_routing_index = {cached_think} (expected 1)")
    print(f"    /no_think → _cached_routing_index = {cached_no_think} (expected 0)")
    print(f"    no control → _cached_routing_index = {cached_plain} (expected 0, default)")

    if cached_think == 1 and cached_no_think == 0 and cached_plain == 0:
        print("\n  ✓ Control-token auto-routing is fully correct!")
    else:
        print("\n  ✗ Control-token auto-routing has problems!")

    if text_think != text_no_think:
        print("  ✓ /think and /no_think produce different outputs")
    else:
        print("  ✗ /think and /no_think produce identical outputs")

    if text_no_think == text_plain:
        print("  ✓ /no_think and no-control-token outputs are identical (both use default=0)")
    else:
        print("  ~ /no_think and no-control-token outputs differ (the prompts differ; acceptable)")

    # Free the PLE model
    del pl_model
    gc.collect()
    torch.cuda.empty_cache()

    # ===== Summary =====
    section("Diagnostic summary")
    print("""
  1. Are /think and /no_think registered as single-token special tokens? (see Diagnostic 1)
  2. Does _determine_routing_index detect control tokens correctly? (see Diagnostic 2)
  3. Is routing_index passed through correctly in forward? (see Diagnostic 3)
  4. Is an explicit routing_index passed through correctly in generate? (see Diagnostic 4)
  5. Do control tokens in generate trigger the correct routing automatically? (see Diagnostic 5)
    """)


if __name__ == "__main__":
    main()
