#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Register /think and /no_think as special tokens of the PLE tokenizer.

What it does:
1. Adds /think and /no_think to the tokenizer as additional_special_tokens
   - they are recognized as whole tokens and are not split by BPE
   - they get new token IDs (after the existing special tokens)
2. Updates think_token_id and no_think_token_id in config.json
3. If the new token IDs exceed the original vocab_size (e.g. LLaMA 3.1), resizes the embedding and lm_head
4. Verifies the registration

Usage:
    python -m src.utils.register_control_tokens --model_path models/llama_ple_initialized

Notes:
- Qwen family: vocab_size has spare room (152064), so no resize is needed for the new tokens
- LLaMA 3.1: vocab_size=128256 is fully used, so new token IDs 128256/128257 require an embedding resize
- Resizing keeps the original weights: the first N rows are preserved, only randomly initialized rows are appended at the end
- The embeddings of the new tokens are untrained (random init) and are learned during fine-tuning
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Default path (backward compatible)
DEFAULT_PLE_PATH = os.path.join(PROJECT_ROOT, "models", "ple_initialized")

# Control tokens to register
THINK_TOKEN = "/think"
NO_THINK_TOKEN = "/no_think"


def main():
    parser = argparse.ArgumentParser(
        description="Register /think and /no_think as special tokens of the PLE tokenizer"
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_PLE_PATH,
        help=f"PLE model path (default: {DEFAULT_PLE_PATH})"
    )
    args = parser.parse_args()
    PLE_PATH = args.model_path

    from transformers import AutoTokenizer

    print(f"Model path: {PLE_PATH}")

    # ===== 1. Load the tokenizer =====
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PLE_PATH, trust_remote_code=True)
    print(f"  Current vocab size: {len(tokenizer)}")
    print(f"  Current number of special tokens: {len(tokenizer.all_special_tokens)}")

    # Check whether they are already registered
    existing_special = tokenizer.special_tokens_map.get("additional_special_tokens", [])
    # additional_special_tokens may be a list of AddedToken objects; convert to strings
    existing_str = [str(t) for t in existing_special]
    if THINK_TOKEN in existing_str and NO_THINK_TOKEN in existing_str:
        think_id = tokenizer.convert_tokens_to_ids(THINK_TOKEN)
        no_think_id = tokenizer.convert_tokens_to_ids(NO_THINK_TOKEN)
        print(f"\n  Control tokens already registered:")
        print(f"    {THINK_TOKEN} → id={think_id}")
        print(f"    {NO_THINK_TOKEN} → id={no_think_id}")
        print("  Skipping registration, going straight to verification...")
    else:
        # ===== 2. Register the new special tokens =====
        print(f"\n[2] Registering control tokens: {THINK_TOKEN}, {NO_THINK_TOKEN}")

        # Build the new additional_special_tokens list (keeping the existing ones)
        new_special_tokens = existing_str + [THINK_TOKEN, NO_THINK_TOKEN]
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": new_special_tokens
        })
        print(f"  Number of tokens added: {num_added}")

        think_id = tokenizer.convert_tokens_to_ids(THINK_TOKEN)
        no_think_id = tokenizer.convert_tokens_to_ids(NO_THINK_TOKEN)
        print(f"  {THINK_TOKEN} → id={think_id}")
        print(f"  {NO_THINK_TOKEN} → id={no_think_id}")

        # ===== 3. Save the updated tokenizer =====
        print(f"\n[3] Saving tokenizer to {PLE_PATH}")
        tokenizer.save_pretrained(PLE_PATH)
        print("  Tokenizer saved")

    # ===== 4. Update config.json =====
    config_path = os.path.join(PLE_PATH, "config.json")
    print(f"\n[4] Updating config.json...")
    with open(config_path, "r") as f:
        config = json.load(f)

    old_think = config.get("think_token_id")
    old_no_think = config.get("no_think_token_id")
    config["think_token_id"] = think_id
    config["no_think_token_id"] = no_think_id

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")  # trailing newline

    print(f"  think_token_id:    {old_think} → {think_id}")
    print(f"  no_think_token_id: {old_no_think} → {no_think_id}")

    # ===== 4.5. Resize the embedding (only if a token ID exceeds vocab_size) =====
    max_new_id = max(think_id, no_think_id)
    current_vocab_size = config.get("vocab_size", 0)
    if max_new_id >= current_vocab_size:
        new_vocab_size = max_new_id + 1
        print(f"\n[4.5] Resize embedding: {current_vocab_size} → {new_vocab_size}")
        print(f"  New token ID ({max_new_id}) >= original vocab_size ({current_vocab_size}), resize needed")
        print(f"  The original {current_vocab_size} embedding rows are kept; appending {new_vocab_size - current_vocab_size} randomly initialized rows")

        from transformers import AutoModelForCausalLM
        import torch

        print(f"  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            PLE_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.resize_token_embeddings(new_vocab_size)
        print(f"  embed_tokens shape: {model.get_input_embeddings().weight.shape}")
        if hasattr(model, 'lm_head'):
            print(f"  lm_head shape: {model.lm_head.weight.shape}")

        print(f"  Saving resized model weights...")
        model.save_pretrained(PLE_PATH)
        print(f"  Model weights saved")

        # Update vocab_size in config.json
        with open(config_path, "r") as f:
            config = json.load(f)
        config["vocab_size"] = new_vocab_size
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  config.json vocab_size updated: {current_vocab_size} → {new_vocab_size}")

        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"\n  vocab_size={current_vocab_size} is large enough for the new tokens (max_id={max_new_id}); no resize needed")

    # ===== 5. Verify =====
    print(f"\n[5] Verifying...")

    # Reload and verify
    tokenizer2 = AutoTokenizer.from_pretrained(PLE_PATH, trust_remote_code=True)

    # Verify single-token encoding
    think_encoded = tokenizer2.encode(THINK_TOKEN, add_special_tokens=False)
    no_think_encoded = tokenizer2.encode(NO_THINK_TOKEN, add_special_tokens=False)

    assert len(think_encoded) == 1, (
        f"{THINK_TOKEN} should encode to a single token, got: {think_encoded}"
    )
    assert len(no_think_encoded) == 1, (
        f"{NO_THINK_TOKEN} should encode to a single token, got: {no_think_encoded}"
    )
    assert think_encoded[0] == think_id, (
        f"{THINK_TOKEN} encoded ID mismatch: {think_encoded[0]} != {think_id}"
    )
    assert no_think_encoded[0] == no_think_id, (
        f"{NO_THINK_TOKEN} encoded ID mismatch: {no_think_encoded[0]} != {no_think_id}"
    )

    # Verify they are not split inside a sentence
    test_text = f"Hello {THINK_TOKEN} world"
    test_ids = tokenizer2.encode(test_text, add_special_tokens=False)
    assert think_id in test_ids, (
        f"{THINK_TOKEN} not recognized as a single token inside a sentence: {test_ids}"
    )

    test_text2 = f"Hello {NO_THINK_TOKEN} world"
    test_ids2 = tokenizer2.encode(test_text2, add_special_tokens=False)
    assert no_think_id in test_ids2, (
        f"{NO_THINK_TOKEN} not recognized as a single token inside a sentence: {test_ids2}"
    )

    # Verify decode
    decoded_think = tokenizer2.decode([think_id])
    decoded_no_think = tokenizer2.decode([no_think_id])
    assert decoded_think.strip() == THINK_TOKEN, (
        f"decode mismatch: {repr(decoded_think)} != {repr(THINK_TOKEN)}"
    )
    assert decoded_no_think.strip() == NO_THINK_TOKEN, (
        f"decode mismatch: {repr(decoded_no_think)} != {repr(NO_THINK_TOKEN)}"
    )

    # Verify the config was updated
    with open(config_path, "r") as f:
        config_check = json.load(f)
    assert config_check["think_token_id"] == think_id
    assert config_check["no_think_token_id"] == no_think_id

    # Verify the embedding layer is large enough (should hold after resize)
    vocab_size = config_check.get("vocab_size", 0)
    assert vocab_size > max(think_id, no_think_id), (
        f"vocab_size={vocab_size} is too small for the new tokens (max_id={max(think_id, no_think_id)})"
    )

    print(f"\n  All checks passed!")
    print(f"\n{'='*60}")
    print(f"  Registration summary:")
    print(f"    {THINK_TOKEN:12s} → id={think_id}")
    print(f"    {NO_THINK_TOKEN:12s} → id={no_think_id}")
    print(f"    vocab_size={vocab_size}")
    print(f"    config.json updated")
    print(f"    tokenizer updated")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
