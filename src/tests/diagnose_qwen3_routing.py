#!/usr/bin/env python3
"""Qwen3 PLE 诊断脚本：不使用量化，直接对比 bf16 加载的输出。

用于排查 PLE vs 源模型输出不一致的问题。
"""

import sys
import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

SOURCE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "source", "Qwen", "Qwen3-4B-Instruct-2507"
)
PLE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen3_4b_instruct_ple_initialized")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ===== Test 1: Check config =====
    print("\n===== Test 1: Config check =====")
    import json
    with open(os.path.join(PLE_MODEL_PATH, "config.json")) as f:
        pl_config = json.load(f)
    print(f"  model_type: {pl_config['model_type']}")
    print(f"  tie_word_embeddings: {pl_config.get('tie_word_embeddings')}")
    print(f"  think_token_id: {pl_config.get('think_token_id')}")
    print(f"  no_think_token_id: {pl_config.get('no_think_token_id')}")
    print(f"  default_routing_index: {pl_config.get('default_routing_index')}")

    # ===== Test 2: Load source model (bf16, no quantization) =====
    print("\n===== Test 2: Load source model (bf16) =====")
    source_tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_PATH)
    source_model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    source_model.eval()
    print(f"  Source model loaded: {type(source_model).__name__}")
    print(f"  lm_head weight id == embed_tokens weight id: "
          f"{source_model.lm_head.weight.data_ptr() == source_model.model.embed_tokens.weight.data_ptr()}")

    # ===== Test 3: Load PLE model (bf16, no quantization) =====
    print("\n===== Test 3: Load PLE model (bf16) =====")
    ple_tokenizer = AutoTokenizer.from_pretrained(PLE_MODEL_PATH)
    ple_model = AutoModelForCausalLM.from_pretrained(
        PLE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ple_model.eval()
    print(f"  PLE model loaded: {type(ple_model).__name__}")
    print(f"  lm_head weight id == embed_tokens weight id: "
          f"{ple_model.lm_head.weight.data_ptr() == ple_model.model.embed_tokens.weight.data_ptr()}")

    # ===== Test 4: Single forward pass comparison (no generate) =====
    print("\n===== Test 4: Single forward pass (logits comparison) =====")
    prompt = "What is 2+3?"
    inputs_src = source_tokenizer(prompt, return_tensors="pt").to(device)
    inputs_moe = ple_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        src_out = source_model(**inputs_src)
        moe_out = ple_model(**inputs_moe)

    src_logits = src_out.logits[0, -1, :]  # last token logits
    moe_logits = moe_out.logits[0, -1, :]

    print(f"  Source last-token logits: mean={src_logits.mean():.4f}, std={src_logits.std():.4f}")
    print(f"  PLE last-token logits: mean={moe_logits.mean():.4f}, std={moe_logits.std():.4f}")
    print(f"  Max abs diff: {(src_logits - moe_logits).abs().max():.6f}")
    print(f"  All close (atol=1e-4): {torch.allclose(src_logits, moe_logits, atol=1e-4)}")

    src_next = src_logits.argmax().item()
    moe_next = moe_logits.argmax().item()
    print(f"  Source next token: {src_next} = '{source_tokenizer.decode([src_next])}'")
    print(f"  PLE next token: {moe_next} = '{ple_tokenizer.decode([moe_next])}'")

    # ===== Test 5: Multi-step generation comparison =====
    print("\n===== Test 5: Generate comparison (greedy, 20 tokens) =====")
    with torch.no_grad():
        src_gen = source_model.generate(
            **inputs_src, max_new_tokens=20, do_sample=False,
            temperature=None, top_p=None,
        )
        moe_gen = ple_model.generate(
            **inputs_moe, max_new_tokens=20, do_sample=False,
            temperature=None, top_p=None,
            routing_index=0,
        )

    src_new = src_gen[0][inputs_src["input_ids"].shape[1]:]
    moe_new = moe_gen[0][inputs_moe["input_ids"].shape[1]:]

    print(f"  Source: {source_tokenizer.decode(src_new)}")
    print(f"  PLE: {ple_tokenizer.decode(moe_new)}")
    print(f"  Source IDs: {src_new.tolist()}")
    print(f"  PLE IDs: {moe_new.tolist()}")
    print(f"  Match: {torch.equal(src_new, moe_new)}")

    # ===== Test 6: Check intermediate layer outputs =====
    print("\n===== Test 6: Layer-by-layer hidden states comparison =====")
    # Compare hidden states after first decoder layer
    with torch.no_grad():
        src_embeds = source_model.model.embed_tokens(inputs_src["input_ids"])
        moe_embeds = ple_model.model.embed_tokens(inputs_moe["input_ids"])
        print(f"  Embed match: {torch.allclose(src_embeds, moe_embeds, atol=1e-6)}")
        print(f"  Embed max diff: {(src_embeds - moe_embeds).abs().max():.8f}")

    # ===== Test 7: Check expert 0 MLP weights match source MLP =====
    print("\n===== Test 7: Runtime weight comparison (layer 0 MLP) =====")
    src_gate = source_model.model.layers[0].mlp.gate_proj.weight.data
    moe_gate = ple_model.model.layers[0].mlp.experts[0].gate_proj.weight.data
    print(f"  gate_proj match: {torch.equal(src_gate, moe_gate)}")
    print(f"  gate_proj max diff: {(src_gate.float() - moe_gate.float()).abs().max():.8f}")

    src_up = source_model.model.layers[0].mlp.up_proj.weight.data
    moe_up = ple_model.model.layers[0].mlp.experts[0].up_proj.weight.data
    print(f"  up_proj match: {torch.equal(src_up, moe_up)}")

    src_down = source_model.model.layers[0].mlp.down_proj.weight.data
    moe_down = ple_model.model.layers[0].mlp.experts[0].down_proj.weight.data
    print(f"  down_proj match: {torch.equal(src_down, moe_down)}")

    # Check attention weights too
    src_q = source_model.model.layers[0].self_attn.q_proj.weight.data
    moe_q = ple_model.model.layers[0].self_attn.q_proj.weight.data
    print(f"  q_proj match: {torch.equal(src_q, moe_q)}")

    print("\n===== Diagnosis complete =====")

    # Cleanup
    del source_model, ple_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
