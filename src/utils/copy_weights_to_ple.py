#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy source-model weights into a PLE model (memory-optimized, supports Qwen2/Qwen3)

Streams the safetensors shards, reading and writing one shard at a time instead of loading the whole model.
Peak memory is bounded by the size of one shard (about 5GB), so it runs on machines with limited RAM.

Weight-copy strategy (PLE, no gate/router weights):
- All weights are copied from a single source model:
  - Embedding, Attention, LayerNorm, lm_head -> copied directly (shared layers)
  - MLP weights (gate_proj, up_proj, down_proj) -> copied to both Expert 0 and Expert 1
- The two experts start identical and diverge naturally during SFT.
- PLE has no gate/router weights; routing is decided entirely by the control tokens.

Supported model families:
- qwen2: Qwen2.5-7B-Instruct etc. (attention has bias, no q/k norm)
- qwen3: Qwen3-4B-Instruct etc. (attention has no bias, has q/k norm)

Usage:
    python -m src.utils.copy_weights_to_ple \\
        --source_model_path ./models/source/Qwen2.5-7B-Instruct \\
        --output_path ./models/ple_initialized

    python -m src.utils.copy_weights_to_ple \\
        --source_model_path ./models/source/Qwen3-4B-Instruct-2507 \\
        --output_path ./models/qwen3_4b_instruct_ple_initialized

Dependencies:
    pip install transformers torch safetensors
"""

import argparse
import gc
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Project root directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== Model-family config mapping ====================

# PLE config parameters for each model family
MODEL_FAMILY_CONFIG = {
    "qwen2": {
        "model_type": "qwen2_ple",
        "architectures": ["Qwen2PleForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_qwen2_ple.Qwen2PleConfig",
            "AutoModelForCausalLM": "modeling_qwen2_ple.Qwen2PleForCausalLM",
        },
        "model_code_dir": "qwen2_ple",
        "config_filename": "configuration_qwen2_ple.py",
        "modeling_filename": "modeling_qwen2_ple.py",
        # Per-layer weight count for Qwen2 (after PLE conversion):
        # self_attn: q.w, q.b, k.w, k.b, v.w, v.b, o.w = 7
        # layernorm: input_layernorm.w, post_attention_layernorm.w = 2
        # MLP experts: 3 weights × 2 experts = 6
        # Total = 7 + 2 + 6 = 15
        "expected_per_layer": 15,
    },
    "qwen3": {
        "model_type": "qwen3_ple",
        "architectures": ["Qwen3PLEForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_qwen3_ple.Qwen3PLEConfig",
            "AutoModelForCausalLM": "modeling_qwen3_ple.Qwen3PLEForCausalLM",
        },
        "model_code_dir": "qwen3_ple",
        "config_filename": "configuration_qwen3_ple.py",
        "modeling_filename": "modeling_qwen3_ple.py",
        # Per-layer weight count for Qwen3 (after PLE conversion):
        # self_attn: q.w, k.w, v.w, o.w = 4 (no bias, attention_bias=False)
        # self_attn: q_norm.w, k_norm.w = 2 (QK normalization, Qwen3-specific)
        # layernorm: input_layernorm.w, post_attention_layernorm.w = 2
        # MLP experts: 3 weights × 2 experts = 6
        # Total = 4 + 2 + 2 + 6 = 14
        "expected_per_layer": 14,
    },
    "llama": {
        "model_type": "llama_ple",
        "architectures": ["LlamaPleForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_llama_ple.LlamaPleConfig",
            "AutoModelForCausalLM": "modeling_llama_ple.LlamaPleForCausalLM",
        },
        "model_code_dir": "llama_ple",
        "config_filename": "configuration_llama_ple.py",
        "modeling_filename": "modeling_llama_ple.py",
        # Per-layer weight count for LLaMA (after PLE conversion):
        # self_attn: q.w, k.w, v.w, o.w = 4 (no bias, no q/k norm)
        # layernorm: input_layernorm.w, post_attention_layernorm.w = 2
        # MLP experts: 3 weights × 2 experts = 6
        # Total = 4 + 2 + 6 = 12
        "expected_per_layer": 12,
    },
    "phi3": {
        "model_type": "phi4_mini_ple",
        "architectures": ["Phi4MiniPleForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_phi4_mini_ple.Phi4MiniPleConfig",
            "AutoModelForCausalLM": "modeling_phi4_mini_ple.Phi4MiniPleForCausalLM",
        },
        "model_code_dir": "phi4_mini_ple",
        "config_filename": "configuration_phi4_mini_ple.py",
        "modeling_filename": "modeling_phi4_mini_ple.py",
        # Per-layer weight count for Phi3/Phi-4-mini (after PLE conversion):
        # self_attn: qkv_proj.w, o_proj.w = 2 (fused QKV, no bias)
        # layernorm: input_layernorm.w, post_attention_layernorm.w = 2
        # MLP experts: 2 weights (gate_up_proj, down_proj) × 2 experts = 4
        # Total = 2 + 2 + 4 = 8
        "expected_per_layer": 8,
    },
}


def detect_model_family(source_config: dict) -> str:
    """Auto-detect the model family from the source model config

    Args:
        source_config: contents of the source model's config.json

    Returns:
        model family name: "qwen2" or "qwen3"

    Raises:
        ValueError: unsupported model type
    """
    model_type = source_config.get("model_type", "")
    if model_type.startswith("qwen3"):
        return "qwen3"
    elif model_type.startswith("qwen2"):
        return "qwen2"
    elif model_type == "llama":
        return "llama"
    elif model_type == "phi3":
        return "phi3"
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Currently supported: qwen2, qwen3, llama, phi3. "
            f"Use --model_family to specify it manually."
        )


# ==================== PLE config generation ====================

def create_ple_config(source_config: dict, output_path: str, model_family: str) -> dict:
    """Create the PLE config from the source model config

    The PLE config inherits all base parameters of the source model (hidden_size, attention parameters, etc.),
    changes model_type and architectures, and adds the routing-related settings.

    Args:
        source_config: contents of the source model's config.json
        output_path: output directory path
        model_family: model family ("qwen2" or "qwen3")

    Returns:
        PLE config dict
    """
    family_config = MODEL_FAMILY_CONFIG[model_family]

    print(f"\n{'='*60}")
    print(f"Creating PLE config (model family: {model_family})")
    print(f"{'='*60}")

    ple_config = source_config.copy()

    # Change the model type identifiers (depends on the model family)
    ple_config['model_type'] = family_config['model_type']
    ple_config['architectures'] = family_config['architectures']

    # PLE routing config
    # think_token_id / no_think_token_id are set by register_control_tokens.py after the tokenizer is configured
    ple_config['think_token_id'] = None
    ple_config['no_think_token_id'] = None
    # Default to expert 0 (no_think), since at initialization the shared layers are fully compatible with expert 0
    ple_config['default_routing_index'] = 0

    # auto_map: supports loading the custom model with trust_remote_code
    ple_config['auto_map'] = family_config['auto_map']

    # Save the config
    os.makedirs(output_path, exist_ok=True)
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(ple_config, f, indent=2, ensure_ascii=False)

    print(f"  - model_type: {family_config['model_type']}")
    print(f"  - architectures: {family_config['architectures']}")
    print(f"  - default_routing_index: 0 (no_think mode)")
    print(f"  - Config saved to: {config_path}")

    return ple_config


# ==================== Auxiliary file copying ====================

def copy_auxiliary_files(source_model_path: str, output_path: str, model_family: str):
    """Copy auxiliary model files (tokenizer, model code, etc.)

    Args:
        source_model_path: source model path
        output_path: output directory path
        model_family: model family ("qwen2" or "qwen3")
    """
    family_config = MODEL_FAMILY_CONFIG[model_family]

    print(f"\n{'='*60}")
    print("Copying auxiliary model files")
    print(f"{'='*60}")

    os.makedirs(output_path, exist_ok=True)

    # 1. Copy the tokenizer and generation config files
    files_to_copy = [
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'vocab.json', 'merges.txt', 'added_tokens.json',
        'generation_config.json', 'chat_template.jinja',
    ]

    for filename in files_to_copy:
        src = os.path.join(source_model_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, filename))
            print(f"  ✅ Copied: {filename}")

    # 2. Copy the PLE model code files (for loading with trust_remote_code)
    ple_dir = os.path.join(project_root, "src", "models", family_config["model_code_dir"])
    model_code_files = [family_config["config_filename"], family_config["modeling_filename"]]

    for filename in model_code_files:
        src = os.path.join(ple_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, filename))
            print(f"  ✅ Copied model code: {filename}")
        else:
            print(f"  ⚠️ Model code file not found: {src}")


# ==================== safetensors shard helpers ====================

def get_shard_files(model_path: str) -> List[str]:
    """Get the list of safetensors shard files of a model"""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = json.load(f)
        return sorted(set(index.get("weight_map", {}).values()))

    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        return ["model.safetensors"]

    raise FileNotFoundError(f"No safetensors files found: {model_path}")


# ==================== Core: streaming weight processing ====================

def process_weights_streaming(
    source_model_path: str,
    output_path: str,
    num_layers: int,
    verbose: bool = True
) -> int:
    """Process weights in a streaming fashion (memory-optimized, single-source PLE only)

    Strategy:
    - Read the source model shards one by one
    - Non-MLP weights are copied directly (shared layers)
    - MLP weights are copied to both Expert 0 and Expert 1 (identical initialization)
    - Write output files according to the shard size limit

    Args:
        source_model_path: source model path (e.g. Qwen2.5-7B-Instruct)
        output_path: output path
        num_layers: number of model layers
        verbose: whether to print detailed logs

    Returns:
        copied_count: number of parameters copied
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    print(f"\n{'='*60}")
    print("Streaming weight processing (single source model, both experts initialized identically)")
    print(f"{'='*60}")

    source_shards = get_shard_files(source_model_path)
    print(f"\nSource model shards: {len(source_shards)}")

    # Use a temporary directory for intermediate results
    temp_dir = tempfile.mkdtemp(prefix="ple_weights_")
    print(f"Temporary directory: {temp_dir}")

    # Regex for matching MLP weights
    # gate_proj/up_proj/down_proj: Qwen2/Qwen3/LLaMA (3 MLP weights per layer)
    # gate_up_proj/down_proj: Phi3/Phi-4-mini (fused gate+up, 2 MLP weights per layer)
    mlp_pattern = re.compile(
        r'model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)\.weight'
    )

    # Output shard management
    weight_map = {}
    total_size = 0
    copied_count = 0
    output_shard_idx = 1

    # Shard size limit (5GB)
    SHARD_SIZE_LIMIT = 5 * 1024 * 1024 * 1024
    current_shard = {}
    current_shard_size = 0

    def save_current_shard():
        """Save the current shard to the temporary directory"""
        nonlocal output_shard_idx, current_shard, current_shard_size, weight_map, total_size
        if not current_shard:
            return

        shard_name = f"model-{output_shard_idx:05d}-of-PLACEHOLDER.safetensors"
        shard_path = os.path.join(temp_dir, shard_name)
        save_file(current_shard, shard_path)

        for k in current_shard:
            weight_map[k] = shard_name

        total_size += current_shard_size
        print(f"    💾 Saving shard {output_shard_idx}: "
              f"{current_shard_size / 1e9:.2f} GB, {len(current_shard)} parameters")

        output_shard_idx += 1
        current_shard = {}
        current_shard_size = 0
        gc.collect()

    def add_weight(key: str, tensor: torch.Tensor):
        """Add a weight to the current shard (shards automatically)"""
        nonlocal current_shard, current_shard_size
        tensor_size = tensor.numel() * tensor.element_size()

        if current_shard_size + tensor_size > SHARD_SIZE_LIMIT and current_shard:
            save_current_shard()

        current_shard[key] = tensor
        current_shard_size += tensor_size

    # ========== Process the source model weights ==========
    print(f"\nProcessing source model weights...")
    print(f"  - Non-MLP weights: copied directly (shared layers)")
    print(f"  - MLP weights: copied to both Expert 0 and Expert 1 (identical initialization)")

    for shard_idx, shard_file in enumerate(source_shards):
        print(f"\n  📂 Shard {shard_idx + 1}/{len(source_shards)}: {shard_file}")
        shard_path = os.path.join(source_model_path, shard_file)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                tensor = f.get_tensor(key)

                mlp_match = mlp_pattern.match(key)

                if mlp_match:
                    # MLP weights -> copied to both Expert 0 and Expert 1
                    layer_idx = mlp_match.group(1)
                    proj_type = mlp_match.group(2)

                    key_e0 = f'model.layers.{layer_idx}.mlp.experts.0.{proj_type}.weight'
                    key_e1 = f'model.layers.{layer_idx}.mlp.experts.1.{proj_type}.weight'

                    add_weight(key_e0, tensor)
                    add_weight(key_e1, tensor.clone())  # clone to avoid shared storage
                    copied_count += 2

                    if verbose:
                        print(f"    [E0+E1] {key} → experts.0 + experts.1  shape={list(tensor.shape)}")
                else:
                    # Non-MLP weights: copy directly
                    add_weight(key, tensor)
                    copied_count += 1

                    if verbose:
                        print(f"    [CP] {key}  shape={list(tensor.shape)}")

        gc.collect()

    # Save the last shard
    save_current_shard()

    print(f"\n  ✅ Weight processing done: {copied_count} parameter tensors in total")

    # ========== Move files to the output directory ==========
    print(f"\n{'='*60}")
    print("Moving files to the output directory")
    print(f"{'='*60}")

    total_shards = output_shard_idx - 1

    for i in range(1, total_shards + 1):
        old_name = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"

        old_path = os.path.join(temp_dir, old_name)
        new_path = os.path.join(output_path, new_name)

        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            file_size = os.path.getsize(new_path)
            print(f"  ✅ {new_name} ({file_size / 1e9:.2f} GB)")

        for k, v in list(weight_map.items()):
            if v == old_name:
                weight_map[k] = new_name

    # Save the index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }

    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"  ✅ model.safetensors.index.json")

    shutil.rmtree(temp_dir, ignore_errors=True)

    return copied_count


# ==================== Verification ====================

def verify_output(output_path: str, num_layers: int, model_family: str):
    """Verify the basic integrity of the output model

    Args:
        output_path: output model path
        num_layers: number of model layers
        model_family: model family ("qwen2" or "qwen3")
    """
    family_config = MODEL_FAMILY_CONFIG[model_family]

    print(f"\n{'='*60}")
    print(f"Verifying output model integrity (model family: {model_family})")
    print(f"{'='*60}")

    required_files = ['config.json', 'model.safetensors.index.json']
    for f in required_files:
        path = os.path.join(output_path, f)
        assert os.path.exists(path), f"Missing file: {f}"
        print(f"  ✅ {f} exists")

    with open(os.path.join(output_path, "model.safetensors.index.json"), 'r') as f:
        index = json.load(f)

    weight_keys = sorted(index['weight_map'].keys())

    # Per-layer weight count depends on the model family:
    # - qwen2: 15 (attn 7 + layernorm 2 + expert MLP 6)
    # - qwen3: 14 (attn 4 + q/k_norm 2 + layernorm 2 + expert MLP 6)
    # Global weights: embed_tokens + norm = 2 (+1 lm_head if tie_word_embeddings=false)
    expected_per_layer = family_config["expected_per_layer"]
    has_lm_head = any('lm_head' in k for k in weight_keys)
    num_global = 3 if has_lm_head else 2
    expected_total = expected_per_layer * num_layers + num_global

    print(f"  - Number of weights: {len(weight_keys)} (expected: {expected_total})")
    print(f"  - Expected per layer: {expected_per_layer} weights")
    print(f"  - Global weights: {num_global} ({'with lm_head' if has_lm_head else 'tie_word_embeddings, no lm_head'})")

    # Check that there are no gate weights
    gate_keys = [k for k in weight_keys if 'gate.weight' in k and 'gate_proj' not in k]
    if gate_keys:
        print(f"  ❌ Unexpected gate weights found: {gate_keys}")
    else:
        print(f"  ✅ No gate/router weights (matches the PLE design)")

    # Check that the expert structure is complete
    expert0_keys = [k for k in weight_keys if '.experts.0.' in k]
    expert1_keys = [k for k in weight_keys if '.experts.1.' in k]
    # MLP weights per layer: Qwen/LLaMA = 3 (gate_proj, up_proj, down_proj), Phi = 2 (gate_up_proj, down_proj)
    mlp_weights_per_layer = (expected_per_layer - len([
        k for k in weight_keys
        if 'layers.0.' in k and '.experts.' not in k
    ])) // 2  # divide by 2 because there are two experts
    expected_expert_keys = num_layers * mlp_weights_per_layer

    print(f"  - Expert 0 weights: {len(expert0_keys)} (expected: {expected_expert_keys})")
    print(f"  - Expert 1 weights: {len(expert1_keys)} (expected: {expected_expert_keys})")

    assert len(expert0_keys) == expected_expert_keys
    assert len(expert1_keys) == expected_expert_keys
    assert len(weight_keys) == expected_total

    print(f"\n✅ Output model verification passed")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Copy source-model weights into a PLE model (both experts initialized identically, supports Qwen2/Qwen3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Qwen2 (model family auto-detected)
    python -m src.utils.copy_weights_to_ple \\
        --source_model_path ./models/source/Qwen2.5-7B-Instruct \\
        --output_path ./models/ple_initialized

    # Qwen3 (model family auto-detected)
    python -m src.utils.copy_weights_to_ple \\
        --source_model_path ./models/source/Qwen3-4B-Instruct-2507 \\
        --output_path ./models/qwen3_4b_instruct_ple_initialized

    # Specify the model family manually (optional)
    python -m src.utils.copy_weights_to_ple \\
        --source_model_path ./models/source/Qwen3-4B-Instruct-2507 \\
        --output_path ./models/qwen3_4b_instruct_ple_initialized \\
        --model_family qwen3

Both experts are initialized from the same source model and diverge naturally during SFT.
        """
    )

    parser.add_argument(
        "--source_model_path", type=str, required=True,
        help="Source model path (used for shared layers + Expert 0/1 MLP), e.g. Qwen2.5-7B-Instruct or Qwen3-4B-Instruct-2507"
    )
    parser.add_argument(
        "--output_path", type=str, default="./models/ple_initialized",
        help="Output model path (default: ./models/ple_initialized)"
    )
    parser.add_argument(
        "--model_family", type=str, default=None, choices=["qwen2", "qwen3", "llama"],
        help="Model family (default: auto-detected from model_type in the source config.json)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print detailed logs (copy info for every weight)"
    )

    args = parser.parse_args()

    # Step 1: read the source model config
    config_path = os.path.join(args.source_model_path, "config.json")
    with open(config_path, 'r') as f:
        source_config = json.load(f)

    # Step 1.5: detect or validate the model family
    if args.model_family is not None:
        model_family = args.model_family
    else:
        model_family = detect_model_family(source_config)

    print("\n" + "="*60)
    print(f"PLE weight-copy tool (model family: {model_family})")
    print("="*60)
    print(f"\nSource model path: {args.source_model_path}")
    print(f"Output path:       {args.output_path}")
    print(f"Model family:      {model_family}")
    print(f"\nStrategy: Expert 0 and Expert 1 are both initialized from the same source model")

    num_layers = source_config['num_hidden_layers']
    print(f"\nSource model parameters:")
    for p in ['model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
              'num_attention_heads', 'vocab_size']:
        print(f"  {p}: {source_config.get(p)}")

    # Step 2: remove old safetensors files from the output directory
    if os.path.exists(args.output_path):
        for f in os.listdir(args.output_path):
            if f.endswith('.safetensors'):
                os.remove(os.path.join(args.output_path, f))
                print(f"  🗑️ Removing old shard: {f}")

    # Step 3: create the PLE config
    create_ple_config(source_config, args.output_path, model_family)

    # Step 4: copy auxiliary files
    copy_auxiliary_files(args.source_model_path, args.output_path, model_family)

    # Step 5: process the weights in a streaming fashion
    copied_count = process_weights_streaming(
        args.source_model_path,
        args.output_path,
        num_layers,
        args.verbose
    )

    # Step 6: verify the output
    verify_output(args.output_path, num_layers, model_family)

    # Step 7: print statistics
    family_config = MODEL_FAMILY_CONFIG[model_family]
    print(f"\n{'='*60}")
    print("PLE weight copy done!")
    print(f"{'='*60}")
    print(f"  - Model family: {model_family}")
    print(f"  - Source model: {args.source_model_path}")
    print(f"  - PLE type: {family_config['model_type']}")
    print(f"  - Expert 0 (no_think): from the source model MLP")
    print(f"  - Expert 1 (think):    from the source model MLP (same as Expert 0)")
    print(f"  - Shared layers:       from the source model (embed, attn, norm, lm_head)")
    print(f"  - Gate/Router:         none (PLE routing is decided by the control tokens)")
    print(f"  - Total:               {copied_count} weight tensors")

    print(f"\nNext steps:")
    print(f"  1. Register control tokens: python -m src.utils.register_control_tokens --model_path {args.output_path}")
    print(f"  2. Run tests to verify weight consistency")


if __name__ == "__main__":
    main()
