#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将源模型权重复制到 PLE 模型（内存优化版本，支持 Qwen2/Qwen3）

采用流式处理 safetensors 的策略，逐分片读取和写入，避免同时加载完整模型。
峰值内存控制在单个分片大小（约 5GB），适合在 RAM 有限的环境下运行。

权重复制策略（PLE，无 gate/router 权重）：
- 从单一源模型复制所有权重：
  - Embedding、Attention、LayerNorm、lm_head → 直接复制（共享层）
  - MLP 权重 (gate_proj, up_proj, down_proj) → 同时复制到 Expert 0 和 Expert 1
- 两个 Expert 初始化相同，SFT 过程中会自然分化。
- PLE 无 gate/router 权重，路由完全由控制 token 硬决定。

支持的模型族：
- qwen2: Qwen2.5-7B-Instruct 等（attention 有 bias，无 q/k norm）
- qwen3: Qwen3-4B-Instruct 等（attention 无 bias，有 q/k norm）

使用方法:
    python -m src.utils.copy_weights_to_ple \\
        --source_model_path ./models/source/Qwen2.5-7B-Instruct \\
        --output_path ./models/ple_initialized

    python -m src.utils.copy_weights_to_ple \\
        --source_model_path ./models/source/Qwen3-4B-Instruct-2507 \\
        --output_path ./models/qwen3_4b_instruct_ple_initialized

依赖:
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

# 项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== 模型族配置映射 ====================

# 各模型族的 PLE 配置参数
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
        # Qwen2 每层权重数（PLE 化后）：
        # self_attn: q.w, q.b, k.w, k.b, v.w, v.b, o.w = 7
        # layernorm: input_layernorm.w, post_attention_layernorm.w = 2
        # MLP experts: 3 weights × 2 experts = 6
        # 总计 = 7 + 2 + 6 = 15
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
        # Qwen3 每层权重数（PLE 化后）：
        # self_attn: q.w, k.w, v.w, o.w = 4（无 bias，attention_bias=False）
        # self_attn: q_norm.w, k_norm.w = 2（QK normalization，Qwen3 特有）
        # layernorm: input_layernorm.w, post_attention_layernorm.w = 2
        # MLP experts: 3 weights × 2 experts = 6
        # 总计 = 4 + 2 + 2 + 6 = 14
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
        # LLaMA 每层权重数（PLE 化后）：
        # self_attn: q.w, k.w, v.w, o.w = 4（无 bias，无 q/k norm）
        # layernorm: input_layernorm.w, post_attention_layernorm.w = 2
        # MLP experts: 3 weights × 2 experts = 6
        # 总计 = 4 + 2 + 6 = 12
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
        # Phi3/Phi-4-mini 每层权重数（PLE 化后）：
        # self_attn: qkv_proj.w, o_proj.w = 2（融合 QKV，无 bias）
        # layernorm: input_layernorm.w, post_attention_layernorm.w = 2
        # MLP experts: 2 weights (gate_up_proj, down_proj) × 2 experts = 4
        # 总计 = 2 + 2 + 4 = 8
        "expected_per_layer": 8,
    },
}


def detect_model_family(source_config: dict) -> str:
    """从源模型配置自动检测模型族

    Args:
        source_config: 源模型 config.json 内容

    Returns:
        模型族名称: "qwen2" 或 "qwen3"

    Raises:
        ValueError: 不支持的模型类型
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
            f"不支持的模型类型: {model_type}。"
            f"当前支持: qwen2, qwen3, llama, phi3。"
            f"可通过 --model_family 参数手动指定。"
        )


# ==================== PLE 配置生成 ====================

def create_ple_config(source_config: dict, output_path: str, model_family: str) -> dict:
    """基于源模型配置创建 PLE 配置

    PLE 配置继承源模型的全部基础参数（hidden_size、attention 参数等），
    并修改 model_type、architectures，以及添加路由相关配置。

    Args:
        source_config: 源模型的 config.json 内容
        output_path: 输出目录路径
        model_family: 模型族（"qwen2" 或 "qwen3"）

    Returns:
        PLE 配置字典
    """
    family_config = MODEL_FAMILY_CONFIG[model_family]

    print(f"\n{'='*60}")
    print(f"创建 PLE 配置（模型族: {model_family}）")
    print(f"{'='*60}")

    ple_config = source_config.copy()

    # 修改模型类型标识（根据模型族）
    ple_config['model_type'] = family_config['model_type']
    ple_config['architectures'] = family_config['architectures']

    # PLE 路由配置
    # think_token_id / no_think_token_id 在 tokenizer 配置完成后由 register_control_tokens.py 设置
    ple_config['think_token_id'] = None
    ple_config['no_think_token_id'] = None
    # 默认走 expert 0（no_think），因为初始化阶段共享层与 expert 0 完全兼容
    ple_config['default_routing_index'] = 0

    # auto_map: 支持 trust_remote_code 加载自定义模型
    ple_config['auto_map'] = family_config['auto_map']

    # 保存配置
    os.makedirs(output_path, exist_ok=True)
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(ple_config, f, indent=2, ensure_ascii=False)

    print(f"  - model_type: {family_config['model_type']}")
    print(f"  - architectures: {family_config['architectures']}")
    print(f"  - default_routing_index: 0 (no_think 模式)")
    print(f"  - 配置已保存到: {config_path}")

    return ple_config


# ==================== 辅助文件复制 ====================

def copy_auxiliary_files(source_model_path: str, output_path: str, model_family: str):
    """复制模型辅助文件（tokenizer、模型代码等）

    Args:
        source_model_path: 源模型路径
        output_path: 输出目录路径
        model_family: 模型族（"qwen2" 或 "qwen3"）
    """
    family_config = MODEL_FAMILY_CONFIG[model_family]

    print(f"\n{'='*60}")
    print("复制模型辅助文件")
    print(f"{'='*60}")

    os.makedirs(output_path, exist_ok=True)

    # 1. 复制 tokenizer 和生成配置文件
    files_to_copy = [
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'vocab.json', 'merges.txt', 'added_tokens.json',
        'generation_config.json', 'chat_template.jinja',
    ]

    for filename in files_to_copy:
        src = os.path.join(source_model_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, filename))
            print(f"  ✅ 复制: {filename}")

    # 2. 复制 PLE 模型代码文件（用于 trust_remote_code 加载）
    ple_dir = os.path.join(project_root, "src", "models", family_config["model_code_dir"])
    model_code_files = [family_config["config_filename"], family_config["modeling_filename"]]

    for filename in model_code_files:
        src = os.path.join(ple_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, filename))
            print(f"  ✅ 复制模型代码: {filename}")
        else:
            print(f"  ⚠️ 模型代码文件不存在: {src}")


# ==================== safetensors 分片工具 ====================

def get_shard_files(model_path: str) -> List[str]:
    """获取模型的 safetensors 分片文件列表"""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = json.load(f)
        return sorted(set(index.get("weight_map", {}).values()))

    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        return ["model.safetensors"]

    raise FileNotFoundError(f"未找到 safetensors 文件: {model_path}")


# ==================== 核心：流式权重处理 ====================

def process_weights_streaming(
    source_model_path: str,
    output_path: str,
    num_layers: int,
    verbose: bool = True
) -> int:
    """流式处理权重（内存优化版本，单源模型 PLE 专用）

    策略：
    - 逐个读取源模型分片
    - 非 MLP 权重直接复制（共享层）
    - MLP 权重同时复制到 Expert 0 和 Expert 1（相同初始化）
    - 按分片大小限制写入输出文件

    Args:
        source_model_path: 源模型路径（Qwen2.5-7B-Instruct）
        output_path: 输出路径
        num_layers: 模型层数
        verbose: 是否输出详细日志

    Returns:
        copied_count: 复制的参数数量
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    print(f"\n{'='*60}")
    print("流式处理权重（单源模型，两个 Expert 相同初始化）")
    print(f"{'='*60}")

    source_shards = get_shard_files(source_model_path)
    print(f"\n源模型分片: {len(source_shards)} 个")

    # 使用临时目录存储中间结果
    temp_dir = tempfile.mkdtemp(prefix="ple_weights_")
    print(f"临时目录: {temp_dir}")

    # MLP 权重匹配正则
    # gate_proj/up_proj/down_proj: Qwen2/Qwen3/LLaMA（3 个 MLP 权重/层）
    # gate_up_proj/down_proj: Phi3/Phi-4-mini（融合 gate+up，2 个 MLP 权重/层）
    mlp_pattern = re.compile(
        r'model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)\.weight'
    )

    # 输出分片管理
    weight_map = {}
    total_size = 0
    copied_count = 0
    output_shard_idx = 1

    # 分片大小限制 (5GB)
    SHARD_SIZE_LIMIT = 5 * 1024 * 1024 * 1024
    current_shard = {}
    current_shard_size = 0

    def save_current_shard():
        """保存当前分片到临时目录"""
        nonlocal output_shard_idx, current_shard, current_shard_size, weight_map, total_size
        if not current_shard:
            return

        shard_name = f"model-{output_shard_idx:05d}-of-PLACEHOLDER.safetensors"
        shard_path = os.path.join(temp_dir, shard_name)
        save_file(current_shard, shard_path)

        for k in current_shard:
            weight_map[k] = shard_name

        total_size += current_shard_size
        print(f"    💾 保存分片 {output_shard_idx}: "
              f"{current_shard_size / 1e9:.2f} GB, {len(current_shard)} 个参数")

        output_shard_idx += 1
        current_shard = {}
        current_shard_size = 0
        gc.collect()

    def add_weight(key: str, tensor: torch.Tensor):
        """添加权重到当前分片（自动分片）"""
        nonlocal current_shard, current_shard_size
        tensor_size = tensor.numel() * tensor.element_size()

        if current_shard_size + tensor_size > SHARD_SIZE_LIMIT and current_shard:
            save_current_shard()

        current_shard[key] = tensor
        current_shard_size += tensor_size

    # ========== 处理源模型权重 ==========
    print(f"\n处理源模型权重...")
    print(f"  - 非 MLP 权重: 直接复制（共享层）")
    print(f"  - MLP 权重: 同时复制到 Expert 0 和 Expert 1（相同初始化）")

    for shard_idx, shard_file in enumerate(source_shards):
        print(f"\n  📂 分片 {shard_idx + 1}/{len(source_shards)}: {shard_file}")
        shard_path = os.path.join(source_model_path, shard_file)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                tensor = f.get_tensor(key)

                mlp_match = mlp_pattern.match(key)

                if mlp_match:
                    # MLP 权重 → 同时复制到 Expert 0 和 Expert 1
                    layer_idx = mlp_match.group(1)
                    proj_type = mlp_match.group(2)

                    key_e0 = f'model.layers.{layer_idx}.mlp.experts.0.{proj_type}.weight'
                    key_e1 = f'model.layers.{layer_idx}.mlp.experts.1.{proj_type}.weight'

                    add_weight(key_e0, tensor)
                    add_weight(key_e1, tensor.clone())  # clone 避免共享存储
                    copied_count += 2

                    if verbose:
                        print(f"    [E0+E1] {key} → experts.0 + experts.1  shape={list(tensor.shape)}")
                else:
                    # 非 MLP 权重：直接复制
                    add_weight(key, tensor)
                    copied_count += 1

                    if verbose:
                        print(f"    [CP] {key}  shape={list(tensor.shape)}")

        gc.collect()

    # 保存最后一个分片
    save_current_shard()

    print(f"\n  ✅ 权重处理完成: 共 {copied_count} 个参数张量")

    # ========== 移动文件到输出目录 ==========
    print(f"\n{'='*60}")
    print("移动文件到输出目录")
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

    # 保存索引文件
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


# ==================== 验证函数 ====================

def verify_output(output_path: str, num_layers: int, model_family: str):
    """验证输出模型的基本完整性

    Args:
        output_path: 输出模型路径
        num_layers: 模型层数
        model_family: 模型族（"qwen2" 或 "qwen3"）
    """
    family_config = MODEL_FAMILY_CONFIG[model_family]

    print(f"\n{'='*60}")
    print(f"验证输出模型完整性（模型族: {model_family}）")
    print(f"{'='*60}")

    required_files = ['config.json', 'model.safetensors.index.json']
    for f in required_files:
        path = os.path.join(output_path, f)
        assert os.path.exists(path), f"缺少文件: {f}"
        print(f"  ✅ {f} 存在")

    with open(os.path.join(output_path, "model.safetensors.index.json"), 'r') as f:
        index = json.load(f)

    weight_keys = sorted(index['weight_map'].keys())

    # 每层权重数根据模型族不同：
    # - qwen2: 15（attn 7 + layernorm 2 + expert MLP 6）
    # - qwen3: 14（attn 4 + q/k_norm 2 + layernorm 2 + expert MLP 6）
    # 全局权重: embed_tokens + norm = 2（若 tie_word_embeddings=false 则 +1 lm_head）
    expected_per_layer = family_config["expected_per_layer"]
    has_lm_head = any('lm_head' in k for k in weight_keys)
    num_global = 3 if has_lm_head else 2
    expected_total = expected_per_layer * num_layers + num_global

    print(f"  - 权重数量: {len(weight_keys)} (预期: {expected_total})")
    print(f"  - 每层预期: {expected_per_layer} 个权重")
    print(f"  - 全局权重: {num_global} 个 ({'含 lm_head' if has_lm_head else 'tie_word_embeddings, 无 lm_head'})")

    # 检查无 gate 权重
    gate_keys = [k for k in weight_keys if 'gate.weight' in k and 'gate_proj' not in k]
    if gate_keys:
        print(f"  ❌ 发现异常 gate 权重: {gate_keys}")
    else:
        print(f"  ✅ 无 gate/router 权重（符合 PLE 设计）")

    # 检查 expert 结构完整
    expert0_keys = [k for k in weight_keys if '.experts.0.' in k]
    expert1_keys = [k for k in weight_keys if '.experts.1.' in k]
    # MLP 权重数/层: Qwen/LLaMA = 3 (gate_proj, up_proj, down_proj), Phi = 2 (gate_up_proj, down_proj)
    mlp_weights_per_layer = (expected_per_layer - len([
        k for k in weight_keys
        if 'layers.0.' in k and '.experts.' not in k
    ])) // 2  # 除以 2 因为有两个 expert
    expected_expert_keys = num_layers * mlp_weights_per_layer

    print(f"  - Expert 0 权重数: {len(expert0_keys)} (预期: {expected_expert_keys})")
    print(f"  - Expert 1 权重数: {len(expert1_keys)} (预期: {expected_expert_keys})")

    assert len(expert0_keys) == expected_expert_keys
    assert len(expert1_keys) == expected_expert_keys
    assert len(weight_keys) == expected_total

    print(f"\n✅ 输出模型验证通过")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="将源模型权重复制到 PLE 模型（两个 Expert 相同初始化，支持 Qwen2/Qwen3）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # Qwen2（自动检测模型族）
    python -m src.utils.copy_weights_to_ple \\
        --source_model_path ./models/source/Qwen2.5-7B-Instruct \\
        --output_path ./models/ple_initialized

    # Qwen3（自动检测模型族）
    python -m src.utils.copy_weights_to_ple \\
        --source_model_path ./models/source/Qwen3-4B-Instruct-2507 \\
        --output_path ./models/qwen3_4b_instruct_ple_initialized

    # 手动指定模型族（可选）
    python -m src.utils.copy_weights_to_ple \\
        --source_model_path ./models/source/Qwen3-4B-Instruct-2507 \\
        --output_path ./models/qwen3_4b_instruct_ple_initialized \\
        --model_family qwen3

两个 Expert 使用相同的源模型初始化，SFT 过程中会自然分化。
        """
    )

    parser.add_argument(
        "--source_model_path", type=str, required=True,
        help="源模型路径（用于共享层 + Expert 0/1 MLP），如 Qwen2.5-7B-Instruct 或 Qwen3-4B-Instruct-2507"
    )
    parser.add_argument(
        "--output_path", type=str, default="./models/ple_initialized",
        help="输出模型路径 (默认: ./models/ple_initialized)"
    )
    parser.add_argument(
        "--model_family", type=str, default=None, choices=["qwen2", "qwen3", "llama"],
        help="模型族（默认: 从源模型 config.json 的 model_type 自动检测）"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="输出详细日志（每个权重的复制信息）"
    )

    args = parser.parse_args()

    # 步骤 1: 读取源模型配置
    config_path = os.path.join(args.source_model_path, "config.json")
    with open(config_path, 'r') as f:
        source_config = json.load(f)

    # 步骤 1.5: 检测或验证模型族
    if args.model_family is not None:
        model_family = args.model_family
    else:
        model_family = detect_model_family(source_config)

    print("\n" + "="*60)
    print(f"PLE 权重复制工具（模型族: {model_family}）")
    print("="*60)
    print(f"\n源模型路径: {args.source_model_path}")
    print(f"输出路径:   {args.output_path}")
    print(f"模型族:     {model_family}")
    print(f"\n策略: Expert 0 和 Expert 1 均从同一源模型初始化")

    num_layers = source_config['num_hidden_layers']
    print(f"\n源模型参数:")
    for p in ['model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
              'num_attention_heads', 'vocab_size']:
        print(f"  {p}: {source_config.get(p)}")

    # 步骤 2: 清理旧的输出目录中的 safetensors 文件
    if os.path.exists(args.output_path):
        for f in os.listdir(args.output_path):
            if f.endswith('.safetensors'):
                os.remove(os.path.join(args.output_path, f))
                print(f"  🗑️ 删除旧分片: {f}")

    # 步骤 3: 创建 PLE 配置
    create_ple_config(source_config, args.output_path, model_family)

    # 步骤 4: 复制辅助文件
    copy_auxiliary_files(args.source_model_path, args.output_path, model_family)

    # 步骤 5: 流式处理权重
    copied_count = process_weights_streaming(
        args.source_model_path,
        args.output_path,
        num_layers,
        args.verbose
    )

    # 步骤 6: 验证输出
    verify_output(args.output_path, num_layers, model_family)

    # 步骤 7: 输出统计
    family_config = MODEL_FAMILY_CONFIG[model_family]
    print(f"\n{'='*60}")
    print("PLE 权重复制完成！")
    print(f"{'='*60}")
    print(f"  - 模型族: {model_family}")
    print(f"  - 源模型: {args.source_model_path}")
    print(f"  - PLE 类型: {family_config['model_type']}")
    print(f"  - Expert 0 (no_think): 来自源模型 MLP")
    print(f"  - Expert 1 (think):    来自源模型 MLP（与 Expert 0 相同）")
    print(f"  - 共享层:              来自源模型 (embed, attn, norm, lm_head)")
    print(f"  - Gate/Router:         无 (PLE 路由由控制 token 决定)")
    print(f"  - 总计:                {copied_count} 个权重张量")

    print(f"\n下一步：")
    print(f"  1. 注册控制 token: python -m src.utils.register_control_tokens --model_path {args.output_path}")
    print(f"  2. 运行测试验证权重一致性")


if __name__ == "__main__":
    main()
