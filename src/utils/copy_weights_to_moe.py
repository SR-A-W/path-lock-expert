#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将源模型权重复制到 TL-MoE 模型（内存优化版本）

采用流式处理 safetensors 的策略，逐分片读取和写入，避免同时加载多个完整模型。
峰值内存控制在单个分片大小（约 5GB），适合在 RAM 有限的环境下运行。

权重复制策略：
1. 从主模型（Qwen2.5-7B-Instruct）复制：
   - Embedding、Attention、LayerNorm、lm_head → 直接复制
   - MLP 权重 (gate_proj, up_proj, down_proj) → Expert 0 (no_think)

2. 从 MLP 权重模型（DeepSeek-R1-Distill-Qwen-7B）复制：
   - 仅 MLP 权重 → Expert 1 (think)

3. 新初始化：Router (gate) 权重使用正态分布 (std=0.02)

使用方法:
    conda activate het
    python -m src.utils.copy_weights_to_moe \\
        --main_model_path ./models/source/Qwen2.5-7B-Instruct \\
        --mlp_weight_model_path ./models/source/DeepSeek-R1-Distill-Qwen-7B \\
        --output_path ./models/tl_moe_initialized

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
from typing import Dict, List, Optional, Tuple

import torch

# 项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== 架构兼容性检查 ====================

def check_model_compatibility(main_model_path: str, mlp_model_path: str) -> Tuple[dict, int]:
    """检查两个源模型的架构兼容性
    
    两个模型必须具有相同的 hidden_size、intermediate_size、num_hidden_layers 等关键参数，
    否则权重无法正确复制。
    
    Args:
        main_model_path: 主模型路径（Qwen2.5-7B-Instruct）
        mlp_model_path: MLP 权重模型路径（DeepSeek-R1-Distill-Qwen-7B）
        
    Returns:
        (main_config_dict, num_layers): 主模型配置字典和层数
    
    Raises:
        ValueError: 如果两个模型架构不兼容
    """
    print(f"\n{'='*60}")
    print("检查模型架构兼容性")
    print(f"{'='*60}")
    
    with open(os.path.join(main_model_path, "config.json"), 'r') as f:
        main_config = json.load(f)
    
    with open(os.path.join(mlp_model_path, "config.json"), 'r') as f:
        mlp_config = json.load(f)
    
    # 关键参数必须一致
    critical_params = ['hidden_size', 'intermediate_size', 'num_hidden_layers', 
                       'num_attention_heads', 'vocab_size']
    
    incompatible = []
    for param in critical_params:
        main_value = main_config.get(param)
        mlp_value = mlp_config.get(param)
        if main_value != mlp_value:
            incompatible.append(f"  - {param}: 主模型={main_value}, MLP模型={mlp_value}")
            print(f"  ❌ {param}: 主模型={main_value}, MLP模型={mlp_value}")
        else:
            print(f"  ✅ {param}: {main_value}")
    
    if incompatible:
        raise ValueError("模型架构不兼容：\n" + "\n".join(incompatible))
    
    print(f"\n✅ 模型架构兼容性检查通过")
    return main_config, main_config['num_hidden_layers']


# ==================== TL-MoE 配置生成 ====================

def create_tl_moe_config(main_config: dict, output_path: str) -> dict:
    """基于主模型配置创建 TL-MoE 配置
    
    将主模型的基础配置扩展为 TL-MoE 的 MoE 配置，添加 MoE 相关参数。
    同时设置 auto_map 以支持 trust_remote_code 加载。
    
    Args:
        main_config: 主模型的 config.json 内容
        output_path: 输出目录路径
        
    Returns:
        TL-MoE 配置字典
    """
    print(f"\n{'='*60}")
    print("创建 TL-MoE 配置")
    print(f"{'='*60}")
    
    tl_moe_config = main_config.copy()
    
    # 修改模型类型标识
    tl_moe_config['model_type'] = 'qwen2_tl_moe'
    tl_moe_config['architectures'] = ['Qwen2TLMoeForCausalLM']
    
    # TL-MoE 核心参数
    tl_moe_config['decoder_sparse_step'] = 1  # 每层都是 MoE
    tl_moe_config['moe_intermediate_size'] = main_config['intermediate_size']  # 与 dense 模型一致
    tl_moe_config['shared_expert_intermediate_size'] = 0  # 移除 shared_expert
    tl_moe_config['num_experts_per_tok'] = 1  # 每个 token 选 1 个专家
    tl_moe_config['num_experts'] = 2  # 2 个专家
    tl_moe_config['norm_topk_prob'] = False
    tl_moe_config['output_router_logits'] = False
    tl_moe_config['router_aux_loss_coef'] = 0.001
    tl_moe_config['mlp_only_layers'] = []
    
    # auto_map: 支持 trust_remote_code 加载自定义模型
    tl_moe_config['auto_map'] = {
        "AutoConfig": "configuration_qwen2_tl_moe.Qwen2TLMoeConfig",
        "AutoModelForCausalLM": "modeling_qwen2_tl_moe.Qwen2TLMoeForCausalLM"
    }
    
    # 保存配置
    os.makedirs(output_path, exist_ok=True)
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(tl_moe_config, f, indent=2)
    
    print(f"  - model_type: qwen2_tl_moe")
    print(f"  - num_experts: 2, num_experts_per_tok: 1")
    print(f"  - moe_intermediate_size: {tl_moe_config['moe_intermediate_size']}")
    print(f"  - decoder_sparse_step: 1 (每层都是 MoE)")
    print(f"  - 配置已保存到: {config_path}")
    
    return tl_moe_config


# ==================== 辅助文件复制 ====================

def copy_model_files(main_model_path: str, output_path: str):
    """复制模型辅助文件（tokenizer、模型代码等）
    
    Args:
        main_model_path: 主模型路径
        output_path: 输出目录路径
    """
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
        src = os.path.join(main_model_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, filename))
            print(f"  ✅ 复制: {filename}")
    
    # 2. 复制 TL-MoE 模型代码文件（用于 trust_remote_code 加载）
    tl_moe_dir = os.path.join(project_root, "src", "models", "qwen2_tl_moe")
    model_code_files = ['configuration_qwen2_tl_moe.py', 'modeling_qwen2_tl_moe.py']
    
    for filename in model_code_files:
        src = os.path.join(tl_moe_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, filename))
            print(f"  ✅ 复制模型代码: {filename}")
        else:
            print(f"  ⚠️ 模型代码文件不存在: {src}")


# ==================== safetensors 分片工具 ====================

def get_shard_files(model_path: str) -> List[str]:
    """获取模型的 safetensors 分片文件列表
    
    Args:
        model_path: 模型路径
        
    Returns:
        分片文件名列表（已排序去重）
    
    Raises:
        FileNotFoundError: 如果未找到 safetensors 文件
    """
    # 优先检查分片索引文件
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = json.load(f)
        return sorted(set(index.get("weight_map", {}).values()))
    
    # 回退到单文件
    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        return ["model.safetensors"]
    
    raise FileNotFoundError(f"未找到 safetensors 文件: {model_path}")


def get_weight_map(model_path: str) -> Dict[str, str]:
    """获取模型的权重名 → 分片文件映射
    
    Args:
        model_path: 模型路径
        
    Returns:
        {权重名: 分片文件名} 的字典
    """
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = json.load(f)
        return index.get("weight_map", {})
    return {}


def load_tensor(model_path: str, key: str, weight_map: Optional[Dict[str, str]] = None) -> torch.Tensor:
    """从 safetensors 加载单个张量
    
    Args:
        model_path: 模型路径
        key: 权重名
        weight_map: 可选的权重映射字典（避免重复读取索引文件）
        
    Returns:
        权重张量
    
    Raises:
        KeyError: 如果权重不存在
    """
    from safetensors import safe_open
    
    if weight_map is None:
        weight_map = get_weight_map(model_path)
    
    if key in weight_map:
        shard_file = weight_map[key]
    else:
        shard_file = "model.safetensors"
    
    shard_path = os.path.join(model_path, shard_file)
    
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"分片文件不存在: {shard_path}")
    
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if key in f.keys():
            return f.get_tensor(key)
    
    raise KeyError(f"权重不存在: {key} (in {shard_path})")


# ==================== 核心：流式权重处理 ====================

def process_weights_streaming(
    main_model_path: str,
    mlp_model_path: str,
    output_path: str,
    num_layers: int,
    hidden_size: int,
    verbose: bool = True
) -> Tuple[int, int, int]:
    """流式处理权重（内存优化版本）
    
    策略：
    1. 逐个读取主模型分片，将非 MLP 权重直接复制，MLP 权重映射到 Expert 0
    2. 逐个读取 MLP 模型分片，将 MLP 权重映射到 Expert 1
    3. 生成 gate 权重
    4. 按分片大小限制写入输出文件
    
    内存优化关键：
    - 不使用 from_pretrained 加载完整模型
    - 使用 safetensors 的 safe_open 按需加载单个张量
    - 处理完一个分片后立即释放内存
    
    Args:
        main_model_path: 主模型路径
        mlp_model_path: MLP 权重模型路径
        output_path: 输出路径
        num_layers: 模型层数
        hidden_size: 隐藏维度大小
        verbose: 是否输出详细日志
        
    Returns:
        (copied_main, copied_mlp, gate_count): 各阶段复制的参数数量
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    
    print(f"\n{'='*60}")
    print("流式处理权重（内存优化）")
    print(f"{'='*60}")
    
    main_shards = get_shard_files(main_model_path)
    mlp_shards = get_shard_files(mlp_model_path)
    
    print(f"\n主模型分片: {len(main_shards)} 个")
    print(f"MLP 模型分片: {len(mlp_shards)} 个")
    
    # 使用临时目录存储中间结果，最后移动到输出目录
    temp_dir = tempfile.mkdtemp(prefix="tl_moe_weights_")
    print(f"临时目录: {temp_dir}")
    
    # MLP 权重匹配正则
    mlp_pattern = re.compile(
        r'model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight'
    )
    
    # 输出分片管理
    weight_map = {}       # 权重名 → 分片文件名
    total_size = 0        # 总参数字节数
    copied_main = 0       # 从主模型复制的参数数
    copied_mlp = 0        # 从 MLP 模型复制的参数数
    output_shard_idx = 1  # 当前输出分片编号
    
    # 分片大小限制 (5GB)
    SHARD_SIZE_LIMIT = 5 * 1024 * 1024 * 1024
    current_shard = {}        # 当前分片的权重字典
    current_shard_size = 0    # 当前分片的字节数
    
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
        
        # 如果当前分片太大，先保存
        if current_shard_size + tensor_size > SHARD_SIZE_LIMIT and current_shard:
            save_current_shard()
        
        current_shard[key] = tensor
        current_shard_size += tensor_size
    
    # ========== 阶段 1: 处理主模型权重 ==========
    print(f"\n[阶段 1/3] 处理主模型权重...")
    print(f"  - 非 MLP 权重: 直接复制")
    print(f"  - MLP 权重: 映射到 Expert 0 (no_think)")
    
    for shard_idx, shard_file in enumerate(main_shards):
        print(f"\n  📂 分片 {shard_idx + 1}/{len(main_shards)}: {shard_file}")
        shard_path = os.path.join(main_model_path, shard_file)
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                tensor = f.get_tensor(key)
                
                # 检查是否是 MLP 权重
                mlp_match = mlp_pattern.match(key)
                
                if mlp_match:
                    # MLP 权重 → Expert 0
                    layer_idx = mlp_match.group(1)
                    proj_type = mlp_match.group(2)
                    new_key = f'model.layers.{layer_idx}.mlp.experts.0.{proj_type}.weight'
                    add_weight(new_key, tensor)
                    copied_main += 1
                    if verbose:
                        print(f"    [E0] {key} → {new_key}")
                else:
                    # 非 MLP 权重：直接复制（保持原始 key）
                    add_weight(key, tensor)
                    copied_main += 1
                    if verbose:
                        print(f"    [CP] {key}")
        
        # 每个分片处理完后释放内存
        gc.collect()
    
    print(f"\n  ✅ 阶段 1 完成: 从主模型复制 {copied_main} 个参数")
    
    # ========== 阶段 2: 处理 MLP 权重模型 ==========
    print(f"\n[阶段 2/3] 处理 MLP 权重模型...")
    print(f"  - 仅复制 MLP 权重到 Expert 1 (think)")
    
    for shard_idx, shard_file in enumerate(mlp_shards):
        print(f"\n  📂 分片 {shard_idx + 1}/{len(mlp_shards)}: {shard_file}")
        shard_path = os.path.join(mlp_model_path, shard_file)
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                # 只处理 MLP 权重，跳过其他所有权重
                mlp_match = mlp_pattern.match(key)
                
                if mlp_match:
                    tensor = f.get_tensor(key)
                    layer_idx = mlp_match.group(1)
                    proj_type = mlp_match.group(2)
                    new_key = f'model.layers.{layer_idx}.mlp.experts.1.{proj_type}.weight'
                    add_weight(new_key, tensor)
                    copied_mlp += 1
                    if verbose:
                        print(f"    [E1] {key} → {new_key}")
        
        gc.collect()
    
    print(f"\n  ✅ 阶段 2 完成: 从 MLP 模型复制 {copied_mlp} 个参数")
    
    # ========== 阶段 3: 创建 gate 权重 ==========
    print(f"\n[阶段 3/3] 创建 Router (gate) 权重...")
    print(f"  - 初始化方式: 正态分布, std=0.02")
    print(f"  - 形状: [{2}, {hidden_size}]")
    
    gate_count = 0
    for layer_idx in range(num_layers):
        gate_key = f'model.layers.{layer_idx}.mlp.gate.weight'
        # 使用 bfloat16 与其他权重一致
        gate_weight = (torch.randn(2, hidden_size) * 0.02).to(torch.bfloat16)
        add_weight(gate_key, gate_weight)
        gate_count += 1
        if verbose:
            print(f"    [GT] layer {layer_idx}: mean={gate_weight.float().mean():.4f}, "
                  f"std={gate_weight.float().std():.4f}")
    
    # 保存最后一个分片
    save_current_shard()
    
    print(f"\n  ✅ 阶段 3 完成: 创建 {gate_count} 个 gate 权重")
    
    # ========== 移动文件到输出目录 ==========
    print(f"\n{'='*60}")
    print("移动文件到输出目录")
    print(f"{'='*60}")
    
    total_shards = output_shard_idx - 1
    
    # 重命名分片文件（将 PLACEHOLDER 替换为实际总数）
    for i in range(1, total_shards + 1):
        old_name = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        
        old_path = os.path.join(temp_dir, old_name)
        new_path = os.path.join(output_path, new_name)
        
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            file_size = os.path.getsize(new_path)
            print(f"  ✅ {new_name} ({file_size / 1e9:.2f} GB)")
        
        # 更新 weight_map 中的文件名
        for k, v in list(weight_map.items()):
            if v == old_name:
                weight_map[k] = new_name
    
    # 保存索引文件
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    
    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"  ✅ model.safetensors.index.json")
    
    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return copied_main, copied_mlp, gate_count


# ==================== 兼容性导出函数 ====================
# 这些函数供 __init__.py 导入使用

def load_state_dict_from_path(
    model_path: str,
    device: str = "cpu",
    filter_fn: Optional[callable] = None
) -> Dict[str, torch.Tensor]:
    """从模型路径加载 state_dict（内存友好版本）
    
    Args:
        model_path: 模型路径
        device: 加载设备
        filter_fn: 可选的过滤函数，只加载符合条件的权重
        
    Returns:
        {权重名: 张量} 字典
    """
    from safetensors import safe_open
    
    state_dict = {}
    for shard_file in get_shard_files(model_path):
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in f.keys():
                if filter_fn is None or filter_fn(key):
                    state_dict[key] = f.get_tensor(key)
    return state_dict


def copy_main_model_weights(*args, **kwargs):
    """已弃用，使用 process_weights_streaming"""
    raise NotImplementedError("请使用 process_weights_streaming 进行流式处理")

def copy_mlp_model_weights(*args, **kwargs):
    """已弃用，使用 process_weights_streaming"""
    raise NotImplementedError("请使用 process_weights_streaming 进行流式处理")

def verify_gate_weights(*args, **kwargs):
    """gate 权重验证（由单元测试负责）"""
    pass

def save_model(*args, **kwargs):
    """已弃用，使用 process_weights_streaming"""
    raise NotImplementedError("请使用新的流式处理流程")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="将源模型权重复制到 TL-MoE 模型（内存优化版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python -m src.utils.copy_weights_to_moe \\
        --main_model_path ./models/source/Qwen2.5-7B-Instruct \\
        --mlp_weight_model_path ./models/source/DeepSeek-R1-Distill-Qwen-7B \\
        --output_path ./models/tl_moe_initialized

内存优化：采用流式处理 safetensors，峰值内存约 10GB
        """
    )
    
    parser.add_argument(
        "--main_model_path", type=str, required=True,
        help="主模型路径（用于非 MLP 层和 Expert 0），如 Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--mlp_weight_model_path", type=str, required=True,
        help="MLP 权重模型路径（仅用于 Expert 1 MLP），如 DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument(
        "--output_path", type=str, default="./models/tl_moe_initialized",
        help="输出模型路径 (默认: ./models/tl_moe_initialized)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="输出详细日志（每个权重的复制信息）"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TL-MoE 权重复制工具（内存优化版本）")
    print("="*60)
    print(f"\n主模型路径:      {args.main_model_path}")
    print(f"MLP 权重模型路径: {args.mlp_weight_model_path}")
    print(f"输出路径:         {args.output_path}")
    
    # 步骤 1: 检查模型架构兼容性
    main_config, num_layers = check_model_compatibility(
        args.main_model_path, args.mlp_weight_model_path
    )
    
    # 步骤 2: 创建 TL-MoE 配置
    tl_moe_config = create_tl_moe_config(main_config, args.output_path)
    
    # 步骤 3: 复制辅助文件（tokenizer、模型代码等）
    copy_model_files(args.main_model_path, args.output_path)
    
    # 步骤 4: 流式处理权重
    copied_main, copied_mlp, gate_count = process_weights_streaming(
        args.main_model_path,
        args.mlp_weight_model_path,
        args.output_path,
        num_layers,
        main_config['hidden_size'],
        args.verbose
    )
    
    # 步骤 5: 输出统计
    total_params = copied_main + copied_mlp + gate_count
    print(f"\n{'='*60}")
    print("权重复制完成！")
    print(f"{'='*60}")
    print(f"  - 从主模型复制:    {copied_main} 个参数 (非 MLP + Expert 0)")
    print(f"  - 从 MLP 模型复制: {copied_mlp} 个参数 (Expert 1)")
    print(f"  - 新建 Gate 权重:  {gate_count} 个")
    print(f"  - 总计:            {total_params} 个参数")
    
    print(f"\n输出文件列表:")
    for f_name in sorted(os.listdir(args.output_path)):
        f_path = os.path.join(args.output_path, f_name)
        size = os.path.getsize(f_path)
        if size > 1e9:
            print(f"  - {f_name} ({size/1e9:.2f} GB)")
        elif size > 1e6:
            print(f"  - {f_name} ({size/1e6:.2f} MB)")
        else:
            print(f"  - {f_name} ({size/1e3:.2f} KB)")
    
    print(f"\n下一步：运行单元测试验证权重复制")
    print(f"  python -m pytest src/tests/test_tl_moe_weights_lite.py -v -s")


if __name__ == "__main__":
    main()
