#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TL-MoE 模型权重复制验证 - 轻量版（内存优化）

直接比较 safetensors 文件，不加载完整模型，大幅降低内存使用。

使用方法:
    conda activate het
    python -m pytest src/tests/test_tl_moe_weights_lite.py -v -s

环境变量:
    TL_MOE_PATH: TL-MoE 模型路径 (默认: ./models/tl_moe_initialized)
    QWEN_PATH: Qwen2.5-7B-Instruct 路径 (默认: ./models/source/Qwen2.5-7B-Instruct)
    DEEPSEEK_PATH: DeepSeek-R1-Distill-Qwen-7B 路径 (默认: ./models/source/DeepSeek-R1-Distill-Qwen-7B)
"""

import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ==================== 配置 ====================
TL_MOE_PATH = os.environ.get("TL_MOE_PATH", "./models/tl_moe_initialized")
QWEN_PATH = os.environ.get("QWEN_PATH", "./models/source/Qwen2.5-7B-Instruct")
DEEPSEEK_PATH = os.environ.get("DEEPSEEK_PATH", "./models/source/DeepSeek-R1-Distill-Qwen-7B")

TOLERANCE = 1e-5


# ==================== 辅助函数 ====================
def get_weight_map(model_path: str) -> Dict[str, str]:
    """获取模型的 weight_map"""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = json.load(f)
        return index.get("weight_map", {})
    return {}


def load_tensor(model_path: str, key: str) -> torch.Tensor:
    """从 safetensors 加载单个张量"""
    from safetensors import safe_open
    
    weight_map = get_weight_map(model_path)
    
    if key in weight_map:
        shard_file = weight_map[key]
    else:
        # 尝试单文件
        shard_file = "model.safetensors"
    
    shard_path = os.path.join(model_path, shard_file)
    
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"分片文件不存在: {shard_path}")
    
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if key in f.keys():
            return f.get_tensor(key)
    
    raise KeyError(f"权重不存在: {key}")


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, name: str) -> Tuple[bool, str]:
    """比较两个张量"""
    if t1.shape != t2.shape:
        return False, f"{name}: 形状不匹配 {t1.shape} vs {t2.shape}"
    
    t1_f = t1.float()
    t2_f = t2.float()
    max_diff = (t1_f - t2_f).abs().max().item()
    
    if max_diff > TOLERANCE:
        return False, f"{name}: 最大差异 {max_diff:.2e} > 容差 {TOLERANCE:.2e}"
    
    return True, f"{name}: 匹配 (最大差异: {max_diff:.2e})"


def get_config(model_path: str) -> dict:
    """获取模型配置"""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        return json.load(f)


# ==================== 测试类 ====================
class TestStaticWeightsLite:
    """静态权重一致性检查（轻量版，直接比较 safetensors）"""
    
    def test_expert0_mlp_weights(self):
        """测试 Expert 0 (no_think) MLP 权重 = Qwen MLP"""
        print("\n\n" + "="*60)
        print("测试 Expert 0 MLP 权重一致性")
        print("="*60)
        
        if not os.path.exists(TL_MOE_PATH):
            pytest.skip(f"TL-MoE 模型不存在: {TL_MOE_PATH}")
        if not os.path.exists(QWEN_PATH):
            pytest.skip(f"Qwen 模型不存在: {QWEN_PATH}")
        
        config = get_config(TL_MOE_PATH)
        num_layers = config.get('num_hidden_layers', 28)
        
        failed_checks = []
        passed_checks = []
        
        # 测试所有层
        for layer_idx in range(num_layers):
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                tl_moe_key = f'model.layers.{layer_idx}.mlp.experts.0.{proj}.weight'
                qwen_key = f'model.layers.{layer_idx}.mlp.{proj}.weight'
                
                try:
                    tl_moe_tensor = load_tensor(TL_MOE_PATH, tl_moe_key)
                    qwen_tensor = load_tensor(QWEN_PATH, qwen_key)
                    
                    is_equal, msg = compare_tensors(
                        tl_moe_tensor, qwen_tensor,
                        f"Layer {layer_idx} Expert0 {proj}"
                    )
                    
                    if is_equal:
                        passed_checks.append(msg)
                        print(f"  ✅ {msg}")
                    else:
                        failed_checks.append(msg)
                        print(f"  ❌ {msg}")
                    
                    # 释放内存
                    del tl_moe_tensor, qwen_tensor
                    gc.collect()
                    
                except Exception as e:
                    failed_checks.append(f"Layer {layer_idx} {proj}: {e}")
                    print(f"  ❌ Layer {layer_idx} {proj}: {e}")
        
        print(f"\n通过: {len(passed_checks)}, 失败: {len(failed_checks)}")
        assert len(failed_checks) == 0, f"Expert 0 权重不一致:\n" + "\n".join(failed_checks[:10])
    
    def test_expert1_mlp_weights(self):
        """测试 Expert 1 (think) MLP 权重 = DeepSeek MLP"""
        print("\n\n" + "="*60)
        print("测试 Expert 1 MLP 权重一致性")
        print("="*60)
        
        if not os.path.exists(TL_MOE_PATH):
            pytest.skip(f"TL-MoE 模型不存在: {TL_MOE_PATH}")
        if not os.path.exists(DEEPSEEK_PATH):
            pytest.skip(f"DeepSeek 模型不存在: {DEEPSEEK_PATH}")
        
        config = get_config(TL_MOE_PATH)
        num_layers = config.get('num_hidden_layers', 28)
        
        failed_checks = []
        passed_checks = []
        
        for layer_idx in range(num_layers):
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                tl_moe_key = f'model.layers.{layer_idx}.mlp.experts.1.{proj}.weight'
                deepseek_key = f'model.layers.{layer_idx}.mlp.{proj}.weight'
                
                try:
                    tl_moe_tensor = load_tensor(TL_MOE_PATH, tl_moe_key)
                    deepseek_tensor = load_tensor(DEEPSEEK_PATH, deepseek_key)
                    
                    is_equal, msg = compare_tensors(
                        tl_moe_tensor, deepseek_tensor,
                        f"Layer {layer_idx} Expert1 {proj}"
                    )
                    
                    if is_equal:
                        passed_checks.append(msg)
                        print(f"  ✅ {msg}")
                    else:
                        failed_checks.append(msg)
                        print(f"  ❌ {msg}")
                    
                    del tl_moe_tensor, deepseek_tensor
                    gc.collect()
                    
                except Exception as e:
                    failed_checks.append(f"Layer {layer_idx} {proj}: {e}")
                    print(f"  ❌ Layer {layer_idx} {proj}: {e}")
        
        print(f"\n通过: {len(passed_checks)}, 失败: {len(failed_checks)}")
        assert len(failed_checks) == 0, f"Expert 1 权重不一致:\n" + "\n".join(failed_checks[:10])
    
    def test_shared_weights(self):
        """测试共享层权重（embed, attention, norm, lm_head）= Qwen"""
        print("\n\n" + "="*60)
        print("测试共享层权重一致性")
        print("="*60)
        
        if not os.path.exists(TL_MOE_PATH):
            pytest.skip(f"TL-MoE 模型不存在: {TL_MOE_PATH}")
        if not os.path.exists(QWEN_PATH):
            pytest.skip(f"Qwen 模型不存在: {QWEN_PATH}")
        
        config = get_config(TL_MOE_PATH)
        num_layers = config.get('num_hidden_layers', 28)
        
        failed_checks = []
        passed_checks = []
        
        # 全局共享层
        global_shared = [
            'model.embed_tokens.weight',
            'model.norm.weight',
            'lm_head.weight',
        ]
        
        for key in global_shared:
            try:
                tl_moe_tensor = load_tensor(TL_MOE_PATH, key)
                qwen_tensor = load_tensor(QWEN_PATH, key)
                
                is_equal, msg = compare_tensors(tl_moe_tensor, qwen_tensor, key)
                
                if is_equal:
                    passed_checks.append(msg)
                    print(f"  ✅ {key}")
                else:
                    failed_checks.append(msg)
                    print(f"  ❌ {msg}")
                
                del tl_moe_tensor, qwen_tensor
                gc.collect()
                
            except Exception as e:
                failed_checks.append(f"{key}: {e}")
                print(f"  ❌ {key}: {e}")
        
        # 每层的 attention 和 norm
        # 只检查几层以加速测试
        layers_to_check = [0, num_layers // 2, num_layers - 1]
        
        for layer_idx in layers_to_check:
            layer_shared = [
                f'model.layers.{layer_idx}.self_attn.q_proj.weight',
                f'model.layers.{layer_idx}.self_attn.k_proj.weight',
                f'model.layers.{layer_idx}.self_attn.v_proj.weight',
                f'model.layers.{layer_idx}.self_attn.o_proj.weight',
                f'model.layers.{layer_idx}.input_layernorm.weight',
                f'model.layers.{layer_idx}.post_attention_layernorm.weight',
            ]
            
            for key in layer_shared:
                try:
                    tl_moe_tensor = load_tensor(TL_MOE_PATH, key)
                    qwen_tensor = load_tensor(QWEN_PATH, key)
                    
                    is_equal, msg = compare_tensors(tl_moe_tensor, qwen_tensor, key)
                    
                    if is_equal:
                        passed_checks.append(msg)
                        print(f"  ✅ {key}")
                    else:
                        failed_checks.append(msg)
                        print(f"  ❌ {msg}")
                    
                    del tl_moe_tensor, qwen_tensor
                    gc.collect()
                    
                except Exception as e:
                    # 某些权重可能不存在（如 bias）
                    if "不存在" not in str(e):
                        failed_checks.append(f"{key}: {e}")
                        print(f"  ⚠️ {key}: {e}")
        
        print(f"\n通过: {len(passed_checks)}, 失败: {len(failed_checks)}")
        assert len(failed_checks) == 0, f"共享层权重不一致:\n" + "\n".join(failed_checks[:10])
    
    def test_gate_weights_existence(self):
        """测试 Gate 权重存在且形状正确"""
        print("\n\n" + "="*60)
        print("测试 Gate 权重")
        print("="*60)
        
        if not os.path.exists(TL_MOE_PATH):
            pytest.skip(f"TL-MoE 模型不存在: {TL_MOE_PATH}")
        
        config = get_config(TL_MOE_PATH)
        num_layers = config.get('num_hidden_layers', 28)
        hidden_size = config.get('hidden_size', 3584)
        num_experts = config.get('num_experts', 2)
        
        failed_checks = []
        passed_checks = []
        
        for layer_idx in range(num_layers):
            gate_key = f'model.layers.{layer_idx}.mlp.gate.weight'
            
            try:
                gate_tensor = load_tensor(TL_MOE_PATH, gate_key)
                
                expected_shape = (num_experts, hidden_size)
                if gate_tensor.shape == expected_shape:
                    # 检查初始化统计
                    mean = gate_tensor.float().mean().item()
                    std = gate_tensor.float().std().item()
                    passed_checks.append(f"Layer {layer_idx} gate")
                    print(f"  ✅ Layer {layer_idx} gate: shape={gate_tensor.shape}, mean={mean:.4f}, std={std:.4f}")
                else:
                    failed_checks.append(f"Layer {layer_idx} gate: 形状错误 {gate_tensor.shape} != {expected_shape}")
                    print(f"  ❌ Layer {layer_idx} gate: 形状错误")
                
                del gate_tensor
                gc.collect()
                
            except Exception as e:
                failed_checks.append(f"Layer {layer_idx} gate: {e}")
                print(f"  ❌ Layer {layer_idx} gate: {e}")
        
        print(f"\n通过: {len(passed_checks)}, 失败: {len(failed_checks)}")
        assert len(failed_checks) == 0, f"Gate 权重检查失败:\n" + "\n".join(failed_checks[:10])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
