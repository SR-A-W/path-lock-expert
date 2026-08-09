#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TL-MoE 模型前向传播行为测试

测试内容：
1. Embedding 层输出一致性（TestForwardBehavior::test_embedding_layer_output）
   - Embedding 是共享层，输出应与 Qwen2.5-7B-Instruct 完全一致

2. 强制路由到 Expert 0，第一层输出一致性（TestForwardBehavior::test_expert0_first_layer_output）
   - 通过 hook 修改 gate logits，强制所有 token 路由到 Expert 0 (no_think)
   - 第一层的 hidden_states 应与 Qwen2.5-7B-Instruct 完全一致

3. 正常路由下模型不胡言乱语测试（TestForwardBehavior::test_model_generates_reasonable_output）
   - 多个 prompt 的 greedy 生成测试，检查输出合理性

使用方法:
    conda activate het
    
    # 运行所有前向传播测试
    python -m pytest src/tests/test_tl_moe_weights.py -v -s
    
    # 运行单个测试
    python -m pytest src/tests/test_tl_moe_weights.py::TestForwardBehavior::test_embedding_layer_output -v -s
    
    # 只运行轻量测试（不需要 Qwen 模型）
    python -m pytest src/tests/test_tl_moe_weights.py::TestForwardBehavior::test_model_generates_reasonable_output -v -s

环境变量:
    TL_MOE_PATH: TL-MoE 模型路径 (默认: ./models/tl_moe_initialized)
    QWEN_PATH: Qwen2.5-7B-Instruct 路径 (默认: ./models/source/Qwen2.5-7B-Instruct)
    DEEPSEEK_PATH: DeepSeek-R1-Distill-Qwen-7B 路径 (默认: ./models/source/DeepSeek-R1-Distill-Qwen-7B)

注意：
    - 前向传播测试需要加载完整模型，约需 40GB+ 内存
    - 使用 bfloat16 精度运行在 CPU 上
    - 如果内存不足，可以只运行 test_model_generates_reasonable_output（仅需加载 TL-MoE 模型）
"""

import gc
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch
from torch import nn

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== 配置 ====================
# 模型路径（可通过环境变量覆盖）
TL_MOE_PATH = os.environ.get("TL_MOE_PATH", "./models/tl_moe_initialized")
QWEN_PATH = os.environ.get("QWEN_PATH", "./models/source/Qwen2.5-7B-Instruct")
DEEPSEEK_PATH = os.environ.get("DEEPSEEK_PATH", "./models/source/DeepSeek-R1-Distill-Qwen-7B")


# ==================== 辅助函数 ====================
def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name: str,
    tolerance: float = 1e-5
) -> Tuple[bool, str]:
    """比较两个张量是否相等
    
    Args:
        tensor1: 第一个张量
        tensor2: 第二个张量
        name: 参数名称（用于日志）
        tolerance: 容差
        
    Returns:
        (is_equal, message)
    """
    if tensor1.shape != tensor2.shape:
        return False, f"{name}: 形状不匹配 {tensor1.shape} vs {tensor2.shape}"
    
    # 转换为 float32 进行精确比较
    t1 = tensor1.float()
    t2 = tensor2.float()
    
    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    if max_diff > tolerance:
        return False, (f"{name}: 最大差异 {max_diff:.2e} > 容差 {tolerance:.2e} "
                       f"(平均差异: {mean_diff:.2e})")
    
    return True, f"{name}: 匹配 (最大差异: {max_diff:.2e}, 平均差异: {mean_diff:.2e})"


def force_expert_routing(model, expert_idx: int):
    """通过 forward hook 强制路由到指定专家
    
    为每层的 gate (nn.Linear) 注册 forward hook，
    将 gate 输出修改为：目标专家的 logit 极大，其他极小。
    
    Args:
        model: TL-MoE 模型
        expert_idx: 要强制使用的专家索引 (0 或 1)
        
    Returns:
        hooks: 需要在使用后移除的 hook 列表
    """
    hooks = []
    
    def create_gate_hook(expert_idx):
        """创建修改 gate logits 的 hook"""
        def hook(module, input, output):
            # output shape: (batch * seq_len, num_experts)
            # 将目标专家的 logit 设为极大值，其他设为极小值
            new_output = torch.full_like(output, -1e9)
            new_output[:, expert_idx] = 1e9
            return new_output
        return hook
    
    # 为每层的 MoE block 的 gate 添加 hook
    for layer in model.model.layers:
        if hasattr(layer.mlp, 'gate'):
            hook = layer.mlp.gate.register_forward_hook(create_gate_hook(expert_idx))
            hooks.append(hook)
    
    return hooks


# ==================== Fixtures ====================
@pytest.fixture(scope="module")
def tl_moe_model():
    """加载 TL-MoE 模型（模块级别缓存，整个测试模块只加载一次）"""
    from src.models.qwen2_tl_moe import Qwen2TLMoeForCausalLM, Qwen2TLMoeConfig
    
    if not os.path.exists(TL_MOE_PATH):
        pytest.skip(f"TL-MoE 模型不存在: {TL_MOE_PATH}")
    
    print(f"\n加载 TL-MoE 模型: {TL_MOE_PATH}")
    
    # 使用 trust_remote_code=True 加载（模型代码在输出目录中）
    # 但由于我们已经有本地的模型类，直接用配置加载
    config = Qwen2TLMoeConfig.from_pretrained(TL_MOE_PATH)
    model = Qwen2TLMoeForCausalLM.from_pretrained(
        TL_MOE_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()
    
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  - dtype: {next(model.parameters()).dtype}")
    
    return model


@pytest.fixture(scope="module")
def qwen_model():
    """加载 Qwen2.5-7B-Instruct 模型（模块级别缓存）"""
    from transformers import AutoModelForCausalLM
    
    if not os.path.exists(QWEN_PATH):
        pytest.skip(f"Qwen 模型不存在: {QWEN_PATH}")
    
    print(f"\n加载 Qwen 模型: {QWEN_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()
    
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """加载 tokenizer（使用 Qwen 的 tokenizer）"""
    from transformers import AutoTokenizer
    
    # 优先使用 TL-MoE 的 tokenizer，其次 Qwen 的
    path = TL_MOE_PATH if os.path.exists(TL_MOE_PATH) else QWEN_PATH
    if not os.path.exists(path):
        pytest.skip(f"Tokenizer 路径不存在: {path}")
    
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


# ==================== 前向传播行为测试 ====================
class TestForwardBehavior:
    """前向传播行为测试
    
    核心逻辑：
    - Embedding 和 Attention 层是两个模型共享的
    - Expert 0 的 MLP 权重来自 Qwen2.5-7B-Instruct
    - 因此，当强制路由到 Expert 0 时，TL-MoE 的行为应该等价于 Qwen2.5-7B-Instruct
    """
    
    def test_embedding_layer_output(self, tl_moe_model, qwen_model, tokenizer):
        """测试 Embedding 层输出一致性
        
        Embedding 层是共享的，应该与 Qwen 完全一致。
        这是最基本的 sanity check。
        """
        print("\n\n" + "="*60)
        print("测试 Embedding 层输出一致性")
        print("="*60)
        
        test_input = "Hello, how are you today?"
        inputs = tokenizer(test_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"测试输入: '{test_input}'")
        print(f"Token 数量: {input_ids.shape[1]}")
        
        with torch.no_grad():
            # 获取 TL-MoE embedding 输出
            tl_moe_embeds = tl_moe_model.model.embed_tokens(input_ids)
            # 获取 Qwen embedding 输出
            qwen_embeds = qwen_model.model.embed_tokens(input_ids)
        
        is_equal, msg = compare_tensors(
            tl_moe_embeds, qwen_embeds,
            "Embedding output",
            tolerance=1e-5
        )
        
        print(f"\n结果: {msg}")
        assert is_equal, msg
        print("✅ Embedding 层输出一致性测试通过")
    
    def test_expert0_first_layer_output(self, tl_moe_model, qwen_model, tokenizer):
        """测试强制路由到 Expert 0 时，第一层输出与 Qwen 一致
        
        核心原理：
        - 第一层的输入是 embedding 输出（共享层，已验证一致）
        - 第一层的 attention 权重来自 Qwen（共享层）
        - 第一层的 MLP Expert 0 权重来自 Qwen
        - 因此，当强制所有 token 路由到 Expert 0 时，
          第一层的输出应该与 Qwen 完全一致
        
        注意：由于 bfloat16 精度和计算顺序的微小差异，
        使用较宽松的容差（1e-3）
        """
        print("\n\n" + "="*60)
        print("测试强制路由到 Expert 0，第一层输出一致性")
        print("="*60)
        
        # 准备输入
        test_input = "The capital of France is Paris."
        inputs = tokenizer(test_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"测试输入: '{test_input}'")
        print(f"Token IDs: {input_ids.tolist()}")
        print(f"Token 数量: {input_ids.shape[1]}")
        
        # 1. 获取 Qwen 的各层输出
        print("\n运行 Qwen 模型...")
        with torch.no_grad():
            qwen_outputs = qwen_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        
        # hidden_states[0] = embedding 输出
        # hidden_states[1] = 第一层输出
        # hidden_states[28] = 最后一层输出
        qwen_hidden_states = qwen_outputs.hidden_states
        print(f"  - Qwen hidden_states 数量: {len(qwen_hidden_states)}")
        print(f"  - 第一层输出 shape: {qwen_hidden_states[1].shape}")
        
        # 2. 强制 TL-MoE 路由到 Expert 0，获取各层输出
        print("\n运行 TL-MoE 模型（强制路由到 Expert 0）...")
        hooks = force_expert_routing(tl_moe_model, expert_idx=0)
        
        try:
            with torch.no_grad():
                tl_moe_outputs = tl_moe_model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
        finally:
            # 确保移除所有 hooks
            for hook in hooks:
                hook.remove()
        
        tl_moe_hidden_states = tl_moe_outputs.hidden_states
        print(f"  - TL-MoE hidden_states 数量: {len(tl_moe_hidden_states)}")
        
        # 3. 逐层比较 hidden states
        print("\n逐层比较 hidden states:")
        
        # 比较 embedding 输出（第 0 层）
        is_equal, msg = compare_tensors(
            tl_moe_hidden_states[0], qwen_hidden_states[0],
            "Layer 0 (Embedding)", tolerance=1e-5
        )
        print(f"  {msg}")
        assert is_equal, f"Embedding 输出不一致: {msg}"
        
        # 比较第一层输出
        is_equal, msg = compare_tensors(
            tl_moe_hidden_states[1], qwen_hidden_states[1],
            "Layer 1 output", tolerance=1e-3  # bfloat16 精度
        )
        print(f"  {msg}")
        
        # 详细差异统计
        diff = (tl_moe_hidden_states[1].float() - qwen_hidden_states[1].float()).abs()
        print(f"    - 最大差异: {diff.max().item():.6e}")
        print(f"    - 平均差异: {diff.mean().item():.6e}")
        print(f"    - 标准差:   {diff.std().item():.6e}")
        print(f"    - > 1e-4 的元素比例: {(diff > 1e-4).float().mean().item():.2%}")
        
        assert is_equal, f"第一层输出不一致: {msg}"
        
        # 比较更多层（检查误差是否随层数增加而累积）
        print("\n更多层的差异统计:")
        for layer_idx in [2, 5, 10, 14, 20, 27]:
            if layer_idx + 1 < len(tl_moe_hidden_states):
                diff = (tl_moe_hidden_states[layer_idx + 1].float() - 
                       qwen_hidden_states[layer_idx + 1].float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"  Layer {layer_idx + 1}: max_diff={max_diff:.6e}, "
                      f"mean_diff={mean_diff:.6e}")
        
        print("\n✅ 强制路由到 Expert 0 的第一层输出一致性测试通过")
    
    def test_model_generates_reasonable_output(self, tl_moe_model, tokenizer):
        """测试模型在正常路由下能生成合理的输出（不胡言乱语）
        
        这是一个基本的 sanity check，验证模型能够生成连贯的文本。
        使用 greedy decoding（do_sample=False）确保结果可重复。
        """
        print("\n\n" + "="*60)
        print("测试模型生成合理性（正常路由）")
        print("="*60)
        
        # 使用多个不同难度的 prompt 进行测试
        test_prompts = [
            ("The capital of France is", "简单事实回答"),
            ("2 + 2 =", "简单数学"),
            ("Hello! My name is", "简单对话"),
            ("The meaning of life is", "开放性问题"),
        ]
        
        all_passed = True
        
        for prompt, description in test_prompts:
            print(f"\n--- {description} ---")
            print(f"输入: '{prompt}'")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = tl_moe_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=30,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            print(f"完整输出: '{generated_text}'")
            print(f"新生成:   '{new_text}'")
            print(f"新 token 数: {len(new_tokens)}")
            
            # 检查 1：输出长度应该大于输入
            if len(generated_text) <= len(prompt):
                print(f"  ⚠️ 模型没有生成新内容")
                all_passed = False
                continue
            
            # 检查 2：新生成的文本不应该全是特殊字符或乱码
            # 简单启发式：检查可打印字符比例
            printable_ratio = sum(1 for c in new_text if c.isprintable() or c.isspace()) / max(len(new_text), 1)
            if printable_ratio < 0.5:
                print(f"  ⚠️ 输出可能包含乱码（可打印字符比例: {printable_ratio:.2%}）")
                all_passed = False
                continue
            
            # 检查 3：不应该是无意义的重复
            if len(new_text) > 10:
                # 检查是否有严重重复（同一个短 pattern 重复多次）
                words = new_text.split()
                if len(words) > 3:
                    unique_words = set(words)
                    uniqueness = len(unique_words) / len(words)
                    if uniqueness < 0.1:
                        print(f"  ⚠️ 输出严重重复（词汇唯一性: {uniqueness:.2%}）")
                        all_passed = False
                        continue
            
            print(f"  ✅ 通过")
        
        assert all_passed, "部分 prompt 的生成质量不达标"
        print("\n✅ 模型生成合理性测试通过")
    
    def test_expert0_full_model_logits(self, tl_moe_model, qwen_model, tokenizer):
        """测试强制路由到 Expert 0 时，最终 logits 与 Qwen 的接近程度
        
        这是一个进阶测试，比较全模型（28层）通过后的最终 logits 输出。
        由于 bfloat16 精度误差在 28 层传播后会累积，
        使用更宽松的容差，并主要关注 top-k token 的一致性。
        """
        print("\n\n" + "="*60)
        print("测试 Expert 0 全模型 logits 一致性（进阶）")
        print("="*60)
        
        test_input = "The capital of France is"
        inputs = tokenizer(test_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"测试输入: '{test_input}'")
        
        # 1. Qwen 模型的 logits
        print("\n运行 Qwen 模型...")
        with torch.no_grad():
            qwen_outputs = qwen_model(input_ids=input_ids)
        qwen_logits = qwen_outputs.logits
        
        # 2. TL-MoE 强制路由 Expert 0 的 logits
        print("运行 TL-MoE 模型（强制路由到 Expert 0）...")
        hooks = force_expert_routing(tl_moe_model, expert_idx=0)
        try:
            with torch.no_grad():
                tl_moe_outputs = tl_moe_model(input_ids=input_ids)
        finally:
            for hook in hooks:
                hook.remove()
        tl_moe_logits = tl_moe_outputs.logits
        
        # 3. 比较最后一个 token 位置的 top-k 预测
        last_pos = -1
        qwen_last_logits = qwen_logits[0, last_pos]
        tl_moe_last_logits = tl_moe_logits[0, last_pos]
        
        # Top-10 token 对比
        qwen_top10 = torch.topk(qwen_last_logits, 10)
        tl_moe_top10 = torch.topk(tl_moe_last_logits, 10)
        
        print(f"\nQwen Top-10 预测:")
        for i, (idx, score) in enumerate(zip(qwen_top10.indices, qwen_top10.values)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' (id={idx.item()}, logit={score.item():.4f})")
        
        print(f"\nTL-MoE (Expert 0) Top-10 预测:")
        for i, (idx, score) in enumerate(zip(tl_moe_top10.indices, tl_moe_top10.values)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' (id={idx.item()}, logit={score.item():.4f})")
        
        # 检查 top-1 预测是否一致
        qwen_top1 = qwen_top10.indices[0].item()
        tl_moe_top1 = tl_moe_top10.indices[0].item()
        
        qwen_top1_token = tokenizer.decode([qwen_top1])
        tl_moe_top1_token = tokenizer.decode([tl_moe_top1])
        
        print(f"\nTop-1 预测对比:")
        print(f"  Qwen:   '{qwen_top1_token}' (id={qwen_top1})")
        print(f"  TL-MoE: '{tl_moe_top1_token}' (id={tl_moe_top1})")
        
        if qwen_top1 == tl_moe_top1:
            print("  ✅ Top-1 预测一致")
        else:
            print("  ⚠️ Top-1 预测不一致（bfloat16 累积误差可能导致）")
        
        # 检查 top-5 overlap
        qwen_top5_set = set(qwen_top10.indices[:5].tolist())
        tl_moe_top5_set = set(tl_moe_top10.indices[:5].tolist())
        overlap = len(qwen_top5_set & tl_moe_top5_set)
        print(f"\nTop-5 重叠度: {overlap}/5")
        
        # Logits 整体差异统计
        diff = (qwen_last_logits.float() - tl_moe_last_logits.float()).abs()
        print(f"\n最后位置 logits 差异统计:")
        print(f"  - 最大差异: {diff.max().item():.6e}")
        print(f"  - 平均差异: {diff.mean().item():.6e}")
        print(f"  - 标准差:   {diff.std().item():.6e}")
        
        # 断言：Top-1 应该一致（在大多数情况下），至少 Top-5 重叠 >= 3
        assert overlap >= 3, f"Top-5 重叠度太低: {overlap}/5，模型行为可能存在问题"
        
        print("\n✅ 全模型 logits 一致性测试通过")


# ==================== 独立运行支持 ====================
if __name__ == "__main__":
    # 允许直接运行此文件进行测试
    pytest.main([__file__, "-v", "-s"])
