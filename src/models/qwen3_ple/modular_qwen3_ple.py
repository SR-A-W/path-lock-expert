# coding=utf-8
# Copyright 2026 Hybrid-Expert-Thinking Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""PyTorch Qwen3 PLE (Path-Lock Expert) model.

序列级硬路由：根据用户消息末尾的控制词 `/think` 或 `/no_think` 决定整条序列走 expert 0 还是 expert 1。

路由检测规则：
1. **锚点定位**：找最后一个 `<|im_start|>assistant` token 对（特殊 token 151644 + 77091），
   作为 scan 区间的右边界。这把 prompt 区和 response 区干净切开。
2. **scan_region**：锚点之前的 token（branch ①）；找不到锚点时（裸推理无 chat template，
   branch ②）整段扫描。
3. **decode + last-wins**：tokenizer.decode 后用 regex `(?<!<)/no_think(?!\w)|(?<!<)/think(?!\w)`
   findall，取**最后一个**匹配。`(?<!<)` 排除 `</think>` 闭合标签，`(?!\w)` 防 `/think_hard` 类子串。
4. **NO_MATCH 行为**：由 `config.strict_routing_match` 控制——
   - `True`（默认）→ raise（fail-fast，防 V2 灾难重演）
   - `False` → log warning + 退 `config.default_routing_index`

**接口签名 Phase 2 余量**：`_determine_routing_index` 返回 `LongTensor[B]`，MLP / DecoderLayer 收
`LongTensor[B]`——Phase 1 内部 `assert B==1`；Phase 2 拆断言 + 改 dispatch 为 mask 形式即可，
外层接口零改动。
"""

import logging as py_logging
import re
from typing import Optional

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.generic import check_model_inputs
from ..qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
)
from .configuration_qwen3_ple import Qwen3PLEConfig


logger = py_logging.getLogger(__name__)


# 锚点 token 对：<|im_start|> + assistant
# 这两个特殊 token 在自然文本编码中不会同时连续出现（除非用户字面注入，已 Empiricist
# M6 实测 37,500 训练数据 0 注入；推理侧作为 API 契约文档化）
_ANCHOR_BIGRAM = (151644, 77091)

# Routing 检测的核心 regex
# - alternation 顺序：/no_think 在前（字符串重叠保护，/no_think 含 /think 子串）
# - (?<!<)：negative lookbehind，排除 </think> 闭合标签
# - (?!\w)：negative lookahead，防 /think_hard 等子串误触发
_ROUTING_PATTERN = re.compile(r'(?<!<)/no_think(?!\w)|(?<!<)/think(?!\w)')


# 模块级 tokenizer 缓存，按 model path 索引。
# 避免每次路由检测都重新加载 tokenizer；多个 model 实例共享同一 tokenizer。
_TOKENIZER_CACHE: dict = {}


def _get_routing_tokenizer(config):
    """惰性加载 tokenizer，按 config._name_or_path 索引缓存。

    路由检测需要 tokenizer.decode 把 token 序列还原成字符串做 regex 匹配。
    此函数返回与模型权重同源的 tokenizer。
    """
    path = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    if path is None:
        raise ValueError(
            "Qwen3PLE routing detection requires a tokenizer, but config._name_or_path is not set. "
            "This is normally set automatically by from_pretrained(). "
            "If loading manually, pass tokenizer reference via prepare_inputs_for_generation "
            "or call from_pretrained with model directory."
        )
    if path not in _TOKENIZER_CACHE:
        from ..auto import AutoTokenizer
        _TOKENIZER_CACHE[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _TOKENIZER_CACHE[path]


def _find_last_bigram(ids_row: torch.LongTensor, anchor=_ANCHOR_BIGRAM) -> Optional[int]:
    """从右向左找最后一个匹配 `anchor` token 对的起始位置。

    Args:
        ids_row: 1D LongTensor (单条序列)
        anchor: 长度 2 的 tuple (t0, t1)

    Returns:
        Optional[int]: 起始位置 i 使得 ids_row[i]==t0 且 ids_row[i+1]==t1；找不到返回 None
    """
    if ids_row.shape[0] < 2:
        return None
    t0, t1 = anchor
    # 向量化：matches[i] = (ids[i]==t0 AND ids[i+1]==t1)
    matches = (ids_row[:-1] == t0) & (ids_row[1:] == t1)
    if not matches.any():
        return None
    return int(matches.nonzero()[-1, 0].item())


def _detect_routing_for_row(
    ids_row: torch.LongTensor, config: Qwen3PLEConfig, tokenizer
) -> int:
    """Per-row 路由检测核心 (anchor + regex + last-wins)。

    Args:
        ids_row: 1D LongTensor，单条序列
        config: Qwen3PLEConfig
        tokenizer: HuggingFace tokenizer (用于 decode)

    Returns:
        int: 0 (no_think / expert 0) 或 1 (think / expert 1)
    """
    default_idx = getattr(config, "default_routing_index", 0)
    strict = getattr(config, "strict_routing_match", True)

    # Step 1: 锚点 (右边界)
    a_pos = _find_last_bigram(ids_row, _ANCHOR_BIGRAM)

    # Step 2: 分支选 scan_region
    if a_pos is not None:
        # branch ①：训练 / 套模板推理。锚点之前 = 用户 prompt 区（含 chat template
        # 的 system 和 user 轮），response 段被结构性排除
        scan_region = ids_row[:a_pos]
    else:
        # branch ②：裸推理无模板。整段 = 用户输入本身（推理时通常无 response）
        scan_region = ids_row

    # Step 3: decode + regex 后向 last-wins
    if scan_region.shape[0] == 0:
        s = ""
    else:
        s = tokenizer.decode(scan_region.tolist())
    matches = _ROUTING_PATTERN.findall(s)

    # Step 4: NO_MATCH 行为
    if not matches:
        if strict:
            raise ValueError(
                f"Qwen3PLE routing: control token /think or /no_think not found in input "
                f"(scan_region len={int(scan_region.shape[0])}, has_anchor={a_pos is not None}). "
                f"Either add control word to input end, or set config.strict_routing_match=False "
                f"to fall back to default_routing_index={default_idx}."
            )
        logger.warning(
            "Qwen3PLE routing NO_MATCH; falling back to config.default_routing_index=%d. "
            "Set config.strict_routing_match=True to raise instead.",
            default_idx,
        )
        return default_idx

    last_match = matches[-1]
    # last_match 是字符串 "/think" 或 "/no_think"
    return 0 if last_match == "/no_think" else 1


def _determine_routing_index(
    input_ids: torch.LongTensor, config: Qwen3PLEConfig, tokenizer=None
) -> torch.LongTensor:
    """Per-sample routing detection。返回 `LongTensor[B]`。

    Phase 1：通常 B=1（LLaMA-Factory + per_device_train_batch_size=1）。
    Phase 2：同函数支持 B>1，无需改动签名或实现。

    Args:
        input_ids: 2D LongTensor [B, S]
        config: Qwen3PLEConfig
        tokenizer: 可选；若 None 则自动 lazy-load

    Returns:
        LongTensor[B]: 每条序列的路由索引（0 或 1）
    """
    if tokenizer is None:
        tokenizer = _get_routing_tokenizer(config)
    B = input_ids.shape[0]
    results = [_detect_routing_for_row(input_ids[b], config, tokenizer) for b in range(B)]
    return torch.tensor(results, dtype=torch.long, device=input_ids.device)


class Qwen3PLEExpertMLP(Qwen3MLP):
    """单个 MLP 专家（与 Qwen3MLP 结构完全一致），用于 PLE 的双专家容器。

    Qwen3MLP 继承自 GemmaMLP，包含 gate_proj、up_proj、down_proj，均无 bias。
    forward: down_proj(act_fn(gate_proj(x)) * up_proj(x))
    """

    pass


class Qwen3PLEMLP(nn.Module):
    """Path-Lock 双专家 MLP 容器（Scheme A 训练 / Scheme B 推理，按梯度上下文自动切换）。

    包含两个独立的 MLP 专家：
    - experts[0]：no_think 专家
    - experts[1]：think 专家

    **forward 按 `torch.is_grad_enabled()` 自动选 dispatch 方案**：

    **Scheme A（梯度开启 = 训练 / 有反向）**：两个 expert **都算**，per-sample mask
    选择 `out = out0*(1-mask) + out1*mask`。代价 2× MLP FLOPs，换来：
    - per-sample 梯度正确（同 batch 内可混合 think/no_think）
    - **ZeRO-2 结构性安全**：两 expert 永远参与 forward，不出现"某 expert 单步零梯度"
      的边角（同质 batch / 多卡 reduce 下都良性；详见 Phase 3 D1 硬门）

    **Scheme B（梯度关闭 = 推理 / 无反向）**：只算被路由到的 expert，~1× MLP FLOPs。
    推理无反向 → 无梯度 → 零梯度问题不存在 → 安全且省一半 MLP 算力。
    - 同质 batch（推理常见，所有样本同 routing）→ 单次 expert 调用
    - 混合 batch → 按 routing 分段，每段只调对应 expert

    **数值等价**：Scheme B 对被选中 expert 的输出与 Scheme A bit-identical
    （A 中 `out_sel*1.0 + out_other*0.0` == out_sel），切换不改变结果、只改算力。
    判别用 `torch.is_grad_enabled()` 而非 `self.training`：grad 开着=可能反向=必须 A；
    grad 关着=纯前向=B 安全。比 model.training 更稳（eval-by-loss 边角也不误判）。
    """

    def __init__(self, config: Qwen3PLEConfig):
        super().__init__()
        self.experts = nn.ModuleList([Qwen3PLEExpertMLP(config) for _ in range(2)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_index: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S, H]
            routing_index: [B] LongTensor，0=no_think (expert 0)，1=think (expert 1)

        Returns:
            [B, S, H]，按 routing_index 逐样本从对应 expert 取输出
        """
        # ============================================================
        # 【总开关】当前是"训练（有反向传播）"还是"推理（无反向）"？
        #   torch.is_grad_enabled()==True  → 在算梯度（训练）→ 走 Scheme A
        #   torch.is_grad_enabled()==False → 无梯度（推理/no_grad）→ 走 Scheme B
        #   用它而非 self.training：grad 开=可能反向=必须 A；grad 关=纯前向=B 安全
        # ============================================================
        if torch.is_grad_enabled():
            # ========================================================
            # 【Scheme A — 训练】两个 expert 都算，再用 mask 逐样本挑。
            #   代价：2× MLP 算力（两个 expert 都跑一遍）
            #   好处：两 expert 都参与 forward → 反向时都拿到真实梯度 →
            #         不出现"某 expert 整步零梯度"（ZeRO-2 安全，详见 Tutor 讲解）
            # ========================================================
            out0 = self.experts[0](hidden_states)               # 〔A-步1〕expert0(no_think) 跑整 batch → [B,S,H]
            out1 = self.experts[1](hidden_states)               # 〔A-步2〕expert1(think)   跑整 batch → [B,S,H]
            # 〔A-步3〕routing_index 从 [B] 变形成 [B,1,1] 以便与 [B,S,H] 广播相乘
            #   例：routing=[0,1] → mask=[[[0.]],[[1.]]]（dtype 对齐 out0 避免类型转换）
            mask = routing_index.view(-1, 1, 1).to(out0.dtype)
            # 〔A-步4〕逐样本线性选择：
            #   routing=0 → out0*(1-0)+out1*0 = out0（取 expert0）
            #   routing=1 → out0*(1-1)+out1*1 = out1（取 expert1）
            return out0 * (1.0 - mask) + out1 * mask

        # ============================================================
        # 【Scheme B — 推理】只算被选中的那个 expert，~1× MLP 算力。
        #   推理无反向 → 无梯度 → "零梯度"问题不存在 → 不必两个都算。
        #   与 Scheme A 对被选中 expert 的输出 bit-identical（A 里另一个乘 0 是精确的）。
        #
        #   注：原有"同质快路径"（torch.unique + 单 expert 直调）已于 2026-05-23
        #   注释掉（benchmark job 3359416, A100）。依据：
        #     - 端到端 generate(batch=1, prefill+64 decode) 删掉快路径仅 -0.3%~+0.7%（噪声级）
        #     - 孤立 MLP 的 decode(S=1) 层级快路径确有 ~62% 优势，但被 generate 端
        #       attention/采样/launch 等开销掩盖 → 端到端不可见
        #   故移除以减少分支、提升 robustness（少一条 code path = 少一处出错）。
        #   下面的混合路径本身就涵盖同质情形（同质时另一个 expert 的 sel 全 False、被跳过），
        #   所以删快路径不影响正确性。若未来需要高吞吐 HF-native decode 服务再恢复。
        #
        #   --- 原快路径（保留作注释，便于将来恢复）-------------------------
        #   unique = torch.unique(routing_index)        # 〔B-步1〕batch 里几种 routing 值
        #   if unique.numel() == 1:                     # 〔B-步2·快路径〕整 batch 同一种
        #       return self.experts[int(unique.item())](hidden_states)
        #   -----------------------------------------------------------------
        # ============================================================
        # 〔B-步3〕按 routing 分段：每个 expert 只算自己那部分（同质时另一段被跳过）
        out = torch.empty_like(hidden_states)                   # 先开一个与输入同形状的空输出张量
        for e in (0, 1):                                        # 分别处理 expert 0 和 expert 1
            sel = routing_index == e                            # 布尔掩码：哪些样本走 expert e（例 e=0→[T,F,T]）
            if sel.any():                                       # 若有样本走这个 expert
                out[sel] = self.experts[e](hidden_states[sel])  # 只把这些样本喂给 expert e、写回原位
        return out                                              # 每个 expert 只算自己那部分 → 总算力 ~1×


class Qwen3PLEDecoderLayer(Qwen3DecoderLayer):
    """Decoder 层：Attention 与 Qwen3 一致（含 q_norm/k_norm），MLP 替换为 Qwen3PLEMLP，
    forward 透传 `routing_index: LongTensor[B]`。
    """

    def __init__(self, config: Qwen3PLEConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        # 将父类创建的 Qwen3MLP 替换为 PLE 双专家容器
        self.mlp = Qwen3PLEMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        routing_index: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if routing_index is None:
            raise ValueError(
                "Qwen3PLEDecoderLayer.forward requires routing_index (LongTensor[B]); "
                "got None. This should be supplied by Qwen3PLEModel.forward."
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention（共享层，含 q_norm/k_norm）
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected（路由到指定 expert）
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routing_index=routing_index)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3PLEModel(Qwen3Model):
    """骨干模型：forward 开头确定 `routing_index: LongTensor[B]`，透传给每层。

    routing_index 来源（按优先级）：
    1. kwargs 中显式传入（generate 阶段由 prepare_inputs_for_generation 缓存；用户也可手动传）
    2. 自动从 input_ids 中检测（训练 / generate 首步走这里）
    3. 兜底：torch.full([B], config.default_routing_index)（仅 input_ids 为 None 时）
    """

    def __init__(self, config: Qwen3PLEConfig):
        super().__init__(config)
        # 将父类创建的 Qwen3DecoderLayer 替换为 Qwen3PLEDecoderLayer
        self.layers = nn.ModuleList(
            [Qwen3PLEDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # ===== 路由决策 =====
        # 优先级 1：从 kwargs 中取出已缓存的 routing_index (LongTensor[B] 或 int)
        routing_index = kwargs.pop("routing_index", None)

        # 优先级 2：从 input_ids 中的控制 token 自动检测
        if routing_index is None and input_ids is not None:
            routing_index = _determine_routing_index(input_ids, self.config)
        elif routing_index is not None and not isinstance(routing_index, torch.Tensor):
            # 兼容：用户传 int / list / 其它非 tensor，转成 LongTensor
            device = input_ids.device if input_ids is not None else (
                inputs_embeds.device if inputs_embeds is not None else "cpu"
            )
            if isinstance(routing_index, int):
                B = (input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0])
                routing_index = torch.full((B,), routing_index, dtype=torch.long, device=device)
            else:
                routing_index = torch.tensor(routing_index, dtype=torch.long, device=device)

        # 优先级 3：兜底默认值（仅 input_ids 为 None 且未显式指定时触发）
        if routing_index is None:
            default_idx = getattr(self.config, "default_routing_index", 0)
            B = inputs_embeds.shape[0] if inputs_embeds is not None else 1
            device = inputs_embeds.device if inputs_embeds is not None else "cpu"
            routing_index = torch.full((B,), default_idx, dtype=torch.long, device=device)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                routing_index=routing_index,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3PLEForCausalLM(Qwen3ForCausalLM):
    """Causal LM 顶层：使用 Qwen3PLEModel 作为 backbone。

    重写 prepare_inputs_for_generation 以在 generate 阶段正确传递路由决策：
    - 若用户通过 `generate(routing_index=N)` 显式指定（int / list / tensor），则始终使用该值
    - 否则从 input_ids 中的控制 token 自动检测
    - 结果缓存到 `self._cached_routing_index` (LongTensor[B]) 供调试/测试检查

    `_cached_routing_index` 类型为 `LongTensor[B]`（Phase 1 时 B=1，Phase 2 时 B 可>1），
    签名不随 Phase 切换变化。
    """

    def __init__(self, config: Qwen3PLEConfig):
        super().__init__(config)
        self.model = Qwen3PLEModel(config)
        # generate 过程中缓存路由决策，保证 prefill 与后续 decode 步骤路由一致
        self._cached_routing_index: Optional[torch.LongTensor] = None

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        routing_index=None,
        **kwargs,
    ):
        """重写以在 generate 阶段正确传递路由决策。

        路由优先级：
        1. 用户通过 `generate(routing_index=N)` 显式传入（int / list / Tensor）→ 转 LongTensor[B] 使用
        2. 未传入 → 从完整 input_ids 中自动检测
        3. 检测 NO_MATCH 时按 config.strict_routing_match 行为

        关键：在 generate 自回归循环中，传入此方法的 input_ids 始终是**完整的累积序列**
        （原始 prompt + 已生成 token）。但由于 `routing_index` 通过 model_kwargs 传递、
        cache_dependant_input_preparation 不裁剪 kwargs，**首步缓存后会保留到所有步骤**。
        """
        device = input_ids.device
        B = input_ids.shape[0]

        if routing_index is not None:
            # 用户显式指定。转成 LongTensor[B]。
            if isinstance(routing_index, int):
                self._cached_routing_index = torch.full(
                    (B,), routing_index, dtype=torch.long, device=device
                )
            elif isinstance(routing_index, torch.Tensor):
                self._cached_routing_index = routing_index.to(dtype=torch.long, device=device)
            else:
                self._cached_routing_index = torch.tensor(
                    routing_index, dtype=torch.long, device=device
                )
        else:
            # 从完整 input_ids 中自动检测控制 token
            self._cached_routing_index = _determine_routing_index(input_ids, self.config)

        # 调用父类的 prepare_inputs_for_generation 获取标准 model_inputs
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        # 注入 routing_index，使其透传到 Qwen3PLEModel.forward 的 kwargs 中
        model_inputs["routing_index"] = self._cached_routing_index
        return model_inputs


__all__ = [
    "Qwen3PLEExpertMLP",
    "Qwen3PLEMLP",
    "Qwen3PLEDecoderLayer",
    "Qwen3PLEModel",
    "Qwen3PLEForCausalLM",
]
