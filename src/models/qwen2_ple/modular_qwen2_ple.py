# WARNING: This file doesn't work here. It is copied from transformers/src/transformers/models/qwen2_ple/modular_qwen2_ple.py.
# Have to put it in transformers subdirectory and use transformer's converter to generate modeling_qwen2_ple.py. Otherwise, this code will not work.
# This is just an example of how to implement PLE in transformers, and show its structure.


# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2 Path-Lock Expert (PLE) model.

序列级路由：根据 control token（/think 或 /no_think）在整条序列上固定使用 expert 0 或 expert 1。
无 gate 网络，路由由控制 token 硬决定。

路由规则：
- expert 0 = no_think 模式（初始化自 Qwen2.5-7B-Instruct）
- expert 1 = think 模式（初始化自 DeepSeek-R1-Distill-Qwen-7B）
- 默认走 expert 0（no_think），因为初始化阶段共享层来自 Qwen2.5，与 expert 0 兼容
- fine-tune 完成后可根据需求修改 default_routing_index

注意：当前实现假设同一 batch 内所有序列使用相同的路由（单一 routing_index）。
若需要 batch 内混合路由（不同序列走不同 expert），需进一步扩展为 per-sample routing。
"""

from typing import Optional

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.generic import check_model_inputs
from ..qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2MLP,
)
from .configuration_qwen2_ple import Qwen2PleConfig


class Qwen2PleExpertMLP(Qwen2MLP):
    """单个 MLP 专家（与 Qwen2MLP 结构完全一致），用于 PLE 的双专家容器。"""
    pass


class Qwen2PleMLP(nn.Module):
    """Path-Lock 双专家 MLP 容器。

    包含两个独立的 MLP 专家：
    - experts[0]：no_think 专家
    - experts[1]：think 专家
    根据 routing_index 选择使用哪个专家，整条序列一致。
    """

    def __init__(self, config: Qwen2PleConfig):
        super().__init__()
        self.experts = nn.ModuleList([Qwen2PleExpertMLP(config) for _ in range(2)])

    def forward(self, hidden_states: torch.Tensor, routing_index: int = 0) -> torch.Tensor:
        return self.experts[routing_index](hidden_states)


def _determine_routing_index(input_ids: torch.LongTensor, config: Qwen2PleConfig) -> int:
    """根据 input_ids 中的控制 token 确定路由索引。

    路由决策逻辑：
    1. 若 think_token_id 和 no_think_token_id 均未配置 → 返回 default_routing_index
    2. 扫描 input_ids，找到两种控制 token 各自最后出现的位置
    3. 以最后出现的控制 token 为准（防御注入：系统模板中的控制 token 通常在用户输入之后）
    4. 若均未找到 → 返回 default_routing_index

    Args:
        input_ids: 输入 token 序列，shape [batch_size, seq_len]
        config: PLE 配置

    Returns:
        int: 0（no_think / expert 0）或 1（think / expert 1）
    """
    think_id = config.think_token_id
    no_think_id = config.no_think_token_id
    default_idx = getattr(config, "default_routing_index", 0)

    # 未配置任何控制 token → 直接返回默认值
    if think_id is None and no_think_id is None:
        return default_idx

    # 在整个 batch 的 input_ids 中查找各控制 token 最后出现的位置
    # （注意：当前假设 batch 内所有序列使用相同路由）
    last_think_pos = -1
    last_no_think_pos = -1

    if think_id is not None:
        think_mask = (input_ids == think_id)
        if think_mask.any():
            # 找到最后一个 think_token 的位置（展平后的全局索引）
            last_think_pos = think_mask.nonzero()[-1, -1].item()

    if no_think_id is not None:
        no_think_mask = (input_ids == no_think_id)
        if no_think_mask.any():
            last_no_think_pos = no_think_mask.nonzero()[-1, -1].item()

    # 两种控制 token 均未在 input_ids 中找到 → 使用默认值
    if last_think_pos == -1 and last_no_think_pos == -1:
        return default_idx

    # 以最后出现的控制 token 为准
    # 这样设计的安全性：如果用户在消息中注入了 /think，但系统模板在之后放置了 /no_think，
    # 则 /no_think 的位置更靠后，会覆盖用户的注入
    if last_think_pos > last_no_think_pos:
        return 1  # think 模式
    else:
        return 0  # no_think 模式


class Qwen2PleDecoderLayer(Qwen2DecoderLayer):
    """Decoder 层：Attention 与 Qwen2 一致，MLP 替换为 Qwen2PleMLP，forward 透传 routing_index。"""

    def __init__(self, config: Qwen2PleConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2PleMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        routing_index: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routing_index=routing_index)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2PleModel(Qwen2Model):
    """骨干模型：在 forward 开头确定 routing_index，并透传给每一层的 MLP。

    routing_index 的来源（按优先级）：
    1. kwargs 中显式传入的 routing_index（来自 generate 阶段的 prepare_inputs_for_generation 缓存，或用户手动指定）
    2. 根据 input_ids 中的控制 token 自动计算（训练阶段）
    3. 兜底使用 config.default_routing_index（默认 = 0 / no_think）
    """

    def __init__(self, config: Qwen2PleConfig):
        super().__init__(config)
        # 将父类创建的 Qwen2DecoderLayer 替换为 Qwen2PleDecoderLayer（双专家 MLP + routing_index）
        self.layers = nn.ModuleList(
            [Qwen2PleDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
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
        # 优先级 1：从 kwargs 中取出已缓存的 routing_index
        #   - generate 阶段：由 Qwen2PleForCausalLM.prepare_inputs_for_generation 在首步计算并缓存
        #   - 用户也可以手动传入 routing_index=0/1 来强制路由
        routing_index = kwargs.pop("routing_index", None)

        # 优先级 2：从 input_ids 中的控制 token 自动计算
        #   - 训练阶段：每次 forward 都会走到这里，因为训练时不经过 prepare_inputs_for_generation
        #   - generate 首步：如果 prepare_inputs_for_generation 未设置，也会走到这里
        if routing_index is None and input_ids is not None:
            routing_index = _determine_routing_index(input_ids, self.config)

        # 优先级 3：兜底默认值
        #   - 仅在 input_ids 为 None（只传了 inputs_embeds）且未显式指定 routing_index 时触发
        if routing_index is None:
            routing_index = getattr(self.config, "default_routing_index", 0)

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


class Qwen2PleForCausalLM(Qwen2ForCausalLM):
    """Causal LM 顶层：使用 Qwen2PleModel 作为 backbone。

    重写 prepare_inputs_for_generation 以在 generate 阶段正确传递路由决策：
    - 若用户通过 generate(routing_index=N) 显式指定了路由，则始终使用该值
    - 若未显式指定，则从 input_ids 中的控制 token 自动检测
    - 结果缓存到 self._cached_routing_index 供调试/测试检查
    """

    def __init__(self, config: Qwen2PleConfig):
        super().__init__(config)
        self.model = Qwen2PleModel(config)
        # 缓存 generate 过程中的 routing_index，避免后续步骤丢失路由信息
        self._cached_routing_index = None

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
        1. 用户通过 generate(routing_index=N) 显式传入 → 直接使用，不做自动检测
        2. 未传入（routing_index=None）→ 从 input_ids 中的控制 token 自动检测
        3. 自动检测也未命中 → 使用 config.default_routing_index

        关键事实：在 generate 的自回归循环中，传入此方法的 input_ids 始终是
        **完整的累积序列**（原始 prompt + 已生成的 token），而非只有新 token。
        裁剪发生在父类方法内部（cache_dependant_input_preparation）。

        注意：routing_index 参数在 generate 循环中会通过 model_kwargs 持续传递，
        所以用户在首步传入的值会自动保留到后续所有步骤。
        """
        # 路由决策：优先使用用户显式传入的值，否则自动检测
        if routing_index is not None:
            # 用户通过 generate(routing_index=N) 显式指定
            self._cached_routing_index = routing_index
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

        # 注入 routing_index，使其透传到 Qwen2PleModel.forward 的 kwargs 中
        model_inputs["routing_index"] = self._cached_routing_index
        return model_inputs


__all__ = [
    "Qwen2PleExpertMLP",
    "Qwen2PleMLP",
    "Qwen2PleDecoderLayer",
    "Qwen2PleModel",
    "Qwen2PleForCausalLM",
]
