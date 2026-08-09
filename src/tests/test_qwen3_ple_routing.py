# coding=utf-8
# Copyright 2026 Hybrid-Expert-Thinking Project.
"""Qwen3 PLE 路由检测规则单元测试。

覆盖：
- Branch ① (有锚点) / Branch ② (无锚点裸推理) 两种 scan 路径
- last-occurrence-wins 语义
- regex negative-lookbehind 排除 `</think>` 闭合标签
- regex negative-lookahead 排除 `/think_hard` 类子串
- safe-mode 变量：NO_MATCH 时 strict raise vs lenient default
- F1 (KV cache)：generate smoke test，GPU 可达时跑
- F4 config 向后兼容：缺 `strict_routing_match` 时 getattr 默认 True
- F5：所有 import 走 `src.models.qwen3_ple`（生产路径）

跑法：python -m pytest src/tests/test_qwen3_ple_routing.py -v
"""

import os
import sys

import pytest
import torch

# 确保从仓库根导入 src.models.qwen3_ple（生产路径，对应 plan F5）
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoTokenizer  # noqa: E402

from src.models.qwen3_ple.configuration_qwen3_ple import Qwen3PLEConfig  # noqa: E402
from src.models.qwen3_ple.modeling_qwen3_ple import (  # noqa: E402
    _ANCHOR_BIGRAM,
    _detect_routing_for_row,
    _determine_routing_index,
    _find_last_bigram,
)


CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "models/qwen3_4b_ple_initialized")


# ---------------------- fixtures ----------------------

@pytest.fixture(scope="module")
def tokenizer():
    """V2 tokenizer (qwen3 native BPE, 无注册控制 token)。"""
    return AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)


@pytest.fixture
def strict_config():
    """safe-mode 默认开启：NO_MATCH 时 raise。"""
    cfg = Qwen3PLEConfig(strict_routing_match=True, default_routing_index=0)
    cfg._name_or_path = CHECKPOINT_DIR  # _get_routing_tokenizer 用得到
    return cfg


@pytest.fixture
def lenient_config():
    """safe-mode 关闭：NO_MATCH 时退默认 + warning。"""
    cfg = Qwen3PLEConfig(strict_routing_match=False, default_routing_index=0)
    cfg._name_or_path = CHECKPOINT_DIR
    return cfg


def _ids_batch(tokenizer, text: str) -> torch.LongTensor:
    """把单条文本 encode 成 [1, S] LongTensor。"""
    return torch.tensor([tokenizer.encode(text, add_special_tokens=False)], dtype=torch.long)


# ---------------------- 基础 helper ----------------------

class TestFindLastBigram:
    """`_find_last_bigram` 单元测试。"""

    def test_finds_last_occurrence(self):
        # 序列里有 3 个 (1,2) bigram；应取最后一个
        ids = torch.tensor([1, 2, 3, 1, 2, 4, 1, 2, 5], dtype=torch.long)
        assert _find_last_bigram(ids, (1, 2)) == 6

    def test_not_found(self):
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        assert _find_last_bigram(ids, (5, 6)) is None

    def test_too_short(self):
        ids = torch.tensor([1], dtype=torch.long)
        assert _find_last_bigram(ids, (1, 2)) is None

    def test_anchor_bigram_constant(self):
        # 锚点是 (151644=<|im_start|>, 77091=assistant)
        assert _ANCHOR_BIGRAM == (151644, 77091)


# ---------------------- 路由检测主体测试 ----------------------

class TestRoutingBranch1WithAnchor:
    """Branch ①：套了 chat template、锚点存在。"""

    def test_think_prompt(self, tokenizer, strict_config):
        prompt = "<|im_start|>user\nWhat is 2+2? /think<|im_end|>\n<|im_start|>assistant\n"
        result = _determine_routing_index(_ids_batch(tokenizer, prompt), strict_config, tokenizer)
        assert result.shape == (1,)
        assert result.dtype == torch.long
        assert int(result[0]) == 1, f"/think 应路由 1，得到 {int(result[0])}"

    def test_no_think_prompt(self, tokenizer, strict_config):
        prompt = "<|im_start|>user\nWhat is 2+2? /no_think<|im_end|>\n<|im_start|>assistant\n"
        assert int(_determine_routing_index(_ids_batch(tokenizer, prompt), strict_config, tokenizer)[0]) == 0

    def test_no_think_with_close_tag_in_response(self, tokenizer, strict_config):
        """关键测试：no_think 完整训练序列。Response 含 `<think></think>` 占位符（也是 V2 灾难根因），
        decode 出 `/think` 子串。锚点 + regex negative-lookbehind 应正确排除，路由仍 no_think (0)。"""
        full = (
            "<|im_start|>user\nQuestion /no_think<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n</think>\nAnswer<|im_end|>"
        )
        result = _determine_routing_index(_ids_batch(tokenizer, full), strict_config, tokenizer)
        assert int(result[0]) == 0, "no_think 训练样本不应被 </think> 误匹配"

    def test_think_with_full_response(self, tokenizer, strict_config):
        full = (
            "<|im_start|>user\nQ /think<|im_end|>\n"
            "<|im_start|>assistant\n<think>reasoning here</think>Final answer<|im_end|>"
        )
        assert int(_determine_routing_index(_ids_batch(tokenizer, full), strict_config, tokenizer)[0]) == 1

    def test_last_wins_in_prompt(self, tokenizer, strict_config):
        """prompt 里出现多次控制词，取最后一个。"""
        prompt = (
            "<|im_start|>user\nFirst I said /no_think but actually /think<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        assert int(_determine_routing_index(_ids_batch(tokenizer, prompt), strict_config, tokenizer)[0]) == 1

    def test_substring_not_matched(self, tokenizer, strict_config):
        """`/think_hard` 含 `/think` 子串，但 regex `(?!\\w)` 应阻止误匹配。
        最后真实控制词是 /no_think → 路由 0。"""
        prompt = (
            "<|im_start|>user\nExplain /think_hard mode briefly. /no_think<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        assert int(_determine_routing_index(_ids_batch(tokenizer, prompt), strict_config, tokenizer)[0]) == 0


class TestRoutingBranch2NoAnchor:
    """Branch ②：裸推理无 chat template。"""

    def test_bare_think(self, tokenizer, strict_config):
        text = "What is 2+2? /think"
        assert int(_determine_routing_index(_ids_batch(tokenizer, text), strict_config, tokenizer)[0]) == 1

    def test_bare_no_think(self, tokenizer, strict_config):
        text = "Solve: 5*7 /no_think"
        assert int(_determine_routing_index(_ids_batch(tokenizer, text), strict_config, tokenizer)[0]) == 0

    def test_bare_with_close_tag_injection(self, tokenizer, strict_config):
        """裸推理用户字面注入 `</think>`。regex 应排除，取末尾真控制词。"""
        text = "Now consider </think> closing tag patterns. Final: /think"
        assert int(_determine_routing_index(_ids_batch(tokenizer, text), strict_config, tokenizer)[0]) == 1


class TestNoMatchBehavior:
    """NO_MATCH (没扫到 /think 或 /no_think) 行为 by `strict_routing_match` 变量控制。"""

    def test_strict_raises(self, tokenizer, strict_config):
        prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        with pytest.raises(ValueError, match="not found"):
            _determine_routing_index(_ids_batch(tokenizer, prompt), strict_config, tokenizer)

    def test_lenient_falls_back_to_default(self, tokenizer, lenient_config, caplog):
        prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        result = _determine_routing_index(_ids_batch(tokenizer, prompt), lenient_config, tokenizer)
        assert int(result[0]) == 0  # default_routing_index

    def test_lenient_respects_custom_default(self, tokenizer):
        # default=1 时退到 think
        cfg = Qwen3PLEConfig(strict_routing_match=False, default_routing_index=1)
        cfg._name_or_path = CHECKPOINT_DIR
        prompt = "<|im_start|>user\nNo control word here<|im_end|>\n<|im_start|>assistant\n"
        result = _determine_routing_index(_ids_batch(tokenizer, prompt), cfg, tokenizer)
        assert int(result[0]) == 1


class TestConfigBackwardCompat:
    """F4: 旧 checkpoint config.json 缺 strict_routing_match 字段时 getattr 默认 True (raise)。"""

    def test_missing_field_defaults_to_strict_true(self, tokenizer):
        # 用 Qwen3PLEConfig 但绕开 __init__ 来模拟"老 config 没这字段"
        cfg = Qwen3PLEConfig()
        cfg._name_or_path = CHECKPOINT_DIR
        # 主动删字段模拟旧 checkpoint
        if hasattr(cfg, "strict_routing_match"):
            del cfg.strict_routing_match
        prompt = "<|im_start|>user\nNo control word<|im_end|>\n<|im_start|>assistant\n"
        # getattr 默认 True → 应 raise
        with pytest.raises(ValueError, match="not found"):
            _determine_routing_index(_ids_batch(tokenizer, prompt), cfg, tokenizer)


class TestTokenizerDecodeRoundtrip:
    """H3 防御：tokenizer.decode 必须把控制词 token 序列还原成字面 /think / /no_think。
    若 tokenizer 版本漂移导致 decode 插入额外字符（如 ` /think`），regex 会失配 → NO_MATCH。"""

    def test_think_encoding_round_trips(self, tokenizer):
        # /think 在常见上下文里至少一种编码
        for variant_text in ["/think", " /think", "\\boxed{}./think"]:
            ids = tokenizer.encode(variant_text, add_special_tokens=False)
            decoded = tokenizer.decode(ids)
            assert "/think" in decoded, (
                f"decode round-trip 失真：encode({variant_text!r}) -> {ids} -> decode = {decoded!r}; "
                "字面 /think 不在结果中。regex 会失配。"
            )

    def test_no_think_encoding_round_trips(self, tokenizer):
        for variant_text in ["/no_think", " /no_think", "\\boxed{}./no_think"]:
            ids = tokenizer.encode(variant_text, add_special_tokens=False)
            decoded = tokenizer.decode(ids)
            assert "/no_think" in decoded, (
                f"decode round-trip 失真：encode({variant_text!r}) -> {ids} -> decode = {decoded!r}; "
                "字面 /no_think 不在结果中。regex 会失配。"
            )


class TestLazyTokenizerLoad:
    """B1 回归防御：训练入口不传 tokenizer 参数，走 `_get_routing_tokenizer` lazy load。
    曾因 modular_converter 漏掉 `_TOKENIZER_CACHE` 模块级定义而 NameError 崩。
    本测试故意不传 tokenizer，触发 lazy load 路径。"""

    def test_lazy_load_via_get_routing_tokenizer(self, strict_config):
        """不传 tokenizer 参数；走 _get_routing_tokenizer(config) → AutoTokenizer.from_pretrained。
        config._name_or_path 已在 strict_config fixture 设好指向 checkpoint。"""
        # 自建 input_ids，不依赖 tokenizer fixture
        # 一个含 /think 的简短序列（手动构造避免循环依赖）
        # 用 evalscope/V2 已知的 /think tokenization：[20439, 766]
        ids = torch.tensor([[20439, 766]], dtype=torch.long)
        # 关键：不传 tokenizer，验证 lazy load 不 NameError
        result = _determine_routing_index(ids, strict_config)  # tokenizer=None
        assert result.shape == (1,)
        assert int(result[0]) == 1

    def test_tokenizer_cache_module_variable_exists(self):
        """硬断言 `_TOKENIZER_CACHE` 是模块级可访问的 dict。
        是 B1 回归（converter 漏定义）的最小金丝雀。"""
        from src.models.qwen3_ple import modeling_qwen3_ple
        assert hasattr(modeling_qwen3_ple, "_TOKENIZER_CACHE"), \
            "_TOKENIZER_CACHE module-level variable missing — modular converter regression?"
        assert isinstance(modeling_qwen3_ple._TOKENIZER_CACHE, dict)


class TestBatchHandling:
    """Phase 2 起 batch>1 路径全部打通。
    _determine_routing_index 独立 per-row 计算返回 LongTensor[B]；
    Qwen3PLEMLP.forward 用 Scheme A 同时算两个 expert + per-sample mask 选择。"""

    def test_returns_longtensor_shape_B(self, tokenizer, strict_config):
        prompt = "<|im_start|>user\nQ /think<|im_end|>\n<|im_start|>assistant\n"
        ids = _ids_batch(tokenizer, prompt)
        result = _determine_routing_index(ids, strict_config, tokenizer)
        assert result.shape == (1,)
        assert result.dtype == torch.long

    def test_batched_input_B2_routing(self, tokenizer, strict_config):
        """B=2 batch 路由检测：think + no_think 混合各自路由正确。"""
        p1 = "<|im_start|>user\nQ /think<|im_end|>\n<|im_start|>assistant\n"
        p2 = "<|im_start|>user\nA /no_think<|im_end|>\n<|im_start|>assistant\n"
        ids1 = _ids_batch(tokenizer, p1)
        ids2 = _ids_batch(tokenizer, p2)
        max_len = max(ids1.shape[1], ids2.shape[1])
        pad_id = tokenizer.pad_token_id or 151643
        if ids1.shape[1] < max_len:
            ids1 = torch.cat([ids1, torch.full((1, max_len - ids1.shape[1]), pad_id, dtype=torch.long)], dim=1)
        if ids2.shape[1] < max_len:
            ids2 = torch.cat([ids2, torch.full((1, max_len - ids2.shape[1]), pad_id, dtype=torch.long)], dim=1)
        batched = torch.cat([ids1, ids2], dim=0)
        result = _determine_routing_index(batched, strict_config, tokenizer)
        assert result.shape == (2,)
        assert int(result[0]) == 1
        assert int(result[1]) == 0


class TestPhase2BatchDispatch:
    """Phase 2: Qwen3PLEMLP per-sample mask dispatch（Scheme A）的正确性测试。
    构造小 MLP 实例 (CPU，无需载真权重) 验证：
    - batch=1 行为与 Phase 1 等价（回归保护）
    - batch>1 mixed routing 每个样本走对 expert
    - 梯度仅在被选中的 expert 上累积（mask=0 的 expert 在 mask=0 那行收到 0 梯度）
    """

    @pytest.fixture
    def tiny_mlp(self):
        """构造小尺寸 MLP；experts[0] 全 0 输出（act_fn(0)*x = 0），experts[1] 显著输出。
        通过设置 gate/up 权重 0 vs 1 让两个 expert 输出可区分。"""
        from src.models.qwen3_ple.modeling_qwen3_ple import Qwen3PLEMLP

        # 极小 config 够初始化 MLP
        cfg = Qwen3PLEConfig(
            hidden_size=8,
            intermediate_size=16,
            hidden_act="silu",
        )
        mlp = Qwen3PLEMLP(cfg)
        # expert 0：gate/up/down 全置 0 → 输出全 0
        with torch.no_grad():
            for p in mlp.experts[0].parameters():
                p.zero_()
            # expert 1: 初始化保持默认随机；只要输出不全 0 就能区分
        mlp.eval()
        return mlp

    def test_dispatch_batch_1_think(self, tiny_mlp):
        """B=1, routing=1 → expert 1 输出，非 0。"""
        h = torch.randn(1, 4, 8)  # [B=1, S=4, H=8]
        routing = torch.tensor([1], dtype=torch.long)
        out = tiny_mlp(h, routing)
        assert out.shape == h.shape
        # expert 1 随机权重 → 输出应非全 0
        assert out.abs().sum().item() > 0

    def test_dispatch_batch_1_no_think(self, tiny_mlp):
        """B=1, routing=0 → expert 0（零权重）输出，全 0。"""
        h = torch.randn(1, 4, 8)
        routing = torch.tensor([0], dtype=torch.long)
        out = tiny_mlp(h, routing)
        assert out.shape == h.shape
        # expert 0 权重全 0 → 输出全 0
        assert out.abs().sum().item() == 0.0

    def test_dispatch_mixed_batch_B2(self, tiny_mlp):
        """B=2 mixed: routing=[0, 1] → sample 0 走 expert 0 (全 0), sample 1 走 expert 1 (非 0)。"""
        h = torch.randn(2, 4, 8)
        routing = torch.tensor([0, 1], dtype=torch.long)
        out = tiny_mlp(h, routing)
        assert out.shape == h.shape
        # sample 0 走 expert 0 → 全 0
        assert out[0].abs().sum().item() == 0.0, "sample 0 should route to expert 0 (zero weights)"
        # sample 1 走 expert 1 → 非 0
        assert out[1].abs().sum().item() > 0, "sample 1 should route to expert 1 (random weights)"

    def test_dispatch_mixed_batch_B2_reverse(self, tiny_mlp):
        """B=2 mixed: routing=[1, 0] (反向) → sample 0 非 0, sample 1 全 0。"""
        h = torch.randn(2, 4, 8)
        routing = torch.tensor([1, 0], dtype=torch.long)
        out = tiny_mlp(h, routing)
        assert out[0].abs().sum().item() > 0
        assert out[1].abs().sum().item() == 0.0

    def test_dispatch_batch_4_mixed(self, tiny_mlp):
        """B=4 复杂混合 routing=[1, 0, 1, 0]。"""
        h = torch.randn(4, 4, 8)
        routing = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        out = tiny_mlp(h, routing)
        # routing=1 的样本（0, 2）应有非 0 输出；routing=0 的样本（1, 3）应全 0
        assert out[0].abs().sum().item() > 0
        assert out[1].abs().sum().item() == 0.0
        assert out[2].abs().sum().item() > 0
        assert out[3].abs().sum().item() == 0.0

    def test_dispatch_gradient_isolation(self, tiny_mlp):
        """关键测试：sample 走 expert 1 时，expert 0 的 .grad 不应被该样本污染。
        Scheme A mask 应让被未选中 expert 的梯度精确为 0（per-sample 级）。"""
        tiny_mlp.train()
        h = torch.randn(2, 4, 8, requires_grad=False)
        routing = torch.tensor([1, 1], dtype=torch.long)  # 两个样本都走 expert 1
        out = tiny_mlp(h, routing)
        loss = out.sum()
        loss.backward()
        # expert 0 完全没参与（mask=0），其 .grad 应全 0
        for name, p in tiny_mlp.experts[0].named_parameters():
            assert p.grad is not None, f"expert 0 {name}.grad missing"
            assert p.grad.abs().sum().item() == 0.0, (
                f"expert 0 {name}.grad should be 0 when no sample routes to it; "
                f"got sum={p.grad.abs().sum().item()}"
            )
        # expert 1 应有非 0 梯度
        any_nonzero = False
        for name, p in tiny_mlp.experts[1].named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                any_nonzero = True
                break
        assert any_nonzero, "expert 1 should have non-zero gradients"

    def test_dispatch_mask_dtype_bf16_compatible(self, tiny_mlp):
        """mask 转 out0.dtype，验证 bf16 inference 路径不挂。"""
        tiny_mlp.eval()
        # 把 MLP 转 bf16 模拟训练/推理 dtype
        tiny_mlp.bfloat16()
        h = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        routing = torch.tensor([0, 1], dtype=torch.long)
        out = tiny_mlp(h, routing)
        assert out.dtype == torch.bfloat16
        assert out.shape == h.shape


# ---------------------- F1 KV cache + A3 generate smoke ----------------------
# 跑实模型 forward + generate。CPU 可跑但慢；GPU 上几秒钟。

@pytest.mark.slow
class TestGenerateSmoke:
    """F1: KV cache 参数名 + A3: generate vs prefill decode 一致性。"""

    def test_load_and_generate_5_tokens(self, tokenizer):
        """加载 checkpoint + generate 5 token，验证 KV cache 不静默失效。"""
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT_DIR, trust_remote_code=True, torch_dtype=dtype
        ).to(device).eval()

        prompt = "<|im_start|>user\nWhat is 2+2? /think<|im_end|>\n<|im_start|>assistant\n"
        ids = _ids_batch(tokenizer, prompt).to(device)

        with torch.no_grad():
            output = model.generate(ids, max_new_tokens=5, do_sample=False)

        # 至少生成了 5 个新 token
        assert output.shape[1] >= ids.shape[1] + 5

        # routing index 被缓存且取值正确（A3：generate 循环中没被错误检测覆盖）
        assert model._cached_routing_index is not None
        assert model._cached_routing_index.shape == (1,)
        assert int(model._cached_routing_index[0]) == 1, "/think → expert 1"


class TestSchemeBInference:
    """Scheme B（推理路径，torch.no_grad 下触发）：只算被选中 expert，且与 Scheme A bit-identical。"""

    @pytest.fixture
    def tiny_mlp(self):
        from src.models.qwen3_ple.modeling_qwen3_ple import Qwen3PLEMLP
        cfg = Qwen3PLEConfig(hidden_size=8, intermediate_size=16, hidden_act="silu")
        mlp = Qwen3PLEMLP(cfg)
        # 让两个 expert 权重不同，才能区分"走哪个"
        with torch.no_grad():
            for p in mlp.experts[0].parameters():
                p.zero_()  # expert 0 输出全 0
            # expert 1 保持随机
        mlp.eval()
        return mlp

    def test_scheme_b_homogeneous_no_think(self, tiny_mlp):
        """no_grad + 全 routing=0 → 只调 expert 0（零权重）→ 输出全 0。"""
        h = torch.randn(3, 4, 8)
        routing = torch.tensor([0, 0, 0], dtype=torch.long)
        with torch.no_grad():
            out = tiny_mlp(h, routing)
        assert out.shape == h.shape
        assert out.abs().sum().item() == 0.0

    def test_scheme_b_homogeneous_think(self, tiny_mlp):
        """no_grad + 全 routing=1 → 只调 expert 1（随机权重）→ 输出非 0。"""
        h = torch.randn(3, 4, 8)
        routing = torch.tensor([1, 1, 1], dtype=torch.long)
        with torch.no_grad():
            out = tiny_mlp(h, routing)
        assert out.abs().sum().item() > 0

    def test_scheme_b_mixed_batch(self, tiny_mlp):
        """no_grad + routing=[0,1,0,1] → 分段 dispatch；sample 0/2 走 expert0(零)、1/3 走 expert1(非零)。"""
        h = torch.randn(4, 4, 8)
        routing = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        with torch.no_grad():
            out = tiny_mlp(h, routing)
        assert out[0].abs().sum().item() == 0.0
        assert out[1].abs().sum().item() > 0
        assert out[2].abs().sum().item() == 0.0
        assert out[3].abs().sum().item() > 0

    def test_scheme_a_equals_scheme_b(self, tiny_mlp):
        """关键：同样输入，Scheme A（grad on）与 Scheme B（no_grad）输出 bit-identical。
        证明推理切到 Scheme B 不改变结果、只省算力。"""
        h = torch.randn(4, 4, 8)
        routing = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        # Scheme A: grad enabled (default)
        out_a = tiny_mlp(h, routing)
        # Scheme B: no_grad
        with torch.no_grad():
            out_b = tiny_mlp(h, routing)
        assert torch.equal(out_a, out_b), (
            f"Scheme A vs B mismatch: max diff = {(out_a - out_b).abs().max().item()}"
        )

    def test_scheme_b_bf16(self, tiny_mlp):
        """no_grad + bf16 推理路径不挂。"""
        tiny_mlp.bfloat16()
        h = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        routing = torch.tensor([0, 1], dtype=torch.long)
        with torch.no_grad():
            out = tiny_mlp(h, routing)
        assert out.dtype == torch.bfloat16
        assert out.shape == h.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
