"""Kimi / 通义千问 防风控 + 路由 + snippet 接线（纯内存，不打网络）。"""
from __future__ import annotations

import json
import logging
import time

from webai2api.adapters.kimi.client import KimiWebClient, _build_headers as kimi_headers
from webai2api.adapters.kimi.config import KimiConfig, _stable_device_id as kimi_dev
from webai2api.adapters.qwen.client import QwenWebClient, _build_headers as qwen_headers
from webai2api.adapters.qwen.config import QwenConfig, _stable_device_id as qwen_dev
from webai2api.admin_api import _build_opencode_snippet
from webai2api.providers import infer_provider, DEFAULT_LIMITS


# ── provider 路由 ─────────────────────────────────────────
def test_infer_provider_prefixes() -> None:
    assert infer_provider("kimi-k3") == "kimi"
    assert infer_provider("moonshot-v1") == "kimi"
    assert infer_provider("qwen3.8-max") == "qwen"
    assert infer_provider("GLM-5.3") == "chatglm"
    assert infer_provider("deepseek-reasoner") == "deepseek"
    assert infer_provider("unknown-model") == "deepseek"


def test_limits_cover_all_providers() -> None:
    for p in ("deepseek", "chatglm", "kimi", "qwen"):
        assert p in DEFAULT_LIMITS
        assert DEFAULT_LIMITS[p]["context"] > 0


# ── 指纹稳定 ─────────────────────────────────────────────
def test_kimi_headers_stable() -> None:
    cfg = KimiConfig(auth_token="t", device_id="dev", user_agent="UA")
    h1, h2 = kimi_headers(cfg), kimi_headers(cfg)
    assert h1["Authorization"] == "Bearer t"
    assert h1["User-Agent"] == h2["User-Agent"] == "UA"
    assert h1["X-Request-Id"] != h2["X-Request-Id"]


def test_qwen_headers_stable() -> None:
    cfg = QwenConfig(auth_token="t", user_agent="UA")
    assert qwen_headers(cfg)["Authorization"] == "Bearer t"
    assert qwen_headers(cfg)["User-Agent"] == qwen_headers(cfg)["User-Agent"] == "UA"


def test_stable_device_id_isolated() -> None:
    acc = {"id": "a", "profile": '{"device_id":"from-profile"}'}
    assert kimi_dev(acc) == "from-profile"
    assert qwen_dev({"id": "x", "profile": "{}"}) != qwen_dev({"id": "y", "profile": "{}"})


# ── 会话指纹 / 缓存 / 落盘 ───────────────────────────────
def _kimi(cfg):
    return KimiWebClient(cfg, logging.getLogger("t.kimi"))


def _qwen(cfg):
    return QwenWebClient(cfg, logging.getLogger("t.qwen"))


def test_conv_key_stable_and_isolated() -> None:
    """会话指纹：多轮复用靠「首条 assistant 指纹」锚定；不同对话即使
    system 与开场白完全相同也必须不同（防跨会话串台，实测教训）。"""
    for mk, cfgcls in ((_kimi, KimiConfig), (_qwen, QwenConfig)):
        c = mk(cfgcls(auth_token="t"))
        sys = {"role": "system", "content": "s" * 300}
        u1 = {"role": "user", "content": "第一条"}
        asst1 = {"role": "assistant", "content": "好"}
        turn2 = {"model": "m", "messages": [
            sys, u1, asst1, {"role": "user", "content": "继续"}],
            "_wapi_key_id": "kA"}
        turn3 = {"model": "m", "messages": [
            sys, u1, asst1, {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "好的"},
            {"role": "user", "content": "再继续"}], "_wapi_key_id": "kA"}
        assert c._conv_key(turn2) == c._conv_key(turn3)     # 同对话多轮稳定
        # 关键防线：另一个会话同 system/开场白、但首轮回复不同 → 必须不同 key
        twin = {"model": "m", "messages": [
            sys, u1, {"role": "assistant", "content": "别的回复"},
            {"role": "user", "content": "继续"}], "_wapi_key_id": "kA"}
        assert c._conv_key(twin) != c._conv_key(turn2)
        other = dict(turn2, _wapi_key_id="kB")
        assert c._conv_key(other) != c._conv_key(turn2)      # Key 隔离


def test_conv_cache_persist_and_ttl() -> None:
    cfg = QwenConfig(auth_token="t", max_turns_per_session=2)
    c = _qwen(cfg)
    c._store_conv("k", "chat-1")
    assert c._cached_conv("k") == "chat-1"
    c._store_conv("k", "chat-1")
    assert c._cached_conv("k") == ""                        # 轮次上限

    c2 = _qwen(cfg)
    c2._store_conv("e", "chat-2")
    with c2._conv_lock:
        c2._conv_cache["e"] = ("chat-2", 0.0, 1, "")
    assert c2._cached_conv("e") == ""                        # 过期


# ── 思考档位映射（模型感知）──────────────────────────────
def test_effort_mapping_tier_aware() -> None:
    from webai2api.adapters.kimi.client import _effort_to_payload as ke, _tier_for
    from webai2api.adapters.qwen.client import (
        _effort_to_payload as qe, _qwen_tier, _supports_auto,
    )

    # Kimi 三档
    assert _tier_for("low", "kimi-k3") == "标准"
    assert _tier_for("high", "kimi-k3") == "进阶"
    assert _tier_for("max", "kimi-k3") == "极致"
    assert _tier_for("max", "kimi-k2.8") == "极致"
    assert _tier_for("max", "kimi-k2.6") == "进阶"   # K2.6 无极致档
    assert ke("")["thinking"] is False
    assert ke("high", "kimi-k3")["thinking"] is True
    assert ke("high", "kimi-k3")["reasoningEffort"] == "REASONING_EFFORT_HIGH"
    assert ke("max", "kimi-k2.6")["thinking"] is True               # 降为进阶仍思考
    assert ke("max", "kimi-k2.6")["reasoningEffort"] == "REASONING_EFFORT_HIGH"

    # Qwen：plus / max 三档（自动/思考/快速），flash 两档
    assert _supports_auto("qwen3.7-plus") is True
    assert _supports_auto("qwen3.8-max") is True
    assert _supports_auto("qwen3.8-omni-flash") is False
    assert _qwen_tier("medium", "qwen3.7-plus") == "自动"
    assert _qwen_tier("high", "qwen3.7-plus") == "思考"
    assert _qwen_tier("low", "qwen3.7-plus") == "快速"
    assert _qwen_tier("", "qwen3.7-plus") == "自动"
    # flash 无自动档 → 自动降为思考
    assert _qwen_tier("medium", "qwen3.8-omni-flash") == "思考"
    assert _qwen_tier("low", "qwen3.8-omni-flash") == "快速"
    # payload
    assert qe("medium", "qwen3.7-plus") == {"thinking_enabled": False, "auto_thinking": True}
    assert qe("high", "qwen3.7-plus") == {"thinking_enabled": True, "auto_thinking": False}
    assert qe("low", "qwen3.7-plus") == {"thinking_enabled": False, "auto_thinking": False}
    assert qe("medium", "qwen3.8-omni-flash")["thinking_enabled"] is True


def test_kimi_model_scenario_mapping() -> None:
    """回归：k3/k2.8 是 agent 模型，必须配 SCENARIO_OK_COMPUTER；
    普通对话模型走 SCENARIO_CHAT。配错会 invalid_argument（曾被误判为无权限）。"""
    from webai2api.adapters.kimi.client import _internal_models

    assert _internal_models("kimi-k3")[0] == ("k3-agent", "SCENARIO_OK_COMPUTER")
    assert _internal_models("kimi-k2.8")[0] == ("k28-agent-preview", "SCENARIO_OK_COMPUTER")
    assert _internal_models("kimi-k2.6")[0] == ("k2d6-chat", "SCENARIO_CHAT")
    # 内部 id 直接透传时也能推断出正确 scenario
    assert _internal_models("k3-agent")[0] == ("k3-agent", "SCENARIO_OK_COMPUTER")
    assert _internal_models("k28-instant")[0] == ("k28-instant", "SCENARIO_CHAT")
    # 未知模型名兜底
    assert _internal_models("whatever")[0] == ("k2d6-chat", "SCENARIO_CHAT")


def test_kimi_body_passes_model_to_tier() -> None:
    """回归：极致档必须按模型决定（K2.6 降级），确认 _body 把 model 传进映射。"""
    c = _kimi(KimiConfig(auth_token="t"))
    payload = {"model": "kimi-k2.6",
               "messages": [{"role": "user", "content": "hi"}],
               "reasoning_effort": "max"}
    body = c._body(payload, internal_model="k2d6-chat", scenario="SCENARIO_CHAT")
    assert body["options"]["model"] == "k2d6-chat"
    assert body["scenario"] == "SCENARIO_CHAT"
    assert body["options"]["thinking"] is True           # 降为进阶，仍思考
    assert body["options"]["reasoningEffort"] == "REASONING_EFFORT_HIGH"
    assert body["message"]["blocks"][0]["text"]["content"] == "hi"
    # k3 极致也应思考，且必须走 agent scenario
    p2 = {"model": "kimi-k3", "messages": payload["messages"],
          "reasoning_effort": "max"}
    body2 = c._body(p2, internal_model="k3-agent", scenario="SCENARIO_OK_COMPUTER")
    assert body2["scenario"] == "SCENARIO_OK_COMPUTER"
    assert body2["options"]["thinking"] is True
    assert body2["options"]["reasoningEffort"] == "REASONING_EFFORT_MAX"


# ── risk_stats 形状（面板依赖）────────────────────────────
def test_risk_stats_shape() -> None:
    need = {"sessions_created", "cache_hit", "cache_miss", "cache_hit_rate",
            "cache_stale", "cache_size", "active_sessions", "upstream_requests",
            "rate_limit_hits", "daily_count", "daily_limit"}
    for c in (_kimi(KimiConfig(auth_token="t")), _qwen(QwenConfig(auth_token="t"))):
        assert need.issubset(set(c.risk_stats().keys()))


# ── opencode snippet 生成（含模型感知 variants）──────────────
def test_snippet_variants_by_provider() -> None:
    snip = _build_opencode_snippet(
        "http://x", "sk-1",
        ["GLM-5.3", "kimi-k3", "kimi-k2.6", "qwen3.7-plus", "qwen3.8-max", "deepseek-chat"])
    assert '"快速"' in snip and '"极致"' in snip               # glm
    assert '"标准"' in snip and '"进阶"' in snip                # kimi 三档命名
    assert '{ "id": "极致", "settings": { "reasoningEffort": "max" } }' in snip  # V2 数组形式
    assert '"variants": {' not in snip                           # 对象 map 会被 opencode 丢弃
    assert '"思考研究"' not in snip                              # 旧档名已废弃
    assert '"自动"' in snip and '"思考"' in snip and '"快速"' in snip
    assert '"reasoning": { "supported": false }' in snip       # deepseek-chat 非思考
    assert json.loads("{\n" + snip.strip().rstrip(",") + "\n}")  # 外壳结构可解析

    # k2.6 不应带极致档；flash 三档里没有"自动"
    k26 = _build_opencode_snippet("http://x", "sk", ["kimi-k2.6"])
    assert "极致" not in k26 and "进阶" in k26
    flash = _build_opencode_snippet("http://x", "sk", ["qwen3.8-omni-flash"])
    assert "自动" not in flash and "思考" in flash and "快速" in flash
