"""工具协议提示词的身份与环境锚定回归（GLM / qwen 新增，kimi 防回归）。

背景：kimi 适配器曾把文件写到 sandbox:///mnt/ 虚拟路径并自称「网页版
沙盒」，当时的修复是在 kimi 工具协议提示词开头加【身份与环境】强锚定块。
GLM / qwen 此前从未加同款锚定，导致模型经 ZCode / OpenCode 使用时张口
闭口「沙盒是隔离环境，访问不到本地 D 盘」。本测试锁定：

- 三家构建出的工具协议提示词必须以锚定块开头，且含关键句（身份与环境 /
  不在任何云端 / 本地路径 / 禁止 sandbox:// 虚拟路径）；
- 原有调用格式示例不被锚定块破坏（GLM 单行 JSON 标记、qwen Hermes 标签、
  kimi DSML）；
- GLM 会话复用路径的单行观测日志（轮数 / 累计字符量）。
"""

from __future__ import annotations

import logging

from webai2api.adapters.chatglm import client as glm_client
from webai2api.adapters.chatglm.client import ChatGlmWebClient
from webai2api.adapters.chatglm.config import GlmConfig
from webai2api.adapters.kimi import client as kimi_client
from webai2api.adapters.qwen import client as qwen_client

ANCHOR_HEAD = "【身份与环境（最重要的前提，先读这条）】"
# 锚定块关键句：身份定位、云端/沙盒否认、本地路径口径、虚拟路径禁令
ANCHOR_KEYS = ("身份与环境", "不在任何云端", "本地路径", "/mnt/agents", "sandbox://")

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "写本地文件",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filePath", "content"],
        },
    },
}


def _glm_protocol() -> str:
    """按生产同款方式渲染 GLM 工具协议提示词（清单 + 锚定 + 格式规则）。"""
    return glm_client.TOOL_INSTRUCTION.format(
        tools=glm_client._tools_prompt([WRITE_TOOL]),
        marker=glm_client.TOOL_MARKER,
    )


def _qwen_protocol() -> str:
    return qwen_client.QWEN_TOOL_INSTRUCTION.format(
        tools=qwen_client._tools_prompt([WRITE_TOOL])
    )


# ── GLM：DSML/单行 JSON 标记格式 + 锚定块 ────────────────────
def test_glm_tool_prompt_starts_with_identity_anchor():
    p = _glm_protocol()
    assert p.lstrip().startswith(ANCHOR_HEAD), "GLM 工具提示词未以锚定块开头"
    for key in ANCHOR_KEYS:
        assert key in p, f"GLM 工具提示词缺锚定关键句: {key}"
    # 原有单行 JSON 标记格式示例必须原样保留（不被锚定块破坏）
    assert glm_client.TOOL_MARKER in p
    assert '<<<TOOL_CALL>>>{"name":"工具名","arguments":{"参数名":值}}' in p
    # 规则正文与工具清单占位渲染仍在
    assert "你可以调用以下工具来完成任务" in p
    assert "write" in p and "写本地文件" in p


def test_glm_inject_tools_carries_anchor_via_system():
    """生产注入路径（_inject_tools）合并 system 后锚定块仍在前部。"""
    msgs = glm_client._inject_tools(
        [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "hi"},
        ],
        [WRITE_TOOL],
    )
    assert msgs[0]["role"] == "system"
    content = msgs[0]["content"]
    assert ANCHOR_HEAD in content
    for key in ANCHOR_KEYS:
        assert key in content
    assert glm_client.TOOL_MARKER in content


# ── qwen：Hermes 标签格式 + 锚定块 ───────────────────────────
def test_qwen_tool_prompt_starts_with_identity_anchor():
    p = _qwen_protocol()
    assert p.lstrip().startswith(ANCHOR_HEAD), "qwen 工具提示词未以锚定块开头"
    for key in ANCHOR_KEYS:
        assert key in p, f"qwen 工具提示词缺锚定关键句: {key}"
    # 原有 Hermes 标签格式示例必须原样保留
    assert "<tool_call>" in p and "</tool_call>" in p
    assert "<tool_response>" in p
    assert '{"name": "工具名", "arguments": {"参数名": "值"}}' in p
    assert "# Tools" in p and "write" in p


def test_qwen_inject_paths_carry_anchor():
    """中继（_inject_tools_system）与直连（_inject_tools_messages）都带锚定。"""
    sys_txt = qwen_client._inject_tools_system("你是助手", [WRITE_TOOL])
    assert ANCHOR_HEAD in sys_txt
    for key in ANCHOR_KEYS:
        assert key in sys_txt
    msgs = qwen_client._inject_tools_messages(
        [{"role": "system", "content": "你是助手"}], [WRITE_TOOL]
    )
    assert msgs[0]["role"] == "system"
    assert ANCHOR_HEAD in msgs[0]["content"]
    assert "<tool_call>" in msgs[0]["content"]


# ── kimi：既有锚定防回归（DSML 格式） ────────────────────────
def test_kimi_tool_prompt_keeps_identity_anchor():
    p = kimi_client._tools_protocol([WRITE_TOOL])
    assert p.lstrip().startswith(ANCHOR_HEAD), "kimi 锚定块被意外移除/移位"
    for key in ANCHOR_KEYS:
        assert key in p
    assert "<|DSML|tool_calls>" in p
    assert "<|DSML|invoke" in p


# ── GLM 会话复用观测日志 ─────────────────────────────────────
class _FakeConn:
    def close(self):
        pass


class _FakeResp:
    def read(self):
        return b""


def test_glm_reuse_logs_turns_and_chars(caplog):
    """命中缓存复用上游会话时，输出单行观测日志（轮数 + 近似字符量）。"""
    cfg = GlmConfig(
        refresh_token="rt",
        base_url="http://127.0.0.1:1",  # _open_stream 已打桩，不会真的连接
        anti_rate_limit_enabled=False,
        session_store_path="",
        egress="direct",
    )
    c = ChatGlmWebClient(cfg, logging.getLogger("glm.anchor"))
    payload = {
        "model": "GLM-5.3-Flash",
        "messages": [
            {"role": "user", "content": "第一问：介绍一下这个项目"},
            {"role": "assistant", "content": "这是一个本地网关项目"},
            {"role": "user", "content": "第二问：继续"},
        ],
    }
    # 预置会话缓存 → 本轮命中复用路径（缓存 turns 口径 = 已见 assistant 轮数）
    c._store_conv(c._conv_key(payload), "conv-anchor-1")
    c._access = lambda: "tok"
    c._open_stream = lambda *a, **k: (_FakeConn(), _FakeResp())
    c._iter_events = lambda resp: iter(
        [
            {
                "conversation_id": "conv-anchor-1",
                "status": "finish",
                "parts": [
                    {
                        "logic_id": "a",
                        "status": "finish",
                        "content": [{"type": "text", "text": "好的"}],
                    }
                ],
            }
        ]
    )
    c._upload_image_items = lambda msgs: []
    with caplog.at_level(logging.INFO, logger="glm.anchor"):
        for ev in c._stream_events(payload):
            if ev.get("__done__"):
                break
    recs = [r for r in caplog.records if "GLM会话复用" in r.getMessage()]
    assert recs, "复用路径未输出会话观测日志"
    msg = recs[-1].getMessage()
    assert "轮数=1" in msg, f"轮数口径不符: {msg}"
    assert "累计≈" in msg and "字" in msg, f"缺累计字符量: {msg}"
    assert recs[-1].levelno == logging.INFO
