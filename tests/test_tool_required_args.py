"""工具调用必填参数校验（2026-10-28 用户报障）。

症状：agent 客户端里频繁出现
    Invalid arguments for tool "shell": - command: Missing key
    Arguments provided: {}

根因：网页版模型没有原生 function calling，工具调用靠提示词注入 + DSML XML 解析。
reasoner 这类模型常先吐 <invoke name="shell"> 骨架再填参数，一旦在填参数前停住
（思考被打断 / 上游截断 / 续写没接上），就会留下一个空 invoke。
旧解析器只要拿到工具名就生成调用、arguments 传 "{}"，客户端校验必填项失败，
再让模型重来一轮——于是出现「思考→报错→再思考」的循环。

修法：按工具定义的 required 校验，参数缺失或为空就不生成调用（丢弃本次），
同时提示词里把「缺必填参数=无效」写成硬规则。
"""

from __future__ import annotations

import json

from webai2api.adapters.deepseek.services.tool_calling import (
    _build_tool_prompt_impl,
    parse_tool_calls,
)

D = "\uff5c\uff5cDSML\uff5c\uff5c"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    }
]


def _dsml(inner: str) -> str:
    return f"<{D} calls>{inner}</{D} calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{D} invoke name="{name}">{params}</{D} invoke>'


def _param(name: str, value: str, string: bool = True) -> str:
    flag = 'string="true"' if string else 'string="false"'
    return f'<{D} parameter name="{name}" {flag}>{value}</{D} parameter>'


def test_complete_call_is_kept():
    r = parse_tool_calls(_dsml(_invoke("shell", _param("command", "ls -la"))), TOOLS)
    assert len(r.calls) == 1
    assert json.loads(r.calls[0].arguments) == {"command": "ls -la"}


def test_empty_invoke_is_discarded():
    """模型只吐了 invoke 骨架、没写参数 → 不能生成调用。"""
    r = parse_tool_calls(_dsml(_invoke("shell")), TOOLS)
    assert r.calls == [], "空参数调用必须丢弃，否则客户端报 Missing key"


def test_blank_required_value_is_discarded():
    r = parse_tool_calls(_dsml(_invoke("shell", _param("command", ""))), TOOLS)
    assert r.calls == []


def test_missing_one_of_multiple_required_is_discarded():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["path", "text"],
                },
            },
        }
    ]
    r = parse_tool_calls(_dsml(_invoke("edit", _param("path", "a.py"))), tools)
    assert r.calls == []


def test_optional_params_do_not_block():
    r = parse_tool_calls(_dsml(_invoke("shell", _param("command", "ls"))), TOOLS)
    assert len(r.calls) == 1, "未声明的可选参数缺失不应拦截"


def test_unknown_tool_or_schema_is_not_blocked():
    """定义里没有 required → 一律放行，避免误杀能跑通的调用。

    （未知工具名由 strict_names 白名单逻辑负责，不在本校验范围内。）
    """
    tools_no_req = [
        {
            "type": "function",
            "function": {"name": "ping", "parameters": {"type": "object"}},
        }
    ]
    r2 = parse_tool_calls(_dsml(_invoke("ping")), tools_no_req)
    assert len(r2.calls) == 1


def test_prompt_forbids_incomplete_calls():
    prompt = _build_tool_prompt_impl(TOOLS)
    assert "MISSING any required argument is INVALID" in prompt
    assert "never emit an empty" in prompt
