from webai2api.gateway import _conv_key

TOOL_CALL = {
    "id": "call_x",
    "type": "function",
    "function": {"name": "write", "arguments": "{}"},
}


def test_conv_key_differs_on_tool_call_round():
    """工具调用轮的 assistant.content 为 null：若不把 tool_calls 计入指纹，
    就会与首轮算出同一个 key，上游按首轮处理 → 丢工具结果、重发 tool_call。"""
    first = {"model": "m", "messages": [{"role": "user", "content": "写文件"}]}
    second = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "写文件"},
            {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call_x", "content": "写入成功"},
        ],
    }
    assert _conv_key(first, "k") != _conv_key(second, "k")


def test_conv_key_stable_within_same_round():
    base = [
        {"role": "user", "content": "写文件"},
        {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
    ]
    a = {
        "model": "m",
        "messages": base
        + [{"role": "tool", "tool_call_id": "call_x", "content": "ok"}],
    }
    b = {
        "model": "m",
        "messages": base
        + [{"role": "tool", "tool_call_id": "call_x", "content": "另一段结果"}],
    }
    assert _conv_key(a, "k") == _conv_key(b, "k")
