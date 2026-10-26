import sys
from webai2api.gateway import _split_sse_events, _chunk_has_payload, _event_has_payload


def test_split_multi_event_chunk():
    """上游一个 chunk 里挤多条 data 事件时，必须全部拆出来（不能只取第一条）。"""
    chunk = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        '"function":{"name":"write","arguments":"{}"}}]}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
    )
    assert len(_split_sse_events(chunk)) == 3
    assert _chunk_has_payload(chunk) is True


def test_chunk_without_payload():
    chunk = 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
    assert _split_sse_events(chunk) == [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}'
    ]
    assert _chunk_has_payload(chunk) is False


def test_split_multiline_data():
    chunk = 'data: {"a": 1}\ndata: {"b": 2}\n'
    out = _split_sse_events(chunk)
    assert out[0] == 'data: {"a": 1}'
    assert out[1] == 'data: {"b": 2}'
