from __future__ import annotations

import re

from webai2api.gateway import _conv_key


def _payload(msgs, model="deepseek-chat"):
    return {"model": model, "messages": msgs}


def test_system_change_keeps_key():
    a = _payload([
        {"role": "system", "content": "你是一个助手，今天是周一。"},
        {"role": "user", "content": "修复这个 bug"},
        {"role": "assistant", "content": "好的"},
    ])
    b = _payload([
        {"role": "system", "content": "你是一个助手，今天是周二，环境信息变化了。"},
        {"role": "user", "content": "修复这个 bug"},
        {"role": "assistant", "content": "好的"},
    ])
    assert _conv_key(a, "k1") == _conv_key(b, "k1")


def test_multiple_system_messages_skipped():
    a = _payload([
        {"role": "system", "content": "sys1"},
        {"role": "system", "content": "sys2"},
        {"role": "user", "content": "hello"},
    ])
    b = _payload([
        {"role": "system", "content": "完全不同的 system"},
        {"role": "user", "content": "hello"},
    ])
    assert _conv_key(a, "k1") == _conv_key(b, "k1")


def test_first_user_change_changes_key():
    a = _payload([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问题 A"},
    ])
    b = _payload([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问题 B"},
    ])
    assert _conv_key(a, "k1") != _conv_key(b, "k1")


def test_key_id_and_model_change_key():
    p = _payload([{"role": "user", "content": "hi"}])
    base = _conv_key(p, "k1")
    assert _conv_key(p, "k2") != base
    p2 = _payload([{"role": "user", "content": "hi"}], model="deepseek-reasoner")
    assert _conv_key(p2, "k1") != base


def test_returns_32_hex():
    ck = _conv_key(_payload([{"role": "user", "content": "hi"}]), "k1")
    assert len(ck) == 32
    assert re.fullmatch(r"[0-9a-f]{32}", ck)


def test_injected_key_id_field_ignored():
    p = _payload([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    before = _conv_key(p, "k1")
    p["_wapi_key_id"] = "k1"
    after = _conv_key(p, "k1")
    assert before == after
    # 换一个 _wapi_key_id 也不影响（只读 messages/model/key_id 参数）
    p["_wapi_key_id"] = "other"
    assert _conv_key(p, "k1") == before
