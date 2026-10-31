"""网关粘性键的 1→2 轮过渡回归（2026-11-01）。

首轮请求没有 assistant，粘性存的是「无 assistant」键；第 2 轮客户端回传
首轮回复后键变了，精确查必 miss → 多账号池轮询换号 → 适配器会话缓存永远
跨号 miss（GLM 双健康号实测：每对会话的两轮分别落在两个账号，每轮都新建
上游会话全量压平重发）。修法：精确键 miss 时回退查 ignore_assistant 变体键。
"""

from webai2api import gateway

KID = "key-sticky-test"


def _payload(msgs):
    return {"model": "GLM-5.3-Flash", "messages": msgs}


def _setup():
    with gateway._STICKY_LOCK:
        gateway._STICKY.clear()


def test_round1_and_round2_keys_differ():
    """第 1 轮（无 assistant）与第 2 轮（回传回复）的精确键必然不同。"""
    k1 = gateway._conv_key(_payload([{"role": "user", "content": "你好"}]), KID)
    k2 = gateway._conv_key(
        _payload(
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "嗨"},
                {"role": "user", "content": "继续"},
            ]
        ),
        KID,
    )
    assert k1 and k2 and k1 != k2


def test_ignore_assistant_variant_matches_round1_key():
    """变体键（忽略 assistant）与首轮自然键一致——兜底查得到第 1 轮的粘性。"""
    k1 = gateway._conv_key(_payload([{"role": "user", "content": "你好"}]), KID)
    round2 = _payload(
        [
            {"role": "system", "content": "每次都变的 system 噪声"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "嗨"},
            {"role": "user", "content": "继续"},
        ]
    )
    assert gateway._conv_key(round2, KID, ignore_assistant=True) == k1


def test_tool_call_round_variant_matches_round1_key():
    """工具轮同理：回传的 assistant.tool_calls 在变体键里也被忽略。"""
    k1 = gateway._conv_key(_payload([{"role": "user", "content": "写文件"}]), KID)
    round2 = _payload(
        [
            {"role": "user", "content": "写文件"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": '{"filePath":"D:/a.txt"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_x", "content": "ok"},
        ]
    )
    assert gateway._conv_key(round2, KID, ignore_assistant=True) == k1
    # 精确键仍然区分工具轮（防「工具轮与首轮同键」的既有契约）
    assert gateway._conv_key(round2, KID) != k1


def test_sticky_fallback_bridges_round_transition():
    """完整粘性流：第 1 轮存自然键 → 第 2 轮精确 miss、变体命中 →
    第 2 轮成功后按回传键重设 → 第 3 轮精确命中。"""
    _setup()
    p1 = _payload([{"role": "user", "content": "你好"}])
    p2 = _payload(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "嗨"},
            {"role": "user", "content": "继续"},
        ]
    )
    k1 = gateway._conv_key(p1, KID)
    k2 = gateway._conv_key(p2, KID)
    gateway._sticky_set(k1, "accA")

    # 第 2 轮：精确 miss，回退变体命中 accA（与网关 _chat 查找顺序一致）
    sticky = gateway._sticky_get(k2)
    if not sticky:
        sticky = gateway._sticky_get(
            gateway._conv_key(p2, KID, ignore_assistant=True)
        )
    assert sticky == "accA"

    # 第 2 轮成功后按回传键重设，第 3 轮精确命中
    gateway._sticky_set(k2, "accA")
    assert gateway._sticky_get(k2) == "accA"


def test_fallback_does_not_cross_keys():
    """不同会话（不同首条 user）的变体键不互相串。"""
    _setup()
    ka = gateway._conv_key(_payload([{"role": "user", "content": "会话A"}]), KID)
    pb = _payload(
        [
            {"role": "user", "content": "会话B"},
            {"role": "assistant", "content": "回B"},
            {"role": "user", "content": "继续B"},
        ]
    )
    gateway._sticky_set(ka, "accA")
    sticky = gateway._sticky_get(gateway._conv_key(pb, KID))
    if not sticky:
        sticky = gateway._sticky_get(
            gateway._conv_key(pb, KID, ignore_assistant=True)
        )
    assert sticky == ""
