"""GLM parts 累积器测试（移植 glm2api 的 per-logic_id 机制）。

上游 SSE 每帧的 parts 是**全量快照**（同 part 重发、整段重写、多 part
交错/乱序到达），按到达序盲拼 = 输出咬字交错（用户实测
「上—你是里的编写你本。后到do早之前楼or」）。_GlmParts 按 logic_id
逐 part 冻结最长版本，产出单调增长的全量文本。
"""
from __future__ import annotations

from webai2api.adapters.chatglm.client import _GlmParts


def _part(lid, text, status="append"):
    return {"logic_id": lid, "status": status,
            "content": [{"type": "text", "text": text}]}


def test_single_part_grows():
    """单 part 快照逐帧增长：全量跟着增长且无重复。"""
    acc = _GlmParts()
    acc.consume([_part("a", "你")])
    acc.consume([_part("a", "你好")])
    acc.consume([_part("a", "你好！")])
    assert acc.text == "你好！"


def test_snapshot_rewrite_shorter_keeps_old():
    """整段重写/缩短：不回退（宁旧勿错），全量单调。"""
    acc = _GlmParts()
    acc.consume([_part("a", "第一版完整回答内容")])
    acc.consume([_part("a", "短的")])          # 上游重写变短
    assert acc.text == "第一版完整回答内容"
    acc.consume([_part("a", "第一版完整回答内容。补充")])  # 再增长
    assert acc.text == "第一版完整回答内容。补充"


def test_interleaved_parts_no_bitewing():
    """两 part 交错到达（核心咬字场景）：各 part 独立冻结、按首见序拼接。

    旧实现盲拼快照会产出「A0B0A1B1」式交错或错位。
    """
    acc = _GlmParts()
    acc.consume([_part("a", "第一段开头")])
    acc.consume([_part("b", "第二段开头")])
    acc.consume([_part("a", "第一段开头第一段结尾"),
                 _part("b", "第二段开头第二段结尾")])
    assert acc.text == "第一段开头第一段结尾\n\n第二段开头第二段结尾"


def test_parts_out_of_order_in_single_frame():
    """同帧内 part 顺序翻转：不影响输出顺序（按首见序冻结）。"""
    acc = _GlmParts()
    acc.consume([_part("a", "A")])
    acc.consume([_part("b", "B"), _part("a", "A2")])
    assert acc.text == "A2\n\nB"


def test_think_tracked_separately():
    acc = _GlmParts()
    acc.consume([{"logic_id": "a", "status": "append", "content": [
        {"type": "think", "think": "思考中"},
        {"type": "text", "text": "正文"},
    ]}])
    acc.consume([{"logic_id": "a", "status": "append", "content": [
        {"type": "think", "think": "思考中思考完"},
        {"type": "text", "text": "正文完整版"},
    ]}])
    assert acc.text == "正文完整版"
    assert acc.reasoning == "思考中思考完"


def test_no_logic_id_parts_get_stable_ids():
    """上游不发 logic_id 的 part：按位置给稳定 id，不互相覆盖。"""
    acc = _GlmParts()
    acc.consume([{"status": "append", "content": [{"type": "text", "text": "甲"}]},
                 {"status": "append", "content": [{"type": "text", "text": "乙"}]}])
    assert "甲" in acc.text and "乙" in acc.text


def test_consume_ignores_malformed():
    acc = _GlmParts()
    acc.consume(None)
    acc.consume(["garbage", 42, {"content": "no list"}])
    assert acc.text == ""


def test_monotonic_never_shrinks():
    """全量文本长度永不回缩（流式 prefix-diff 链的前提）。"""
    acc = _GlmParts()
    lens = []
    frames = [
        [_part("a", "很长很长的第一段内容")],
        [_part("b", "B 段")],
        [_part("a", "短")],                      # 重写缩短
        [_part("b", "B 段更多"), _part("a", "很长很长的第一段内容")],
    ]
    for f in frames:
        acc.consume(f)
        lens.append(len(acc.text))
    assert lens == sorted(lens), f"全量回缩: {lens}"
