"""GLM parts 累积器测试（按 2026-10-25 抓帧实测的上游协议）。

真实语义：每帧 parts 是**增量 delta**（每帧几个新字）；part 结束发
status=finish 帧携带**全量快照**（权威校准）；思考/正文不同 logic_id
交错到达；偶发空 tool_calls part。此前把 delta 当快照做前缀差，
是丢字/咬字/JSON 漏正文的总根源。
"""
from __future__ import annotations

from webai2api.adapters.chatglm.client import _GlmParts


def _d(lid, text, status="init"):
    return {"logic_id": lid, "status": status,
            "content": [{"type": "text", "text": text}]}


def test_delta_accumulation():
    """逐帧 delta 累积：输出 = 全部 delta 顺序拼接，一字不丢。"""
    acc = _GlmParts()
    for piece in ["我", "是", "智谱", "清言"]:
        acc.consume([_d("a", piece)])
    assert acc.text == "我是智谱清言"


def test_finish_frame_calibrates_to_full_snapshot():
    """finish 帧携带全量快照：直接校准（若 delta 累计有重复，以快照为准）。"""
    acc = _GlmParts()
    for piece in ["用户要求", "用一句话", "介绍自己"]:
        acc.consume([{"logic_id": "a", "status": "init",
                      "content": [{"type": "think", "think": piece}]}])
    acc.consume([{"logic_id": "a", "status": "finish",
                  "content": [{"type": "think", "think": "用户要求用一句话介绍自己"}]}])
    assert acc.reasoning == "用户要求用一句话介绍自己"


def test_interleaved_think_and_answer_parts():
    """思考(lid=A)与正文(lid=B)交错到达：各自累计、分开产出。"""
    acc = _GlmParts()
    acc.consume([{"logic_id": "A", "status": "init",
                  "content": [{"type": "think", "think": "先想"}]}])
    acc.consume([{"logic_id": "B", "status": "init",
                  "content": [{"type": "text", "text": "我是"}]}])
    acc.consume([{"logic_id": "A", "status": "init",
                  "content": [{"type": "think", "think": "清楚"}]}])
    acc.consume([{"logic_id": "B", "status": "init",
                  "content": [{"type": "text", "text": "智谱清言"}]}])
    assert acc.reasoning == "先想清楚"
    assert acc.text == "我是智谱清言"


def test_empty_tool_calls_part_ignored():
    """偶发的空 tool_calls part（st=finish 无内容）不影响输出。"""
    acc = _GlmParts()
    acc.consume([{"logic_id": "t", "status": "finish",
                  "content": [{"type": "tool_calls", "tool_calls": {}}]}])
    acc.consume([_d("a", "正文")])
    assert acc.text == "正文"


def test_two_text_parts_sequential():
    """两个正文 part（少见）：按首见序拼接，各自 delta 独立累计。"""
    acc = _GlmParts()
    acc.consume([_d("a", "第一段")])
    acc.consume([_d("b", "第二段")])
    acc.consume([_d("a", "的续")])
    assert acc.text == "第一段的续第二段"


def test_monotonic_output():
    """全量输出长度永不回缩（流式 prefix-diff 链的前提）。"""
    acc = _GlmParts()
    lens = []
    frames = [
        [_d("A", "用户")],
        [_d("B", "你好")],
        [_d("A", "用户要求介绍")],
        [_d("B", "你好，我是")],
        [{"logic_id": "A", "status": "finish",
          "content": [{"type": "think", "think": "用户要求介绍自己"}]}],
        [_d("B", "你好，我是智谱清言")],
    ]
    for f in frames:
        acc.consume(f)
        lens.append(len(acc.text) + len(acc.reasoning))
    assert lens == sorted(lens), f"输出回缩: {lens}"


def test_consume_ignores_malformed():
    acc = _GlmParts()
    acc.consume(None)
    acc.consume(["garbage", 42, {"content": "not-a-list"}])
    assert acc.text == "" and acc.reasoning == ""
