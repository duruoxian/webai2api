"""cancelled 用量口径测试（2026-10-28 用户反馈面板出现 13.7s 产 5.8万 token）。

背景：流式请求被客户端中断时上游不给 usage，gateway 用本地估算兜底。
旧实现的估算把 reasoning_content 之外的正文也截断（out_parts<20000 条），
且 usage 解析忽略 completion_tokens_details.reasoning_tokens，
导致取消记录的 completion 数字既不可信也不一致。
"""

from __future__ import annotations

from webai2api import gateway


def test_reasoning_tokens_fallback_when_completion_missing():
    """上游只给 reasoning_tokens 时（思考模型常见），completion 必须取到它。"""
    resp = {
        "usage": {
            "prompt_tokens": 7000,
            "completion_tokens_details": {"reasoning_tokens": 4200},
        }
    }
    p, c, t = gateway._usage_tokens_from_openai(resp)
    assert (p, c) == (7000, 4200)
    assert t == 11200


def test_explicit_completion_wins():
    resp = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 33,
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
    }
    assert gateway._usage_tokens_from_openai(resp) == (10, 33, 43)


def test_estimate_completion_counts_reasoning_text():
    """估算路径：思考文本也要计入 completion（取消时段没有上游 usage）。"""
    text = "思考" * 100  # 200 字符
    c = gateway._estimate_completion({}, text)
    assert c > 0
    assert c == gateway._estimate_tokens(text)


def test_estimate_completion_without_output_falls_back_to_half_max_tokens():
    assert gateway._estimate_completion({"max_tokens": 100}) == 50
