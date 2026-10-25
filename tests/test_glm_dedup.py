"""GLM 重复输出去重测试（问题 3：同一段回答完整输出两遍）。

覆盖两块：
1. _dedup_repeated 非流式折叠（P+P → P，非重复文本不动）；
2. RepeatGuard 流式守卫（完整重复丢弃、发散放行、短文本零干预、
   截断复述保守放行）。
"""
from __future__ import annotations

from webai2api.adapters.chatglm.client import RepeatGuard, _dedup_repeated

# ≥ 80 字的整段回答样本（守卫与折叠的最小生效长度）
BLK = "智谱清言的回答第一段，" + "内容" * 35 + "，到此为止。"


# ── _dedup_repeated（非流式）─────────────────────────────

def test_dedup_folds_exact_double():
    assert _dedup_repeated(BLK + BLK) == BLK


def test_dedup_folds_double_with_trailing_newline():
    assert _dedup_repeated(BLK + BLK + "\n") == BLK


def test_dedup_keeps_normal_text():
    text = BLK + "接着是不同的第二段内容，" + "补充" * 20 + "。"
    assert _dedup_repeated(text) == text


def test_dedup_short_repeat_untouched():
    """短文本（<80 字）即使是两遍也不折叠，避免误伤口语重复。"""
    text = "好的好的"
    assert _dedup_repeated(text) == text


def test_dedup_empty_and_none():
    assert _dedup_repeated("") == ""
    assert _dedup_repeated("单遍文本") == "单遍文本"


# ── RepeatGuard（流式）───────────────────────────────────

def _run_guard(snapshots):
    """喂入全文快照序列，返回 (最终已放行文本, 收尾残余)。"""
    g = RepeatGuard()
    out = ""
    for s in snapshots:
        out = g.feed(s)
    return out, g.finish()


def test_guard_passthrough_no_repeat():
    snaps = ["第一段", "第一段第二段", "第一段第二段结尾"]
    out, tail = _run_guard(snaps)
    assert out + tail == "第一段第二段结尾"


def test_guard_drops_confirmed_full_repeat():
    """整段答案输出两遍：第二遍要被扣住丢弃。"""
    a = BLK
    snaps = [a[:40], a, a + a[:40], a + a]
    out, tail = _run_guard(snaps)
    assert out == a
    assert tail == ""


def test_guard_flushes_when_diverged():
    """开头疑似复述但中途发散 → 全部放行，不丢内容。"""
    a = BLK
    diverged = a + a[:50] + "但这里内容发散了，" + "新内容" * 20
    out, tail = _run_guard([a, diverged])
    assert out + tail == diverged


def test_guard_short_text_never_held():
    """已放行部分 < 80 字不启用扣留：口语化短重复原样通过。"""
    out, tail = _run_guard(["好的", "好的好的"])
    assert out + tail == "好的好的"


def test_guard_truncated_repeat_flushed_at_finish():
    """复述没等到完整就断流：保守放行（宁重复、不丢内容）。"""
    a = BLK
    out, tail = _run_guard([a, a + a[:30]])
    assert out + tail == a + a[:30]


def test_guard_triple_repeat_collapsed():
    """三遍输出 → 只保留一遍。"""
    a = BLK
    out, tail = _run_guard([a, a + a, a + a + a])
    assert out + tail == a


def test_guard_disarms_on_rewrite():
    """上游快照改写（非追加）：守卫放弃，直通全部。"""
    g = RepeatGuard()
    out1 = g.feed(BLK)
    rewritten = "完全不同的新快照内容，" + "改写" * 40
    out2 = g.feed(rewritten)
    assert out1 == BLK
    assert out2 == rewritten
