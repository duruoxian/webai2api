"""GLM 重复输出去重测试（问题 3：同一段回答完整输出两遍）。

覆盖 _dedup_repeated 非流式折叠（P+P → P，非重复文本不动）。（RepeatGuard 已随流式链重构移除：其针对的「全文重发」bug 根治后，快照口径守卫在增量链路里只会引入错位风险。）
"""
from __future__ import annotations

from webai2api.adapters.chatglm.client import _dedup_repeated

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


