"""2026-11-01 夜间批次2后端回归：/state 分页总数与 CSV 导出跟随筛选。

- UI-2：RECENT_STATUS='error' 时 recent_total 必须用「非 ok」口径，
  此前 count_usage 丢掉 status → 分页总数虚高、可翻到空页。
- UI-15：/usage/export 接受 status/key_id/account_id，导出与界面所见一致。
"""

from __future__ import annotations

import time

from webai2api.store import Store


def _seed(tmp_path):
    st = Store(tmp_path / "data" / "webai2api.db")
    now = time.time()
    st.log_usage(
        status="ok", api_key_id="k1", api_key_name="K1", account_id="a1",
        account_name="A1", provider="deepseek", model="deepseek-chat",
        prompt_tokens=10, completion_tokens=5, total_tokens=15, latency_ms=900,
        first_token_ms=200, ts=now - 30,
    )
    st.log_usage(
        status="error:UpstreamAPIError", api_key_id="k1", api_key_name="K1",
        account_id="a1", account_name="A1", provider="deepseek",
        model="deepseek-chat", prompt_tokens=10, completion_tokens=0,
        total_tokens=10, latency_ms=800, error_code="500", ts=now - 20,
    )
    st.log_usage(
        status="cancelled", api_key_id="k2", api_key_name="K2", account_id="a2",
        account_name="A2", provider="chatglm", model="GLM-5.3",
        prompt_tokens=8, completion_tokens=2, total_tokens=10, latency_ms=700,
        error_code="cancelled", ts=now - 10,
    )
    return st


def test_count_usage_error_means_not_ok(tmp_path):
    st = _seed(tmp_path)
    assert st.count_usage() == 3
    assert st.count_usage(status="error") == 2, "只看失败 = 一切非 ok（error:* 与 cancelled）"
    assert st.count_usage(status="ok") == 1
    assert st.count_usage(key_id="k2") == 1
    assert st.count_usage(account_id="a2", status="error") == 1
    # 与 recent_usage 同口径：同一筛选下两侧条数一致
    rows = st.recent_usage(limit=50, offset=0, status="error")
    assert len(rows) == st.count_usage(status="error") == 2


def test_export_usage_follows_filters(tmp_path):
    st = _seed(tmp_path)
    assert len(st.export_usage()) == 3
    assert len(st.export_usage(status="error")) == 2
    assert len(st.export_usage(key_id="k1")) == 2
    assert len(st.export_usage(account_id="a2")) == 1
    assert all(r["api_key_id"] == "k2" for r in st.export_usage(account_id="a2"))


def test_recent_account_errors_latest_per_account(tmp_path):
    st = _seed(tmp_path)  # a1: error:UpstreamAPIError(20s前) + cancelled(10s前,k2/a2)
    errs = st.recent_account_errors()
    assert set(errs) == {"a1", "a2"}
    assert errs["a1"]["code"] == "500", "取账号最近一条失败（优先具体 error_code）"
    assert errs["a2"]["code"] == "cancelled"
    assert errs["a1"]["ts"] >= errs["a2"]["ts"] - 1
    # 成功请求不影响摘要
    st.log_usage(
        status="ok", api_key_id="k1", api_key_name="K1", account_id="a1",
        account_name="A1", provider="deepseek", model="deepseek-chat",
        prompt_tokens=1, completion_tokens=1, total_tokens=2, ts=time.time(),
    )
    assert st.recent_account_errors()["a1"]["code"] == "500", "成功请求不影响失败摘要"
