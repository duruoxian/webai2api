"""PoW challenge 响应异常时的健壮性回归测试。

事故（2026-10-27）：上游 token 失效时返回 ``{"code": 40003, "data": null}``，
``_get_pow_challenge`` 直接 ``payload["data"].get(...)`` 抛 AttributeError，
被网关包成 502「上游服务异常」，掩盖了真正原因（授权失效）。
现在必须抛带 code/msg 的 UpstreamAPIError。
"""

from __future__ import annotations

import pytest

from webai2api.adapters.deepseek.services.deepseek_client import UpstreamAPIError


def _client():
    """构造一个只用于调用 _get_pow_challenge 的轻量客户端。"""
    from types import SimpleNamespace

    from webai2api.adapters.deepseek.services import deepseek_client as dc

    obj = object.__new__(dc.DeepSeekWebClient)
    obj.config = SimpleNamespace(
        base_url="https://chat.deepseek.com",
        egress="direct",
        proxy="",
        pow_challenge_url="https://chat.deepseek.com/api/v0/chat/create_pow_challenge",
    )
    import logging

    obj.logger = logging.getLogger("pow-test")
    obj.auth = SimpleNamespace(
        get_auth_headers=lambda: {}, absorb_response_cookies=lambda r: None
    )
    return obj


@pytest.mark.parametrize(
    "payload,expect_hint",
    [
        (
            {
                "code": 40003,
                "data": None,
                "msg": "Authorization Failed (invalid token)",
            },
            "40003",
        ),
        ({"code": 401, "data": None, "msg": "unauthorized"}, "data 为空"),
        ({"code": 0, "data": {}, "msg": ""}, "格式异常"),
        ({"code": 0, "data": {"biz_data": None}, "msg": ""}, "格式异常"),
    ],
)
def test_pow_challenge_bad_shape_raises_upstream_error(
    payload, expect_hint, monkeypatch
):
    import urllib.request

    client = _client()

    class _Resp:
        status = 200

        def read(self):
            return b""

        def getcode(self):
            return 200

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setattr(
        client.auth, "_read_json_response", lambda r: payload, raising=False
    )

    with pytest.raises(UpstreamAPIError) as ei:
        client._get_pow_challenge()

    assert expect_hint in str(ei.value)


def test_pow_challenge_auth_failure_maps_to_403():
    """code=40003 属于授权失效，应映射 403 而不是 500，便于上层分类换号。"""
    import urllib.request

    client = _client()
    payload = {"code": 40003, "data": None, "msg": "Authorization Failed"}

    class _Resp:
        status = 200

        def read(self):
            return b""

        def getcode(self):
            return 200

    orig = urllib.request.urlopen
    try:
        urllib.request.urlopen = lambda *a, **k: _Resp()  # type: ignore[assignment]
        client.auth._read_json_response = lambda r: payload  # type: ignore[attr-defined]
        with pytest.raises(UpstreamAPIError) as ei:
            client._get_pow_challenge()
        assert ei.value.status_code == 403
    finally:
        urllib.request.urlopen = orig  # type: ignore[assignment]
