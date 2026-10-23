"""Kimi 网页版适配器配置。

协议要点（2026-10 实测确认）：
- 聊天端点 POST https://www.kimi.com/api/chat（无鉴权时 401 auth.token.empty；
  缺 name 字段时 422 "Name is a required field"）
- 鉴权 Authorization: Bearer <accessToken>（登录时从浏览器抓取）
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://www.kimi.com"


@dataclass(slots=True)
class KimiConfig:
    base_url: str = DEFAULT_BASE_URL
    auth_token: str = ""
    cookie: str = ""
    default_model: str = "kimi-k3"
    default_effort: str = ""
    request_timeout: int = 180
    delete_conversation: bool = False
    session_cache_enabled: bool = True
    session_cache_ttl: int = 43200
    session_cache_max_size: int = 2000
    anti_rate_limit_enabled: bool = True
    anti_rate_limit_min_delay: float = 1.5
    anti_rate_limit_max_delay: float = 3.0
    search_enabled: bool = False

    # ── 防风控：设备指纹与节奏 ──
    device_id: str = ""
    user_agent: str = ""
    device_model: str = ""
    long_pause_chance: float = 0.02
    min_request_interval: float = 1.2
    daily_request_limit: int = 600
    max_turns_per_session: int = 40
    session_store_path: str = ""
    proxy: str = ""
    egress: str = "system"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    @property
    def user_url(self) -> str:
        return f"{self.base_url}/api/user"

    def chat_delete_url(self, chat_id: str) -> str:
        return f"{self.base_url}/api/chat/{chat_id}/delete"

    @property
    def login_url(self) -> str:
        return f"{self.base_url}/"


def _parse_profile(account: dict) -> dict:
    raw = account.get("profile")
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw or "{}")
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _stable_device_id(account: dict) -> str:
    """账号级固定设备指纹：优先 profile.device_id，否则由账号 id 派生。"""
    prof = _parse_profile(account)
    did = str(prof.get("device_id") or "").strip()
    if did:
        return did
    seed = str(account.get("id") or account.get("name") or "")
    return hashlib.sha256(f"kimi-device:{seed}".encode()).hexdigest()[:32]


def _session_store_path(account: dict) -> str:
    """会话缓存落盘路径：data/kimi_sessions/<账号>.json。失败返回空串。"""
    try:
        from ...paths import user_data_dir

        d = user_data_dir() / "kimi_sessions"
        d.mkdir(parents=True, exist_ok=True)
        aid = str(account.get("id") or "unknown")
        return str(d / f"{aid}.json")
    except Exception:
        return ""


def build_config(store, account: dict) -> KimiConfig:
    """按账号 + 全局默认构造一份 KimiConfig。"""
    def g(key: str, default: str) -> str:
        return store.get_setting(key, default)

    def gb(key: str, default: bool) -> bool:
        return str(g(key, "true" if default else "false")).strip().lower() in (
            "1", "true", "yes", "on")

    prof = _parse_profile(account)
    return KimiConfig(
        base_url=g("kimi_base_url", DEFAULT_BASE_URL).rstrip("/"),
        auth_token=account.get("auth_token", ""),
        cookie=account.get("cookie", ""),
        default_model=g("kimi_default_model", "kimi-k3"),
        default_effort=g("kimi_default_effort", ""),
        request_timeout=int(g("kimi_timeout", "180")),
        delete_conversation=not gb("kimi_session_cache", True),
        session_cache_enabled=gb("kimi_session_cache", True),
        session_cache_ttl=int(g("kimi_cache_ttl", "43200")),
        session_cache_max_size=int(g("kimi_cache_max_size", "2000")),
        anti_rate_limit_enabled=gb("kimi_anti_rate_limit", True),
        anti_rate_limit_min_delay=float(g("kimi_anti_min", "1.5")),
        anti_rate_limit_max_delay=float(g("kimi_anti_max", "3.0")),
        search_enabled=gb("kimi_search", False),
        device_id=_stable_device_id(account),
        user_agent=str(prof.get("ua") or ""),
        device_model=str(prof.get("platform") or ""),
        long_pause_chance=float(g("kimi_long_pause_chance", "0.02")),
        min_request_interval=float(g("kimi_min_request_interval", "1.2")),
        daily_request_limit=int(g("kimi_daily_limit", "600")),
        max_turns_per_session=int(g("kimi_max_turns", "40")),
        session_store_path=_session_store_path(account),
        proxy=str(account.get("proxy") or ""),
        egress=str(account.get("egress") or "system"),
    )
