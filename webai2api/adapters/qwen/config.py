"""通义千问网页版适配器配置。

协议要点（2026-10 实测确认）：
- OpenWebUI 架构：POST https://chat.qwen.ai/api/v2/chat/completions?chat_id=<id>
- 鉴权 Authorization: Bearer <access_token>（localStorage / 状态树里的 token 字段）
- 响应外层信封：HTTP 200 + {"success": true/false, "data": {...}}
- 模型：qwen3.7-plus / qwen3.8-max / qwen3.8-omni-flash（capabilities.thinking=true）
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://chat.qwen.ai"


@dataclass(slots=True)
class QwenConfig:
    base_url: str = DEFAULT_BASE_URL
    auth_token: str = ""
    cookie: str = ""
    default_model: str = "qwen3.8-max"
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
        return f"{self.base_url}/api/v2/chat/completions"

    @property
    def user_url(self) -> str:
        return f"{self.base_url}/api/v2/auths/"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/api/v2/models/"

    def chat_delete_url(self, chat_id: str) -> str:
        return f"{self.base_url}/api/v2/chats/{chat_id}"

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
    return hashlib.sha256(f"qwen-device:{seed}".encode()).hexdigest()[:32]


def _session_store_path(account: dict) -> str:
    """会话缓存落盘路径：data/qwen_sessions/<账号>.json。失败返回空串。"""
    try:
        from ...paths import user_data_dir

        d = user_data_dir() / "qwen_sessions"
        d.mkdir(parents=True, exist_ok=True)
        aid = str(account.get("id") or "unknown")
        return str(d / f"{aid}.json")
    except Exception:
        return ""


def build_config(store, account: dict) -> QwenConfig:
    """按账号 + 全局默认构造一份 QwenConfig。"""
    def g(key: str, default: str) -> str:
        return store.get_setting(key, default)

    def gb(key: str, default: bool) -> bool:
        return str(g(key, "true" if default else "false")).strip().lower() in (
            "1", "true", "yes", "on")

    prof = _parse_profile(account)
    return QwenConfig(
        base_url=g("qwen_base_url", DEFAULT_BASE_URL).rstrip("/"),
        auth_token=account.get("auth_token", ""),
        cookie=account.get("cookie", ""),
        default_model=g("qwen_default_model", "qwen3.8-max"),
        default_effort=g("qwen_default_effort", ""),
        request_timeout=int(g("qwen_timeout", "180")),
        delete_conversation=not gb("qwen_session_cache", True),
        session_cache_enabled=gb("qwen_session_cache", True),
        session_cache_ttl=int(g("qwen_cache_ttl", "43200")),
        session_cache_max_size=int(g("qwen_cache_max_size", "2000")),
        anti_rate_limit_enabled=gb("qwen_anti_rate_limit", True),
        anti_rate_limit_min_delay=float(g("qwen_anti_min", "1.5")),
        anti_rate_limit_max_delay=float(g("qwen_anti_max", "3.0")),
        search_enabled=gb("qwen_search", False),
        device_id=_stable_device_id(account),
        user_agent=str(prof.get("ua") or ""),
        device_model=str(prof.get("platform") or ""),
        long_pause_chance=float(g("qwen_long_pause_chance", "0.02")),
        min_request_interval=float(g("qwen_min_request_interval", "1.2")),
        daily_request_limit=int(g("qwen_daily_limit", "600")),
        max_turns_per_session=int(g("qwen_max_turns", "40")),
        session_store_path=_session_store_path(account),
        proxy=str(account.get("proxy") or ""),
        egress=str(account.get("egress") or "system"),
    )
