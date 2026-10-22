"""智谱清言适配器配置。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://chatglm.cn"
DEFAULT_ASSISTANT_ID = "65940acff94777010aa6b796"
SIGN_SECRET = "8a1317a7468aa3ad86e997d08f3f31cb"


@dataclass(slots=True)
class GlmConfig:
    base_url: str = DEFAULT_BASE_URL
    refresh_token: str = ""
    default_model: str = "GLM-5.3"
    default_effort: str = ""
    assistant_id: str = DEFAULT_ASSISTANT_ID
    sign_secret: str = SIGN_SECRET
    request_timeout: int = 180
    delete_conversation: bool = True
    session_cache_enabled: bool = True
    session_cache_ttl: int = 43200
    session_cache_max_size: int = 2000
    anti_rate_limit_enabled: bool = True
    anti_rate_limit_min_delay: float = 1.5
    anti_rate_limit_max_delay: float = 3.0
    search_enabled: bool = False

    # ── 防风控：设备指纹与节奏 ──
    device_id: str = ""            # 账号级固定设备指纹（登录时生成并持久化）
    user_agent: str = ""           # 登录时抓到的 UA，缺失时按 device_id 稳定选一个
    device_model: str = ""         # 登录平台的近似值
    long_pause_chance: float = 0.02
    min_request_interval: float = 1.2
    daily_request_limit: int = 600
    max_turns_per_session: int = 40
    session_store_path: str = ""   # 会话缓存落盘路径（空=不持久化）
    proxy: str = ""                # 账号级出口代理（custom 模式使用）
    egress: str = "system"         # direct / system / custom

    @property
    def refresh_url(self) -> str:
        return f"{self.base_url}/chatglm/user-api/user/refresh"

    @property
    def stream_url(self) -> str:
        return f"{self.base_url}/chatglm/backend-api/assistant/stream"

    @property
    def delete_url(self) -> str:
        return f"{self.base_url}/chatglm/backend-api/assistant/conversation/delete"

    @property
    def login_url(self) -> str:
        return f"{self.base_url}/main/alltoolsdetail"


def _parse_profile(account: dict) -> dict:
    import json

    raw = account.get("profile")
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw or "{}")
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _stable_device_id(account: dict) -> str:
    """账号级固定设备指纹：优先 profile.device_id，否则由账号 id 派生（稳定不变）。"""
    import hashlib

    prof = _parse_profile(account)
    did = str(prof.get("device_id") or "").strip()
    if did:
        return did
    seed = str(account.get("id") or account.get("name") or "")
    return hashlib.sha256(f"glm-device:{seed}".encode()).hexdigest()[:32]


def _session_store_path(account: dict) -> str:
    """会话缓存落盘路径：data/glm_sessions/<账号>.json。失败返回空串（不持久化）。"""
    try:
        from ...paths import user_data_dir

        d = user_data_dir() / "glm_sessions"
        d.mkdir(parents=True, exist_ok=True)
        aid = str(account.get("id") or "unknown")
        return str(d / f"{aid}.json")
    except Exception:
        return ""


def build_config(store, account: dict) -> GlmConfig:
    """按账号 + 全局默认构造一份 GlmConfig。"""
    def g(key: str, default: str) -> str:
        return store.get_setting(key, default)

    def gb(key: str, default: bool) -> bool:
        return str(g(key, "true" if default else "false")).strip().lower() in (
            "1", "true", "yes", "on")

    prof = _parse_profile(account)
    return GlmConfig(
        base_url=g("glm_base_url", DEFAULT_BASE_URL).rstrip("/"),
        refresh_token=account.get("auth_token", ""),
        default_model=g("glm_default_model", "GLM-5.3"),
        default_effort=g("glm_default_effort", ""),
        assistant_id=g("glm_assistant_id", DEFAULT_ASSISTANT_ID),
        sign_secret=g("glm_sign_secret", SIGN_SECRET),
        request_timeout=int(g("glm_timeout", "180")),
        delete_conversation=not gb("glm_session_cache", True),
        session_cache_enabled=gb("glm_session_cache", True),
        session_cache_ttl=int(g("glm_cache_ttl", "43200")),
        session_cache_max_size=int(g("glm_cache_max_size", "2000")),
        anti_rate_limit_enabled=gb("glm_anti_rate_limit", True),
        anti_rate_limit_min_delay=float(g("glm_anti_min", "1.5")),
        anti_rate_limit_max_delay=float(g("glm_anti_max", "3.0")),
        search_enabled=gb("glm_search", False),
        device_id=_stable_device_id(account),
        user_agent=str(prof.get("ua") or ""),
        device_model=str(prof.get("platform") or ""),
        long_pause_chance=float(g("glm_long_pause_chance", "0.02")),
        min_request_interval=float(g("glm_min_request_interval", "1.2")),
        daily_request_limit=int(g("glm_daily_limit", "600")),
        max_turns_per_session=int(g("glm_max_turns", "40")),
        session_store_path=_session_store_path(account),
        proxy=str(account.get("proxy") or ""),
        egress=str(account.get("egress") or "system"),
    )
