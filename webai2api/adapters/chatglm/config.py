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
    session_cache_ttl: int = 7200
    session_cache_max_size: int = 500
    anti_rate_limit_enabled: bool = True
    anti_rate_limit_min_delay: float = 1.5
    anti_rate_limit_max_delay: float = 3.0
    search_enabled: bool = False

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


def build_config(store, account: dict) -> GlmConfig:
    """按账号 + 全局默认构造一份 GlmConfig。"""
    def g(key: str, default: str) -> str:
        return store.get_setting(key, default)

    def gb(key: str, default: bool) -> bool:
        return str(g(key, "true" if default else "false")).strip().lower() in (
            "1", "true", "yes", "on")

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
        session_cache_ttl=int(g("glm_cache_ttl", "7200")),
        session_cache_max_size=int(g("glm_cache_max_size", "500")),
        anti_rate_limit_enabled=gb("glm_anti_rate_limit", True),
        anti_rate_limit_min_delay=float(g("glm_anti_min", "1.5")),
        anti_rate_limit_max_delay=float(g("glm_anti_max", "3.0")),
        search_enabled=gb("glm_search", False),
    )
