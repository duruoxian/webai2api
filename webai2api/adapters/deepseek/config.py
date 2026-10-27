from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EXPOSED_MODELS = (
    "deepseek-chat",
    "deepseek-reasoner",
    "default",
)


@dataclass(slots=True)
class AppConfig:
    env_file_path: Path
    env_file_created: bool

    host: str
    port: int
    api_prefix: str
    log_level: str
    debug_dump_all: bool
    request_timeout: int

    server_api_keys: list[str]

    auth_token: str
    device_id: str
    settings_token: str
    cookie: str
    base_url: str
    wasm_path: str

    default_model: str
    delete_conversation: bool
    max_concurrency: int
    queue_wait_timeout: int

    admin_password: str

    # ── 防风控 ──
    anti_rate_limit_enabled: bool = True
    anti_rate_limit_min_delay: float = 1.5
    anti_rate_limit_max_delay: float = 3.0

    # ── 防风控（行为画像 / 配额 / 会话续用）──
    user_agent: str = ""
    device_model: str = ""
    session_store_path: str = ""
    daily_request_limit: int = 150
    long_pause_chance: float = 0.02
    max_turns_per_session: int = 40
    min_request_interval: float = 1.2
    proxy: str = ""  # 账号级出口代理（custom 模式使用）
    egress: str = "system"  # direct / system / custom

    # ── 会话缓存 ──
    session_cache_enabled: bool = True
    session_cache_max_size: int = 500
    session_cache_ttl: int = 7200

    search_enabled: bool = False
    strip_citations: bool = True

    file_upload_enabled: bool = True

    # ── 服务加固 ──
    cors_enabled: bool = False
    max_request_bytes: int = 32 * 1024 * 1024
    http_max_threads: int = 64

    # ── 工具调用解析 ──
    tool_json_fallback: bool = False
    tool_name_strict: bool = True

    # ── DeepSeek API 端点 ──
    @property
    def pow_challenge_url(self) -> str:
        return f"{self.base_url}/api/v0/chat/create_pow_challenge"

    @property
    def session_create_url(self) -> str:
        return f"{self.base_url}/api/v0/chat_session/create"

    @property
    def chat_completion_url(self) -> str:
        return f"{self.base_url}/api/v0/chat/completion"

    @property
    def session_delete_url(self) -> str:
        return f"{self.base_url}/api/v0/chat_session/delete"

    @property
    def session_list_url(self) -> str:
        """上游会话历史列表（分页，has_more 驱动）。"""
        return f"{self.base_url}/api/v0/chat_session/fetch_page"

    @property
    def hif_leim_url(self) -> str:
        return "https://hif-leim.deepseek.com/query"

    @property
    def hif_dliq_url(self) -> str:
        return "https://hif-dliq.deepseek.com/query"
