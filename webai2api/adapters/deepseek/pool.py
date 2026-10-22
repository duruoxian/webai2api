"""多账号客户端池。

为每个"启用的账号"构造一份 DeepSeek 配置与客户端实例（各自独立凭证、
互不干扰），并提供"选一个可用账号"的轮询逻辑。

改进点：
- 客户端按账号缓存，只有账号信息变化时才重建（避免每请求查库/重建）
- 支持把某账号临时标记为"冷却中"（如凭证失效），自动跳到其他账号
- 创建客户端时不长时间持锁
"""
from __future__ import annotations

import json
import threading
import time
from logging import Logger

from ...store import Store
from ...paths import user_data_dir
from ..base import account_ready
from . import config as ds_config
from .services.deepseek_client import DeepSeekWebClient

PROVIDER = "deepseek"
COOLDOWN_SECONDS = 120


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _is_rate_limited(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(w in low for w in ("429", "rate limit", "too many", "限流", "频繁"))


def _base_defaults(store: Store) -> dict:
    """从 settings 读取全局默认（可被面板修改）。"""
    return {
        "base_url": store.get_setting("deepseek_base_url", "https://chat.deepseek.com"),
        "wasm_path": store.get_setting("deepseek_wasm_path", "./sha3_wasm_bg.wasm"),
        "default_model": store.get_setting("deepseek_default_model", "deepseek-chat"),
        "anti_min": float(store.get_setting("deepseek_anti_min", "1.5")),
        "anti_max": float(store.get_setting("deepseek_anti_max", "3.0")),
        "max_concurrency": int(store.get_setting("deepseek_max_concurrency", "1")),
        "timeout": int(store.get_setting("deepseek_timeout", "180")),
        "session_cache": store.get_setting("deepseek_session_cache", "true"),
        "cache_ttl": int(store.get_setting("deepseek_cache_ttl", "43200")),
        "cache_max_size": int(store.get_setting("deepseek_cache_max_size", "2000")),
        "daily_limit": int(store.get_setting("deepseek_daily_limit", "600")),
        "long_pause_chance": float(store.get_setting("deepseek_long_pause_chance", "0.02")),
        "max_turns": int(store.get_setting("deepseek_max_turns", "40")),
        "min_interval": float(store.get_setting("deepseek_min_request_interval", "1.2")),
        "search": store.get_setting("deepseek_search", "false"),
    }


def _parse_profile(raw: object) -> dict:
    """账号 profile 可能是 JSON 字符串 / dict / 空，容错解析。"""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        data = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _session_store_path(account: dict) -> str:
    """每账号一个会话持久化文件：<user_data>/deepseek_sessions/<id>.json。"""
    try:
        d = user_data_dir() / "deepseek_sessions"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"{account['id']}.json")
    except OSError:
        return ""


def build_config(store: Store, account: dict) -> ds_config.AppConfig:
    """按账号 + 全局默认构造一份 AppConfig。"""
    d = _base_defaults(store)
    prof = _parse_profile(account.get("profile"))
    return ds_config.AppConfig(
        env_file_path=store.db_path.parent / ".env.virtual",
        env_file_created=False,
        host="127.0.0.1",
        port=0,
        api_prefix="/v1",
        log_level=store.get_setting("log_level", "INFO"),
        debug_dump_all=False,
        request_timeout=d["timeout"],
        server_api_keys=[],
        auth_token=account.get("auth_token", ""),
        device_id=account.get("device_id", ""),
        settings_token=account.get("settings_token", ""),
        cookie=account.get("cookie", ""),
        base_url=d["base_url"],
        wasm_path=d["wasm_path"],
        default_model=d["default_model"],
        delete_conversation=not _truthy(d["session_cache"]),
        max_concurrency=d["max_concurrency"],
        queue_wait_timeout=60,
        admin_password="",
        anti_rate_limit_enabled=True,
        anti_rate_limit_min_delay=d["anti_min"],
        anti_rate_limit_max_delay=d["anti_max"],
        user_agent=str(prof.get("ua") or ""),
        device_model=str(prof.get("platform") or ""),
        session_store_path=_session_store_path(account),
        daily_request_limit=d["daily_limit"],
        long_pause_chance=d["long_pause_chance"],
        max_turns_per_session=d["max_turns"],
        min_request_interval=d["min_interval"],
        session_cache_enabled=_truthy(d["session_cache"]),
        session_cache_max_size=max(1, d["cache_max_size"]),
        session_cache_ttl=d["cache_ttl"],
        search_enabled=_truthy(d["search"]),
    )


def build_pool(store: Store, logger: Logger) -> "ClientPool":
    return ClientPool(store, logger)


class AccountClient:
    """一个账号 + 它的客户端。"""

    def __init__(self, account: dict, client: DeepSeekWebClient, config: ds_config.AppConfig) -> None:
        self.account = account
        self.client = client
        self.config = config
        self.cooldown_until = 0.0
        self.rate_hits = 0
        # ── 日请求配额（跨进程只做本进程内计数）──
        self.daily_limit = int(getattr(config, "daily_request_limit", 0) or 0)
        self._daily_day = time.strftime("%Y-%m-%d")
        self._daily_count = 0

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_day:
            self._daily_day = today
            self._daily_count = 0

    def daily_exhausted(self) -> bool:
        """当日请求数已达上限则不可用；跨天自动归零。"""
        self._roll_day()
        return self.daily_limit > 0 and self._daily_count >= self.daily_limit

    def healthy(self) -> bool:
        return self.cooldown_until < time.time()

    def cooldown_left(self) -> int:
        return max(0, int(self.cooldown_until - time.time()))

    def mark_bad(self, seconds: int = COOLDOWN_SECONDS, exc: Exception | None = None) -> None:
        if exc is not None and _is_rate_limited(exc):
            self.rate_hits += 1
            seconds = min(900, 60 * (2 ** (self.rate_hits - 1)))
        self.cooldown_until = time.time() + seconds

    def mark_ok(self) -> None:
        self.cooldown_until = 0.0
        self.rate_hits = 0
        self._roll_day()
        self._daily_count += 1


class ClientPool:
    def __init__(self, store: Store, logger: Logger) -> None:
        self.store = store
        self.logger = logger
        self._lock = threading.Lock()
        self._clients: dict[str, AccountClient] = {}
        self._rr = 0
        self._last_sync = 0.0

    def _sync(self) -> None:
        """按数据库状态同步客户端池（调用方需持锁）。

        为避免每个请求都查库，非 refresh 情况下最多每 5 秒同步一次。
        """
        now = time.time()
        force = self._last_sync == 0.0
        if not force and now - self._last_sync < 5:
            return
        self._last_sync = now

        accounts = self.store.list_accounts(PROVIDER)
        want = {a["id"] for a in accounts if a.get("enabled")}
        for aid in list(self._clients.keys()):
            if aid not in want:
                self._clients.pop(aid, None)
        for a in accounts:
            if not a.get("enabled"):
                continue
            cached = self._clients.get(a["id"])
            if cached and cached.account.get("updated_at") == a.get("updated_at"):
                continue
            try:
                cfg = build_config(self.store, a)
                client = DeepSeekWebClient(config=cfg, logger=self.logger)
                old = self._clients.get(a["id"])
                ac = AccountClient(a, client, cfg)
                if old:
                    ac.cooldown_until = old.cooldown_until
                    ac.rate_hits = old.rate_hits
                self._clients[a["id"]] = ac
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("账号 %s 初始化失败: %s", a.get("name"), exc)

    def refresh(self) -> None:
        with self._lock:
            self._clients.clear()
            self._last_sync = 0.0
            self._sync()

    def available(self) -> list[AccountClient]:
        with self._lock:
            self._sync()
            return list(self._clients.values())

    def pick(self, exclude: set[str] | None = None,
             allowed: set[str] | None = None,
             prefer: str | None = None) -> AccountClient | None:
        """轮询选一个"健康"的可用账号，可排除 / 限定账号范围 / 优先指定。

        全部处于冷却时，只有冷却快结束（≤45 秒）才勉强放行。
        """
        exclude = exclude or set()
        with self._lock:
            self._sync()
            items = [c for c in self._clients.values()
                     if c.account["id"] not in exclude and c.healthy()
                     and account_ready(c.account) and not c.daily_exhausted()]
            if not items:
                cooling = [c for c in self._clients.values()
                           if c.account["id"] not in exclude and account_ready(c.account)
                           and not c.daily_exhausted()]
                items = [c for c in cooling if c.cooldown_left() <= 45]
            if allowed:
                items = [c for c in items if c.account["id"] in allowed]
            if not items:
                return None
            if prefer:
                for c in items:
                    if c.account["id"] == prefer:
                        return c
            self._rr = (self._rr + 1) % len(items)
            return items[self._rr]

    def cooldowns(self) -> dict[str, int]:
        """各账号剩余冷却秒数（>0 表示正在休息）。"""
        with self._lock:
            self._sync()
            return {aid: c.cooldown_left() for aid, c in self._clients.items()
                    if c.cooldown_left() > 0}

    def pick_by_id(self, account_id: str) -> AccountClient | None:
        with self._lock:
            self._sync()
            return self._clients.get(account_id)

    def healthy_count(self) -> int:
        with self._lock:
            self._sync()
            return sum(1 for c in self._clients.values() if c.healthy())
