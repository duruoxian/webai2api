"""通用多账号客户端池。

为每个"启用的账号"构造一份配置与客户端实例（各自独立凭证、互不干扰），
并提供"选一个可用账号"的轮询逻辑。具体 provider 通过传入的配置工厂与
客户端工厂定制，供 deepseek / chatglm 等复用。
"""
from __future__ import annotations

import threading
import time
from logging import Logger
from typing import Callable

from ..store import Store

COOLDOWN_SECONDS = 300


def _is_rate_limited(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(w in low for w in ("429", "rate limit", "too many", "限流", "频繁"))


class AccountClient:
    """一个账号 + 它的客户端。"""

    def __init__(self, account: dict, client, config) -> None:
        self.account = account
        self.client = client
        self.config = config
        self.cooldown_until = 0.0

    def healthy(self) -> bool:
        return self.cooldown_until < time.time()

    def mark_bad(self, seconds: int = COOLDOWN_SECONDS, exc: Exception | None = None) -> None:
        # 被限流的账号冷却更久，否则会一直撞墙
        if exc is not None and _is_rate_limited(exc):
            seconds = max(seconds, 900)
        self.cooldown_until = time.time() + seconds

    def mark_ok(self) -> None:
        self.cooldown_until = 0.0


class GenericClientPool:
    """按 provider 维护账号客户端池。

    参数：
      build_config(store, account) -> config
      client_factory(config, logger) -> client（需实现 chat_completion /
        stream_chat_completion）
    """

    def __init__(
        self,
        store: Store,
        provider: str,
        logger: Logger,
        build_config: Callable[[Store, dict], object],
        client_factory: Callable[[object, Logger], object],
    ) -> None:
        self.store = store
        self.provider = provider
        self.logger = logger
        self._build_config = build_config
        self._client_factory = client_factory
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

        accounts = self.store.list_accounts(self.provider)
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
                cfg = self._build_config(self.store, a)
                client = self._client_factory(cfg, self.logger)
                old = self._clients.get(a["id"])
                ac = AccountClient(a, client, cfg)
                if old:
                    ac.cooldown_until = old.cooldown_until
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

    def pick(self, exclude: set[str] | None = None) -> AccountClient | None:
        """轮询选一个"健康"的可用账号，可排除指定账号。"""
        exclude = exclude or set()
        with self._lock:
            self._sync()
            items = [c for c in self._clients.values()
                     if c.account["id"] not in exclude and c.healthy()]
            if not items:
                # 都冷却了就放开健康限制，总比没有强
                items = [c for c in self._clients.values()
                         if c.account["id"] not in exclude]
            if not items:
                return None
            self._rr = (self._rr + 1) % len(items)
            return items[self._rr]

    def pick_by_id(self, account_id: str) -> AccountClient | None:
        with self._lock:
            self._sync()
            return self._clients.get(account_id)

    def healthy_count(self) -> int:
        with self._lock:
            self._sync()
            return sum(1 for c in self._clients.values() if c.healthy())
