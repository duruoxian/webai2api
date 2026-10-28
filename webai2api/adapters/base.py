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


def account_ready(account: dict) -> bool:
    """账号至少要有一种可用凭证，否则选了也只会报错、白等重试。"""
    return bool(
        str(account.get("auth_token") or "").strip()
        or str(account.get("cookie") or "").strip()
    )


def _is_rate_limited(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(w in low for w in ("429", "rate limit", "too many", "限流", "频繁"))


class AccountClient:
    """一个账号 + 它的客户端。"""

    RISK_METRICS = (
        "sessions_created", "sessions_deleted", "cache_hit", "cache_miss",
        "upstream_requests", "rate_limit_hits",
    )

    def __init__(self, account: dict, client, config, store: Store | None = None) -> None:
        self.account = account
        self.client = client
        self.config = config
        self.store = store
        self.cooldown_until = 0.0
        self.rate_hits = 0
        # ── 日请求配额：有 store 时持久化（跨重启累计、按自然日归零）──
        self.daily_limit = int(getattr(config, "daily_request_limit", 0) or 0)
        self._daily_day = time.strftime("%Y-%m-%d")
        self._daily_count = 0
        self._daily_base = self._load_daily()
        self._risk_last_read: dict | None = None
        self._risk_lock = threading.Lock()

    def _account_id(self) -> str:
        return str(self.account.get("id") or "")

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d")

    def _load_daily(self) -> int:
        if self.store is None or not self._account_id():
            return 0
        try:
            return int(self.store.get_stat(self._account_id(), self._daily_day, "requests"))
        except Exception:  # noqa: BLE001
            return 0

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_day:
            self._daily_day = today
            self._daily_count = 0
            self._daily_base = self._load_daily()

    def daily_used(self) -> int:
        """当日已用请求数：持久起点 + 本进程增量（读路径不落库，安全用于 pick() 热路径）。"""
        self._roll_day()
        if self.store is not None and self._account_id():
            return int(self._daily_base) + int(self._daily_count)
        return int(self._daily_count)

    def daily_exhausted(self) -> bool:
        """当日请求数已达上限则不可用（limit<=0 表示不限）。"""
        return self.daily_limit > 0 and self.daily_used() >= self.daily_limit

    def healthy(self) -> bool:
        return self.cooldown_until < time.time()

    def cooldown_left(self) -> int:
        return max(0, int(self.cooldown_until - time.time()))

    def mark_bad(self, seconds: int = COOLDOWN_SECONDS, exc: Exception | None = None) -> None:
        # 限流：首次 60 秒，连续命中逐步翻倍，最长 15 分钟
        if exc is not None and _is_rate_limited(exc):
            self.rate_hits += 1
            seconds = min(900, 60 * (2 ** (self.rate_hits - 1)))
        self.cooldown_until = time.time() + seconds

    def mark_ok(self) -> None:
        self.cooldown_until = 0.0
        self.rate_hits = 0
        self._roll_day()
        self._daily_count += 1
        if self.store is not None and self._account_id():
            try:
                self.store.bump_stat(self._account_id(), self._today(), "requests", 1)
            except Exception:  # noqa: BLE001
                pass

    def persist_and_display_risk(self) -> dict:
        """风险指标：把内存计数相对上次读取的增量持久化到 acc_stats（当日），
        返回「当日持久累计值」供面板展示——跨重启保留、按自然日归零。
        账号级锁防止 /risk 并发读取导致同一差量重复入库。"""
        with self._risk_lock:
            return self._persist_and_display_risk_locked()

    def _persist_and_display_risk_locked(self) -> dict:
        stats: dict = {}
        try:
            stats = dict(self.client.risk_stats() or {}) if self.client is not None else {}
        except Exception:  # noqa: BLE001
            stats = {}
        if not stats:
            return stats
        if self.store is None or not self._account_id():
            return stats
        day = self._today()
        last = self._risk_last_read or {}
        try:
            for m in self.RISK_METRICS:
                now_v = int(stats.get(m) or 0)
                prev = int(last.get(m) or 0)
                delta = now_v - prev
                if delta > 0:
                    self.store.bump_stat(self._account_id(), day, m, delta)
                persisted = int(self.store.get_stat(self._account_id(), day, m))
                # 持久累计值恒 ≥ 本进程内存计数（内存只是本进程的子集）
                stats[m] = max(persisted, now_v)
                last[m] = now_v
            hit, miss = int(stats.get("cache_hit") or 0), int(stats.get("cache_miss") or 0)
            stats["cache_hit_rate"] = round(hit / (hit + miss), 3) if (hit + miss) else 0.0
            self._risk_last_read = last
        except Exception:  # noqa: BLE001
            pass
        return stats


def dispose_client(client) -> None:
    """账号池重建时释放旧客户端独占资源。只关实例自有的对象
    （如 DeepSeek 的 _browser）；模块级共享缓存一律不动。"""
    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
            return
        except Exception:  # noqa: BLE001
            pass
    browser = getattr(client, "_browser", None)
    if browser is not None and hasattr(browser, "close"):
        try:
            browser.close()
        except Exception:  # noqa: BLE001
            pass


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
                ac = AccountClient(a, client, cfg, store=self.store)
                if old:
                    ac.cooldown_until = old.cooldown_until
                    ac.rate_hits = old.rate_hits
                    dispose_client(old.client)
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
        """轮询选一个"健康"的可用账号，可排除 / 限定范围 / 优先指定。

        全部处于冷却时，只有冷却快结束（≤45 秒）才勉强放行，
        否则返回 None，让上层直接告诉用户"稍后再试"，避免继续撞限流。
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
                cooling = [c for c in cooling if c.cooldown_left() <= 45]
                items = cooling
            if allowed:
                items = [c for c in items if c.account["id"] in allowed]
            if not items:
                return None
            # 两级选号：库里 health 标记为 error/expired 的账号不参与轮询，
            # 只作兜底（好号全冷却时才轮到它们）。凭证过期的号只吃 600s
            # 内存冷却，冷却一过又会被选中——连续两次 auth 失败才自动停用，
            # 窗口期内请求反复撞死号（2026-10-29 冒烟实测）。
            # 重新登录或面板健康检查通过会把 health 置回 ok，恢复可选。
            good = [
                c
                for c in items
                if (c.account.get("health") or "ok") not in ("error", "expired")
            ]
            if good:
                items = good
            if prefer:
                # 粘性指向坏号时不再优先：换健康号重建会话（自动），
                # 成功后粘性自然更新
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

