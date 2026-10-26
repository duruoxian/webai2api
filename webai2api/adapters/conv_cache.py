"""三家适配器（chatglm / kimi / qwen）共用的上游会话缓存。

此前 `_load_conv_cache / _persist_conv / _cached_conv / _store_conv /
risk_stats 的缓存部分` 在三个 client 里是逐字复制粘贴的三份（约 120 行
× 3），唯一差异是落盘字段名（conversation_id / chat_id）与附加字段
（无 / last_message_id / response_id）。

统一语义：
- 内存字典 key → (session_id, last_used, turns, extra)，统一四元组；
- 节流落盘（5s，tmp + 原子替换，重启后仍能复用会话）；载入按 TTL 过滤、
  超量按最旧淘汰；
- cached()：TTL 过期 / 轮次达到上限 → 丢弃换新（bump stale）；命中 →
  触达时间刷新（bump hit）；未命中 bump miss 由**调用方**负责（与历史
  行为一致——miss 的判定口径各家调用点略有差异）；
- store()：容量淘汰 + 轮次累计 + 同会话空 extra 保留旧值（Kimi/Qwen
  语义，GLM 无 extra 不受影响）+ 首见 bump created；
- 落盘字段名参数化（id_field / extra_field），**旧缓存文件格式完全兼容**。
"""
from __future__ import annotations

import json
import os
import threading
import time
from logging import Logger

_PERSIST_MIN_INTERVAL = 5.0


class ConversationCache:
    def __init__(
        self,
        *,
        config,
        logger: Logger,
        tag: str,
        id_field: str,
        persist_path: str = "",
        extra_field: str = "",
        bump=None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._tag = tag
        self._id_field = id_field
        self._extra_field = extra_field
        self._bump_cb = bump
        self._path = str(persist_path or "")
        self._lock = threading.Lock()
        # 统一四元组：(session_id, last_used, turns, extra)
        self._cache: dict[str, tuple[str, float, int, str]] = {}
        self._dirty = False
        self._last_persist = 0.0
        if self._path:
            self._load()

    # ── 内部 ───────────────────────────────────────────
    def _bump(self, event: str) -> None:
        if self._bump_cb:
            self._bump_cb(event)

    def _load(self) -> None:
        path = self._path
        if not path:
            return
        try:
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("%s 会话缓存载入失败: %s", self._tag, exc)
            return
        if not isinstance(raw, dict):
            return
        now = time.time()
        loaded = 0
        for key, item in raw.items():
            if not isinstance(key, str) or not isinstance(item, dict):
                continue
            sid = item.get(self._id_field)
            if not isinstance(sid, str) or not sid:
                continue
            try:
                ts = float(item.get("last_used") or now)
                turns = int(item.get("turns") or 0)
            except (TypeError, ValueError):
                continue
            if now - ts > self._config.session_cache_ttl:
                continue
            extra = str(item.get(self._extra_field) or "") if self._extra_field else ""
            self._cache[key] = (sid, ts, turns, extra)
            loaded += 1
        while len(self._cache) > self._config.session_cache_max_size:
            oldest = min(self._cache.items(), key=lambda kv: kv[1][1])[0]
            self._cache.pop(oldest, None)
        if loaded:
            self._logger.info("%s 会话缓存恢复 %d 条（%s）", self._tag, loaded, path)

    def _persist(self, force: bool = False) -> None:
        """调用方需持有 self._lock。节流写盘，异常只记 debug。"""
        path = self._path
        if not path:
            return
        now = time.time()
        if not force and now - self._last_persist < _PERSIST_MIN_INTERVAL:
            self._dirty = True
            return
        try:
            data = {}
            for k, (sid, ts, turns, extra) in self._cache.items():
                item = {self._id_field: sid, "last_used": ts, "turns": turns}
                if self._extra_field:
                    item[self._extra_field] = extra
                data[k] = item
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
            self._dirty = False
            self._last_persist = now
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("%s 会话缓存落盘失败: %s", self._tag, exc)

    # ── 对外 ───────────────────────────────────────────
    def cached(self, key: str) -> tuple[str, str]:
        """查缓存并触达。返回 (session_id, extra)；未命中/过期/超轮次返回 ("", "")。"""
        if not self._config.session_cache_enabled:
            return "", ""
        with self._lock:
            item = self._cache.get(key)
            if not item:
                return "", ""
            sid, ts, turns, extra = item
            if time.time() - ts > self._config.session_cache_ttl:
                self._cache.pop(key, None)
                self._persist()
                self._bump("stale")
                return "", ""
            max_turns = int(getattr(self._config, "max_turns_per_session", 40) or 40)
            if turns >= max_turns:
                self._cache.pop(key, None)
                self._persist()
                self._bump("stale")
                self._logger.info("会话轮次已达上限(%d) → 新建会话", max_turns)
                return "", ""
            self._cache[key] = (sid, time.time(), turns, extra)
            self._persist()
            self._bump("hit")
            return sid, extra

    def store(self, key: str, session_id: str, extra: str = "") -> None:
        """写入/更新会话；同一会话重复 store 时轮次 +1、空 extra 保留旧值。"""
        if not self._config.session_cache_enabled or not session_id:
            return
        with self._lock:
            if len(self._cache) >= self._config.session_cache_max_size:
                oldest = min(self._cache.items(), key=lambda kv: kv[1][1])[0]
                self._cache.pop(oldest, None)
            prev = self._cache.get(key)
            turns = (prev[2] + 1) if prev and prev[0] == session_id else 1
            first_time = prev is None or prev[0] != session_id
            kept = extra or (prev[3] if prev and prev[0] == session_id else "")
            self._cache[key] = (session_id, time.time(), turns, kept)
            self._persist()
            if first_time:
                self._bump("created")

    def flush(self) -> None:
        """强制落盘（关机前调用）。"""
        with self._lock:
            if self._path and self._dirty:
                self._persist(force=True)

    def collect_stale(self) -> list[tuple[str, str]]:
        """取并弹出「TTL 过期或轮次达上限」的会话，供低频批量清理用。

        返回 [(key, session_id), ...]；弹出后立即强制落盘。
        """
        now = time.time()
        ttl = self._config.session_cache_ttl
        max_turns = int(getattr(self._config, "max_turns_per_session", 40) or 40)
        stale: list[tuple[str, str]] = []
        with self._lock:
            for key, (sid, ts, turns, _extra) in list(self._cache.items()):
                if now - ts > ttl or turns >= max_turns:
                    stale.append((key, sid))
                    self._cache.pop(key, None)
            if stale:
                self._persist(force=True)
        return stale

    def stats(self) -> dict:
        """缓存规模与活跃度（risk_stats 的公共部分）。"""
        with self._lock:
            total = len(self._cache)
            now = time.time()
            active = sum(
                1 for (_s, ts, _t, _e) in self._cache.values()
                if now - ts <= self._config.session_cache_ttl
            )
            return {"cache_size": total, "active_sessions": active}
