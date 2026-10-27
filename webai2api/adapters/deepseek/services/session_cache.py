"""
DeepSeek 会话缓存。

基于「对话指纹」（system 首段 + 首条 user 首段 + model + key_id）做 key，
映射到 DeepSeek chat_session_id。指纹不随对话轮次变化，因此同一会话的
每一轮（含工具轮）都能命中同一个上游会话。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("deepseek2api.cache")

_WS_RE = re.compile(r"\s+")


@dataclass(slots=True)
class SessionState:
    session_id: str
    turn_count: int = 0
    last_message_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    msg_count: int = 0
    history_hash: str = ""
    # 该上游会话累计消耗的 token，「快满了就换会话」依据
    used_tokens: int = 0


def prefix_hash(messages: list[dict], model: str = "") -> str:
    canonical = json.dumps(
        {"model": model, "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _content_text(content: object) -> str:
    """把 message.content（str 或多模态 list）拍平成纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _norm(text: str, limit: int) -> str:
    return _WS_RE.sub(" ", text).strip()[:limit]


def conv_fingerprint(messages: list[dict], model: str, key_id: str = "") -> str:
    """
    对话指纹：同一会话的所有轮次返回同一个 key。

    只用 system 消息前 200 字符 + 第一条 user 消息前 500 字符 + model + key_id，
    刻意不使用完整 messages（否则每轮必变）。
    """
    system_text = ""
    first_user_text = ""
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role == "system" and not system_text:
                system_text = _content_text(m.get("content"))
            elif role == "user" and not first_user_text:
                first_user_text = _content_text(m.get("content"))
            if system_text and first_user_text:
                break

    canonical = json.dumps(
        {
            "model": model or "",
            "key_id": key_id or "",
            "system": _norm(system_text, 200),
            "first_user": _norm(first_user_text, 500),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SessionCache:
    """Thread-safe LRU cache: conv fingerprint → SessionState。"""

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: int = 7200,
        persist_path: str | Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._store: OrderedDict[str, SessionState] = OrderedDict()
        self._max_size = max(1, max_size)
        self._ttl = ttl_seconds
        self._persist_path: Path | None = Path(persist_path) if persist_path else None
        # 落盘节流：内存变更后最多每 PERSIST_MIN_INTERVAL 秒写一次，避免每请求全量写盘
        self._dirty = False
        self._last_persist = 0.0
        self._persist_min_interval = 5.0
        if self._persist_path is not None:
            self._load_from_disk()

    # ── persistence ────────────────────────────────────────

    def _load_from_disk(self) -> None:
        path = self._persist_path
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Session cache load failed (%s): %s", path, exc)
            return
        if not isinstance(raw, dict):
            return
        now = time.time()
        loaded = 0
        for key, item in raw.items():
            if not isinstance(key, str) or not isinstance(item, dict):
                continue
            session_id = item.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                continue
            try:
                state = SessionState(
                    session_id=session_id,
                    turn_count=int(item.get("turn_count") or 0),
                    last_message_id=item.get("last_message_id"),
                    created_at=float(item.get("created_at") or now),
                    last_used=float(item.get("last_used") or now),
                    msg_count=int(item.get("msg_count") or 0),
                    history_hash=str(item.get("history_hash") or ""),
                    used_tokens=int(item.get("used_tokens") or 0),
                )
            except (TypeError, ValueError):
                continue
            if now - state.last_used > self._ttl:
                continue
            self._store[key] = state
            loaded += 1
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)
        if loaded:
            logger.info("Session cache restored %d entr(ies) from %s", loaded, path)

    def _persist_locked(self, force: bool = False) -> None:
        """调用方需持有 self._lock。异常只记 debug，不抛。

        默认节流：距上次落盘不足 _persist_min_interval 秒时只置脏标记，
        由后续调用补写；clear()/delete 等关键操作传 force=True 立即写。
        """
        path = self._persist_path
        if path is None:
            return
        now = time.time()
        if not force and now - self._last_persist < self._persist_min_interval:
            self._dirty = True
            return
        try:
            data = {
                k: {
                    "session_id": v.session_id,
                    "turn_count": v.turn_count,
                    "last_message_id": v.last_message_id,
                    "created_at": v.created_at,
                    "last_used": v.last_used,
                    "msg_count": v.msg_count,
                    "history_hash": v.history_hash,
                    "used_tokens": v.used_tokens,
                }
                for k, v in self._store.items()
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            os.replace(tmp, path)
            self._dirty = False
            self._last_persist = now
        except Exception as exc:
            logger.debug("Session cache persist failed (%s): %s", path, exc)

    def flush(self) -> None:
        """强制落盘（关机/清理前调用）。"""
        with self._lock:
            if self._persist_path is not None and self._dirty:
                self._persist_locked(force=True)

    def get(self, key: str) -> SessionState | None:
        with self._lock:
            state = self._store.get(key)
            if state is None:
                return None
            if time.time() - state.last_used > self._ttl:
                del self._store[key]
                logger.debug("Session expired: %s...", key[:16])
                return None
            self._store.move_to_end(key)
            state.last_used = time.time()
            return state

    def put(self, key: str, state: SessionState) -> None:
        with self._lock:
            self._store[key] = state
            self._store.move_to_end(key)
            self._evict()
            self._persist_locked()

    def delete(self, key: str) -> SessionState | None:
        with self._lock:
            removed = self._store.pop(key, None)
            if removed is not None:
                self._persist_locked()
            return removed

    def delete_by_session_id(self, session_id: str) -> int:
        """删除所有指向给定 session_id 的缓存条目，返回删除数量。"""
        if not session_id:
            return 0
        removed = 0
        with self._lock:
            for k in [k for k, v in self._store.items() if v.session_id == session_id]:
                del self._store[k]
                removed += 1
            if removed:
                self._persist_locked()
        return removed

    def _evict(self) -> None:
        while len(self._store) > self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("LRU evicted: %s...", evicted_key[:16])

    def clear(self) -> None:
        """清空全部缓存条目（关机清理用）。"""
        with self._lock:
            self._store.clear()
            self._persist_locked(force=True)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def active_session_ids(self) -> set[str]:
        with self._lock:
            now = time.time()
            return {
                s.session_id
                for s in self._store.values()
                if now - s.last_used <= self._ttl
            }

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            active = sum(
                1 for s in self._store.values() if now - s.last_used <= self._ttl
            )
            return {
                "total": len(self._store),
                "active": active,
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
            }
