"""SQLite 存储层：账号、API Key、用量记录、设置。

单文件数据库（默认 <项目>/data/webai2api.db），线程安全（每次操作独立连接）。
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id           TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    name         TEXT NOT NULL,
    auth_token   TEXT NOT NULL DEFAULT '',
    device_id    TEXT NOT NULL DEFAULT '',
    settings_token TEXT NOT NULL DEFAULT '',
    cookie       TEXT NOT NULL DEFAULT '',
    enabled      INTEGER NOT NULL DEFAULT 1,
    note         TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id           TEXT PRIMARY KEY,
    key          TEXT NOT NULL UNIQUE,
    name         TEXT NOT NULL,
    enabled      INTEGER NOT NULL DEFAULT 1,
    created_at   REAL NOT NULL,
    last_used_at REAL
);

CREATE TABLE IF NOT EXISTS usage_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    api_key_id   TEXT,
    api_key_name TEXT,
    account_id   TEXT,
    account_name TEXT,
    provider     TEXT,
    model        TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_log(ts);
CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_log(model);
CREATE INDEX IF NOT EXISTS idx_usage_ts_model ON usage_log(ts, model);
CREATE INDEX IF NOT EXISTS idx_usage_ts_account ON usage_log(ts, account_id);
CREATE INDEX IF NOT EXISTS idx_usage_ts_key ON usage_log(ts, api_key_id);
CREATE INDEX IF NOT EXISTS idx_usage_status ON usage_log(status);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now() -> float:
    return time.time()


def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            # WAL：读写并发更稳，长期跑不容易被锁住
            try:
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("PRAGMA synchronous=NORMAL")
            except Exception:
                pass

    def prune_usage(self, retain_days: int = 0) -> int:
        """删除超过保留期的用量记录，返回删除条数。

        retain_days <= 0 表示不清理。
        """
        if not retain_days or retain_days <= 0:
            return 0
        cutoff = _now() - retain_days * 86400
        with self._conn() as c:
            cur = c.execute("DELETE FROM usage_log WHERE ts < ?", (cutoff,))
            return cur.rowcount or 0

    def vacuum(self) -> None:
        """整理数据库文件（清理后调用可回收空间）。"""
        with self._conn() as c:
            c.execute("VACUUM")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── 账号 ──────────────────────────────────────────
    def list_accounts(self, provider: str | None = None) -> list[dict]:
        with self._conn() as c:
            if provider:
                rows = c.execute(
                    "SELECT * FROM accounts WHERE provider=? ORDER BY created_at", (provider,)
                ).fetchall()
            else:
                rows = c.execute("SELECT * FROM accounts ORDER BY provider, created_at").fetchall()
        return [dict(r) for r in rows]

    def get_account(self, account_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM accounts WHERE id=?", (account_id,)).fetchone()
        return dict(row) if row else None

    def add_account(self, provider: str, name: str, credentials: dict, note: str = "") -> dict:
        aid = gen_id("acc")
        t = _now()
        with self._conn() as c:
            c.execute(
                """INSERT INTO accounts
                   (id, provider, name, auth_token, device_id, settings_token, cookie,
                    enabled, note, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,1,?,?,?)""",
                (aid, provider, name,
                 credentials.get("auth_token", ""),
                 credentials.get("device_id", ""),
                 credentials.get("settings_token", ""),
                 credentials.get("cookie", ""),
                 note, t, t),
            )
        return self.get_account(aid)  # type: ignore[return-value]

    def update_account(self, account_id: str, **fields: Any) -> None:
        allowed = {"name", "auth_token", "device_id", "settings_token", "cookie",
                   "enabled", "note"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(v)
        if not sets:
            return
        sets.append("updated_at=?")
        vals.append(_now())
        vals.append(account_id)
        with self._conn() as c:
            c.execute(f"UPDATE accounts SET {', '.join(sets)} WHERE id=?", vals)

    def delete_account(self, account_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM accounts WHERE id=?", (account_id,))

    # ── API Key ───────────────────────────────────────
    def list_keys(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM api_keys ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def add_key(self, name: str, key: str | None = None) -> dict:
        kid = gen_id("key")
        key = key or ("sk-" + uuid.uuid4().hex + uuid.uuid4().hex[:8])
        with self._conn() as c:
            c.execute(
                "INSERT INTO api_keys (id, key, name, enabled, created_at) VALUES (?,?,?,1,?)",
                (kid, key, name, _now()),
            )
        return {"id": kid, "key": key, "name": name, "enabled": 1}

    def find_key(self, key: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
        return dict(row) if row else None

    def touch_key(self, key_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE api_keys SET last_used_at=? WHERE id=?", (_now(), key_id))

    def set_key_enabled(self, key_id: str, enabled: bool) -> None:
        with self._conn() as c:
            c.execute("UPDATE api_keys SET enabled=? WHERE id=?", (1 if enabled else 0, key_id))

    def delete_key(self, key_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM api_keys WHERE id=?", (key_id,))

    # ── 用量 ──────────────────────────────────────────
    def log_usage(self, **kw: Any) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO usage_log
                   (ts, api_key_id, api_key_name, account_id, account_name, provider,
                    model, prompt_tokens, completion_tokens, total_tokens, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (_now(),
                 kw.get("api_key_id"), kw.get("api_key_name"),
                 kw.get("account_id"), kw.get("account_name"), kw.get("provider"),
                 kw.get("model"),
                 int(kw.get("prompt_tokens", 0)),
                 int(kw.get("completion_tokens", 0)),
                 int(kw.get("total_tokens", 0)),
                 kw.get("status", "ok")),
            )

    def usage_summary(self, since_ts: float | None = None,
                      until_ts: float | None = None) -> dict:
        """总览：总计 + 按模型 + 按账号 + 按 Key + 近 24h + 每日。

        since_ts / until_ts 不为空时，只统计该时间区间内的数据。
        """
        conds, args = [], []
        if since_ts is not None:
            conds.append("ts>=?")
            args.append(since_ts)
        if until_ts is not None:
            conds.append("ts<=?")
            args.append(until_ts)
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        args = tuple(args)
        with self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) n, "
                "COALESCE(SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END),0) ok, "
                "COALESCE(SUM(prompt_tokens),0) p, "
                "COALESCE(SUM(completion_tokens),0) c, COALESCE(SUM(total_tokens),0) t "
                "FROM usage_log" + where, args
            ).fetchone()
            by_model = c.execute(
                "SELECT model, COUNT(*) n, COALESCE(SUM(prompt_tokens),0) p, "
                "COALESCE(SUM(completion_tokens),0) c, COALESCE(SUM(total_tokens),0) t "
                "FROM usage_log" + where + " GROUP BY model ORDER BY t DESC", args
            ).fetchall()
            by_account = c.execute(
                "SELECT account_name, provider, COUNT(*) n, COALESCE(SUM(total_tokens),0) t "
                "FROM usage_log" + where + " GROUP BY account_id ORDER BY t DESC", args
            ).fetchall()
            by_key = c.execute(
                "SELECT api_key_name, COUNT(*) n, COALESCE(SUM(total_tokens),0) t "
                "FROM usage_log" + where + " GROUP BY api_key_id ORDER BY t DESC", args
            ).fetchall()
            since24 = _now() - 86400
            last24 = c.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(total_tokens),0) t FROM usage_log WHERE ts>=?",
                (since24,),
            ).fetchone()
            daily = c.execute(
                "SELECT strftime('%Y-%m-%d', ts, 'unixepoch', 'localtime') d, "
                "COUNT(*) n, COALESCE(SUM(total_tokens),0) t "
                "FROM usage_log WHERE ts>=? GROUP BY d ORDER BY d",
                (_now() - 86400 * 182,),
            ).fetchall()
        return {
            "total": dict(total),
            "by_model": [dict(r) for r in by_model],
            "by_account": [dict(r) for r in by_account],
            "by_key": [dict(r) for r in by_key],
            "last24h": dict(last24),
            "daily": [dict(r) for r in daily],
        }

    def recent_usage(self, limit: int = 100, offset: int = 0,
                     model: str = "", status: str = "") -> list[dict]:
        """最近请求，支持分页与简单筛选。"""
        conds, args = [], []
        if model:
            conds.append("model = ?")
            args.append(model)
        if status:
            conds.append("status = ?")
            args.append(status)
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        args.extend([max(1, int(limit)), max(0, int(offset))])
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM usage_log" + where + " ORDER BY id DESC LIMIT ? OFFSET ?",
                tuple(args),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_usage(self, model: str = "", status: str = "") -> int:
        conds, args = [], []
        if model:
            conds.append("model = ?")
            args.append(model)
        if status:
            conds.append("status = ?")
            args.append(status)
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) n FROM usage_log" + where, tuple(args)).fetchone()
        return int(row["n"]) if row else 0

    def list_used_models(self, limit: int = 100) -> list[dict]:
        """用量里实际出现过的模型（按最近使用倒序）。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT model, MAX(ts) last_ts, COUNT(*) n FROM usage_log "
                "WHERE model IS NOT NULL AND model <> '' "
                "GROUP BY model ORDER BY last_ts DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── 设置 ──────────────────────────────────────────
    def get_setting(self, key: str, default: str = "") -> str:
        with self._conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO settings (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
