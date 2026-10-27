"""清理 DeepSeek 上游历史会话（防风控用）。

用法（在项目根目录执行）：

    # 1. 先看有哪些会话（不删除）
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py --list

    # 2. 预演：统计会删多少（不删除）
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py --dry-run

    # 3. 真正删除
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py

    # 保留标题含关键字的会话
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py --keep "重要,项目"

    # 只清某个账号（账号名或 id 均可）
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py --account 张严文4536

说明：
- 删除的是 **上游网页版会话历史**，本地数据库（账号/Key/用量）完全不受影响。
- 删除后，本地指向这些会话的缓存条目会自动失效，下次对话重新开新会话。
- 已封禁账号的 token 失效，无法清理，脚本会跳过并提示。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WEBAI2API_EDITION", "pro")

from webai2api.adapters.deepseek import pool as ds_pool  # noqa: E402
from webai2api.store import Store  # noqa: E402


def _data_dir() -> Path:
    """定位数据目录：优先已安装的开发端，其次源码目录。"""
    candidates = [
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\webai2api\data")),
        ROOT / "data",
    ]
    for d in candidates:
        if (d / "webai2api.db").exists():
            return d
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="清理 DeepSeek 上游历史会话")
    ap.add_argument("--list", action="store_true", help="只列出会话，不删除")
    ap.add_argument("--dry-run", action="store_true", help="预演：统计不删除")
    ap.add_argument("--keep", default="", help="逗号分隔，标题命中的保留")
    ap.add_argument("--account", default="", help="只处理指定账号（名称或 id）")
    ap.add_argument("--data-dir", default="", help="指定数据目录")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    data_dir = Path(args.data_dir) if args.data_dir else _data_dir()
    db = data_dir / "webai2api.db"
    if not db.exists():
        print(f"找不到数据库: {db}")
        return 1

    print(f"数据目录: {data_dir}")
    store = Store(db)
    pool = ds_pool.build_pool(store, logging.getLogger("purge"))
    accounts = pool.available()
    if not accounts:
        print("没有可用账号")
        return 1

    keep = [x.strip() for x in args.keep.split(",") if x.strip()]
    total_deleted = total_failed = total_dryrun = 0

    for ac in accounts:
        name = ac.account.get("name", "")
        aid = ac.account.get("id", "")
        if args.account and args.account not in (name, aid):
            continue

        print()
        print(f"=== {name} ===")
        client = ac.client

        try:
            sessions = client.list_upstream_sessions()
        except Exception as exc:
            print(f"  无法读取会话（可能 token 失效/已封）: {str(exc)[:160]}")
            continue

        print(f"  会话总数: {len(sessions)}")
        if args.list:
            for s in sessions[:100]:
                print(
                    f"    - {str(s.get('id'))[:14]}  {(s.get('title') or '(无标题)')[:40]}"
                )
            if len(sessions) > 100:
                print(f"    ... 还有 {len(sessions) - 100} 条")
            continue

        try:
            res = client.purge_upstream_sessions(
                keep_titles=tuple(keep), dry_run=args.dry_run
            )
        except Exception as exc:
            print(f"  清理失败: {str(exc)[:160]}")
            continue

        total_deleted += res.get("deleted", 0)
        total_failed += res.get("failed", 0)
        if args.dry_run:
            total_dryrun += res.get("matched", 0)
        tag = "将删除" if args.dry_run else "已删除"
        print(
            f"  总计 {res['total']} 条 | {tag} {res['matched']} 条 | "
            f"保留 {res['kept']} 条 | 失败 {res['failed']} 条"
        )

    if not args.list:
        print()
        if args.dry_run:
            print(
                f"预演完成：共将删除 {total_dryrun} 条（未实际删除，去掉 --dry-run 才会删）"
            )
        else:
            print(f"完成：共删除 {total_deleted} 条，失败 {total_failed} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
