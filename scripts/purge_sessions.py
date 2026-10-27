"""清理 DeepSeek / GLM / Kimi 上游历史会话（防风控用）。

用法（项目根目录执行）：

    # 列出各平台各账号的会话（不删除）
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py --list

    # 只看某个平台
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py --list --provider glm

    # 预演（统计要删多少，不真删）
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py --dry-run

    # 真正删除
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py

    # 保留标题含关键字的会话
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py --keep "重要,项目"

    # 只处理某个账号
    .venv\\Scripts\\python.exe scripts\\purge_sessions.py --account GLM0393

说明：
- 删除的是**上游网页版会话历史**，本地数据库（账号/Key/用量）不受影响。
- 删除后本地指向这些会话的缓存条目自动失效，下次对话会开新会话。
- 已封禁账号 token 失效，脚本会跳过并提示。
- Qwen 暂不支持（列表接口被 WAF 拦截）。
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

from webai2api.store import Store  # noqa: E402

# 平台 → 该平台 pool 的构建函数模块路径
PROVIDERS = ("deepseek", "glm", "kimi")


def _data_dir() -> Path:
    candidates = [
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\webai2api\data")),
        ROOT / "data",
    ]
    for d in candidates:
        if (d / "webai2api.db").exists():
            return d
    return candidates[0]


def _pool_for(provider: str, store: Store, logger):
    if provider == "deepseek":
        from webai2api.adapters.deepseek import pool
    elif provider == "glm":
        from webai2api.adapters.chatglm import pool
    elif provider == "kimi":
        from webai2api.adapters.kimi import pool
    else:
        raise SystemExit(f"不支持的平台: {provider}")
    return pool.build_pool(store, logger)


def main() -> int:
    ap = argparse.ArgumentParser(description="清理上游历史会话")
    ap.add_argument("--list", action="store_true", help="只列出会话，不删除")
    ap.add_argument("--dry-run", action="store_true", help="预演：统计不删除")
    ap.add_argument("--keep", default="", help="逗号分隔，标题命中的保留")
    ap.add_argument("--account", default="", help="只处理指定账号（名称或 id）")
    ap.add_argument(
        "--provider",
        default="",
        help="只处理指定平台：deepseek / glm / kimi（默认全部）",
    )
    ap.add_argument("--data-dir", default="", help="指定数据目录")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")

    data_dir = Path(args.data_dir) if args.data_dir else _data_dir()
    db = data_dir / "webai2api.db"
    if not db.exists():
        print(f"找不到数据库: {db}")
        return 1

    print(f"数据目录: {data_dir}")
    store = Store(db)
    keep = [x.strip() for x in args.keep.split(",") if x.strip()]
    targets = [args.provider] if args.provider else list(PROVIDERS)

    grand_total = grand_deleted = grand_failed = grand_dry = 0

    for prov in targets:
        print()
        print("#" * 50)
        print(f"# {prov.upper()}")
        print("#" * 50)
        try:
            pool = _pool_for(prov, store, logging.getLogger(prov))
        except Exception as exc:
            print(f"  初始化失败: {exc}")
            continue

        accounts = pool.available()
        if not accounts:
            print("  （无账号）")
            continue

        for ac in accounts:
            name = ac.account.get("name", "")
            aid = ac.account.get("id", "")
            if args.account and args.account not in (name, aid):
                continue

            print()
            print(f"=== {name} ===")
            client = ac.client
            if not hasattr(client, "list_upstream_sessions"):
                print("  该平台暂不支持会话管理")
                continue

            try:
                sessions = client.list_upstream_sessions()
            except Exception as exc:
                print(f"  无法读取会话（token 失效/已封？）: {str(exc)[:150]}")
                continue

            grand_total += len(sessions)
            print(f"  会话总数: {len(sessions)}")

            if args.list:
                for s in sessions[:100]:
                    sid = str(s.get("id") or s.get("conversation_id") or "")[:14]
                    title = str(s.get("title") or s.get("name") or "(无标题)")[:40]
                    print(f"    - {sid}  {title}")
                if len(sessions) > 100:
                    print(f"    ... 还有 {len(sessions) - 100} 条")
                continue

            try:
                res = client.purge_upstream_sessions(
                    keep_titles=tuple(keep), dry_run=args.dry_run
                )
            except Exception as exc:
                print(f"  清理失败: {str(exc)[:150]}")
                continue

            grand_deleted += res.get("deleted", 0)
            grand_failed += res.get("failed", 0)
            if args.dry_run:
                grand_dry += res.get("matched", 0)
            tag = "将删除" if args.dry_run else "已删除"
            print(
                f"  总计 {res['total']} 条 | {tag} {res['matched']} 条 | "
                f"保留 {res['kept']} 条 | 失败 {res['failed']} 条"
            )

    if not args.list:
        print()
        if args.dry_run:
            print(
                f"预演完成：共将删除 {grand_dry} 条（未实际删除，去掉 --dry-run 才会删）"
            )
        else:
            print(
                f"完成：共删除 {grand_deleted} 条，失败 {grand_failed} 条"
                f"（扫描 {grand_total} 条）"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
