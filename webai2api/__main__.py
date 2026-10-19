"""python -m webai2api 入口。"""
from __future__ import annotations

import argparse
import sys


def _force_utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main() -> None:
    _force_utf8()
    parser = argparse.ArgumentParser(description="webai2api —— 网页版 AI 反代为 OpenAI 兼容 API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    import uvicorn

    if args.reload:
        uvicorn.run("webai2api.app:create_default_app", factory=True,
                    host=args.host, port=args.port, reload=True)
    else:
        from .app import create_default_app
        uvicorn.run(create_default_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
