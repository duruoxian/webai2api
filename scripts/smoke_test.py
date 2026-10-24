#!/usr/bin/env python3
"""webai2api 发版前冒烟测试：四家平台各发一条最小请求 + 一轮完整工具调用。

用法（先启动本地实例，默认 8800 开发端）：
    .venv\\Scripts\\python.exe scripts\\smoke_test.py
    .venv\\Scripts\\python.exe scripts\\smoke_test.py --base-url http://127.0.0.1:8800 --only deepseek,qwen
    .venv\\Scripts\\python.exe scripts\\smoke_test.py --key sk-xxx --skip-tools

流程（每个平台）：
  1. 最小对话：非流式发一条「回复：OK」，断言非空回复；
  2. 工具链：发 write 工具让模型发起调用 → 本脚本作为执行器真实写文件
     → 回填 tool 结果 → 断言模型基于结果给出最终回答；
  3. 顺带断言：带工具时回复 content 里没有泄漏任何工具标记。

API Key 取值顺序：--key > 环境变量 WEBAI2API_SMOKE_KEY >
本机开发端数据库（%LOCALAPPDATA%\\Programs\\webai2api\\data）。
任一平台失败退出码为 1。每次调用消耗真实账号额度，消息保持最小。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PLATFORMS = [
    ("deepseek", "deepseek-chat"),
    ("glm", "GLM-5.3-Flash"),
    ("kimi", "kimi-k2.6"),
    ("qwen", "qwen3.8-max"),
]

TOOL_MARKERS = ("<tool_call>", "</tool_call>", "<<<TOOL_CALL>>>")

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "把内容写入用户本地电脑上的指定文本文件",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string", "description": "文件的绝对路径"},
                "content": {"type": "string", "description": "要写入的完整内容"},
            },
            "required": ["filePath", "content"],
        },
    },
}


def _key_from_dev_db() -> str:
    """开发机便利：从本机 8800 实例的库里读第一个启用的 Key。"""
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return ""
    db = Path(base) / "Programs" / "webai2api" / "data" / "webai2api.db"
    if not db.exists():
        return ""
    try:
        import sqlite3

        c = sqlite3.connect(str(db))
        row = c.execute(
            "SELECT key FROM api_keys WHERE enabled=1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return str(row[0]) if row else ""
    except Exception:
        return ""


def _chat(base: str, key: str, body: dict, timeout: int) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, time.time() - t0


def _first_call(resp: dict) -> dict | None:
    try:
        tc = resp["choices"][0]["message"].get("tool_calls") or []
        return tc[0] if tc else None
    except Exception:
        return None


def smoke_platform(name: str, model: str, base: str, key: str,
                   timeout: int, with_tools: bool) -> list[str]:
    """跑一个平台，返回失败原因列表（空 = 通过）。"""
    errors: list[str] = []
    tool_path = Path(tempfile.gettempdir()) / f"webai2api_smoke_{name}.txt"
    marker = f"smoke-{name}-{int(time.time())}"

    # ── 1. 最小对话 ──
    try:
        resp, dt = _chat(base, key, {
            "model": model, "stream": False, "max_tokens": 50,
            "messages": [{"role": "user", "content": "请只回复两个字：OK"}],
        }, timeout)
        content = str(resp["choices"][0]["message"].get("content") or "")
        if not content.strip():
            errors.append(f"最小对话：空回复（{dt:.1f}s）")
        else:
            print(f"    最小对话 PASS（{dt:.1f}s）：{content.strip()[:30]!r}")
    except Exception as exc:
        errors.append(f"最小对话异常：{exc}")
        return errors

    if not with_tools:
        return errors

    # ── 2. 工具链第一跳：模型应发起 write 调用 ──
    try:
        tool_path.unlink(missing_ok=True)
        resp, dt = _chat(base, key, {
            "model": model, "stream": False,
            "messages": [{"role": "user",
                          "content": f"请用 write 工具把「{marker}」写入文件 "
                                     f"{tool_path}，写完文件即完成任务。"}],
            "tools": [WRITE_TOOL],
        }, timeout)
        msg = resp["choices"][0]["message"]
        call = _first_call(resp)
        if not call:
            errors.append(f"工具链：模型未发起工具调用（content={str(msg.get('content'))[:80]!r}）")
            return errors
        fn = call.get("function") or {}
        if fn.get("name") != "write":
            errors.append(f"工具链：调用了错误工具 {fn.get('name')!r}")
            return errors
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            errors.append("工具链：arguments 不是合法 JSON")
            return errors
        leaked = [m for m in TOOL_MARKERS if m in str(msg.get("content") or "")]
        if leaked:
            errors.append(f"工具链：标记泄漏进 content：{leaked}")
        print(f"    工具调用 PASS（{dt:.1f}s）：write({list(args)[:2]}...)")

        # ── 3. 本脚本作为执行器真实写文件，回填结果 ──
        try:
            tool_path.write_text(str(args.get("content") or marker), encoding="utf-8")
            result = f"写入成功：{tool_path}"
        except Exception as exc:
            result = f"写入失败：{exc}"

        resp2, dt2 = _chat(base, key, {
            "model": model, "stream": False,
            "messages": [
                {"role": "user", "content": f"请用 write 工具把「{marker}」写入文件 "
                                            f"{tool_path}，写完文件即完成任务。"},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call.get("id"), "content": result},
            ],
            "tools": [WRITE_TOOL],
        }, timeout)
        final = str(resp2["choices"][0]["message"].get("content") or "")
        if marker not in (tool_path.read_text(encoding="utf-8") if tool_path.exists() else ""):
            errors.append("工具链：文件内容与要求不符")
        if not final.strip():
            errors.append(f"工具链：回填结果后模型未给出最终回答（{dt2:.1f}s）")
        elif any(m in final for m in TOOL_MARKERS):
            errors.append("工具链：最终回答里仍有工具标记")
        else:
            ok_file = tool_path.exists()
            print(f"    多轮回填 PASS（{dt2:.1f}s）：文件真实写入={ok_file}，"
                  f"最终回答={final.strip()[:40]!r}")
    except Exception as exc:
        errors.append(f"工具链异常：{exc}")
    finally:
        tool_path.unlink(missing_ok=True)
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="webai2api 发版前冒烟测试")
    ap.add_argument("--base-url", default="http://127.0.0.1:8800")
    ap.add_argument("--key", default=os.environ.get("WEBAI2API_SMOKE_KEY", ""))
    ap.add_argument("--only", default="", help="逗号分隔平台名，如 deepseek,qwen")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--skip-tools", action="store_true", help="只跑最小对话")
    args = ap.parse_args()

    key = args.key or _key_from_dev_db()
    if not key:
        print("错误：找不到 API Key（--key / WEBAI2API_SMOKE_KEY / 本机开发端库均未命中）")
        return 2
    only = {s.strip().lower() for s in args.only.split(",") if s.strip()}

    print(f"目标实例：{args.base_url}")
    failures = 0
    for name, model in PLATFORMS:
        if only and name not in only:
            continue
        print(f"\n[{name}] {model}")
        errs = smoke_platform(name, model, args.base_url, key,
                              args.timeout, with_tools=not args.skip_tools)
        if errs:
            failures += 1
            for e in errs:
                print(f"    ✗ {e}")
        else:
            print(f"    ✓ [{name}] 全部通过")

    ran = [(n, m) for n, m in PLATFORMS if not only or n in only]
    print(f"\n结果：{len(ran) - failures}/{len(ran)} 个平台通过")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
