"""授权签发工具（只在你本地使用，不要随软件分发）。

首次使用：
    python tools/keygen.py init

生成一对密钥：
    private_key.txt   ← 私钥（32 字节 hex），务必自己保存好，不要提交到仓库
    public_key.txt    ← 公钥（base64），填到 webai2api/license.py 的 PUBLIC_KEY_B64

签发授权：
    python tools/keygen.py issue <机器码> <到期日 YYYY-MM-DD> [客户名] [-o license.key]

查看授权内容：
    python tools/keygen.py show license.key
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webai2api import ed25519  # noqa: E402

PRIV = ROOT / "tools" / "private_key.txt"
PUB = ROOT / "tools" / "public_key.txt"


def _payload(doc: dict) -> bytes:
    body = {k: v for k, v in doc.items() if k != "sig"}
    return json.dumps(body, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def cmd_init(_args) -> None:
    if PRIV.exists():
        print(f"私钥已存在：{PRIV}（如需重置请先手动删除）")
        seed = bytes.fromhex(PRIV.read_text().strip())
    else:
        seed = os.urandom(32)
        PRIV.write_text(seed.hex(), encoding="utf-8")
        print(f"已生成私钥：{PRIV}")
    pk = ed25519.public_key(seed)
    b64 = base64.b64encode(pk).decode()
    PUB.write_text(b64, encoding="utf-8")
    print(f"已生成公钥：{PUB}")
    print(f"\n把下面这行填到 webai2api/license.py 的 PUBLIC_KEY_B64：\n{b64}\n")


def _load_priv() -> bytes:
    if not PRIV.exists():
        print("找不到私钥，请先运行：python tools/keygen.py init")
        raise SystemExit(1)
    return bytes.fromhex(PRIV.read_text().strip())


def cmd_issue(args) -> None:
    seed = _load_priv()
    doc = {
        "machine": args.machine.strip().upper(),
        "expires": args.expires.strip(),
        "edition": args.edition,
        "name": (args.name or "").strip(),
        "issued_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
    }
    sig = ed25519.sign(seed, _payload(doc))
    doc["sig"] = base64.b64encode(sig).decode()
    out = Path(args.out or "license.key")
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已签发：{out}")
    print(json.dumps({k: v for k, v in doc.items() if k != "sig"},
                     ensure_ascii=False, indent=2))


def cmd_show(args) -> None:
    doc = json.loads(Path(args.file).read_text(encoding="utf-8"))
    seed = _load_priv() if PRIV.exists() else None
    ok = False
    if seed:
        ok = ed25519.verify(ed25519.public_key(seed), _payload(doc),
                            base64.b64decode(doc.get("sig", "")))
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    print("签名校验：", "通过" if ok else "失败（或缺少私钥）")


def main() -> None:
    ap = argparse.ArgumentParser(description="webai2api 授权签发工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="生成密钥对")
    p_init.set_defaults(func=cmd_init)

    p_issue = sub.add_parser("issue", help="签发授权文件")
    p_issue.add_argument("machine", help="客户机器码，如 ABCD-EFGH-IJKL-MNOP")
    p_issue.add_argument("expires", help="到期日 YYYY-MM-DD")
    p_issue.add_argument("name", nargs="?", default="", help="客户名（可选）")
    p_issue.add_argument("-o", "--out", default="license.key")
    p_issue.add_argument("--edition", default="lite")
    p_issue.set_defaults(func=cmd_issue)

    p_show = sub.add_parser("show", help="查看/校验授权文件")
    p_show.add_argument("file")
    p_show.set_defaults(func=cmd_show)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
