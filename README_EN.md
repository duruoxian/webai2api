# webai2api

> 中文文档见 [README.md](README.md)

Turn the web versions of DeepSeek, Zhipu (ChatGLM), Kimi and Qwen into a
unified OpenAI-compatible API, with a Chinese admin panel. Runs entirely on
your machine; ships as a ready-to-run Windows executable.

## Features

- **Multi-platform**: DeepSeek web, Zhipu chatglm.cn, Kimi (kimi.com), Qwen (chat.qwen.ai) — 10 models
- **Multi-account**: independent credentials per account, pooled and rotated; does not touch your own browser sessions
- **Multi-key**: issue separate API keys per client/use case; enable, disable or delete at any time
- **Tool calling**: supported on all four platforms — opencode, Claude Code and ZCode agents can read/write files and run commands
- **Image input**: DeepSeek and GLM accept images in messages (uploaded to the platform automatically)
- **Usage stats**: input/output tokens per model, account and key
- **Admin panel** (Chinese): manage accounts, keys and usage at `/admin`
- **Anthropic-compatible**: `POST /v1/messages` (tools + streaming) for native Claude Code / ZCode access
- **Free and open source**: MIT license; no sales, no rentals, no license keys

> Disclaimer: intended for personal study and local use only. Do not
> redistribute or offer it as a public service. See [DISCLAIMER.md](DISCLAIMER.md).

## Quick start (Windows exe)

1. Download `webai2api-win64.zip` from
   [Releases](https://github.com/duruoxian/webai2api/releases), unzip and run
   `webai2api.exe` (no Python needed).
2. SmartScreen may warn about an unsigned binary on first run — this is the
   default Windows prompt, not a virus. Click "More info" → "Run anyway".
   If Windows flags the file as downloaded, right-click the exe → Properties
   → "Unblock".
3. The service starts and opens the admin panel
   (`http://127.0.0.1:8800/admin`, default password `admin`). Closing the
   panel window stops the service.

Source code also runs on Python 3.12+ (Windows/Linux/macOS):

```bash
git clone <this repo>
cd webai2api
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e .
python -m webai2api            # default http://127.0.0.1:8800
```

## Add accounts

Two ways, from the panel's account page:

**Built-in browser login (recommended)**

1. Create an account with a name only, leave the credential fields empty;
2. Click the account's "浏览器登录" (browser login) button — an embedded
   WebView2 window opens (not your daily browser);
3. Sign in to the platform normally (verification code / password / QR);
4. Credentials are captured and saved automatically.

Each account gets its own browser profile directory
(`data/browser_profiles/<account id>`), so accounts never interfere with
each other or with your own browser.

**Manual credentials**

Paste this in the platform page's F12 console (DeepSeek example):

```javascript
copy(JSON.stringify({
  t: localStorage.getItem('userToken'),
  d: localStorage.getItem('deepseek-device-id:chat'),
  s: localStorage.getItem('settingsJwt'),
  c: document.cookie
}));
```

| Panel field | Value |
|---|---|
| auth_token | `value` of `t` (without the `Bearer ` prefix) |
| device_id | `d` |
| settings_token | `value.jwt` of `s` |
| cookie | `c` |

## Connect clients

Available models: `deepseek-chat` / `deepseek-reasoner` / `GLM-5.3` /
`GLM-5.3-Flash` / `kimi-k3` / `kimi-k2.8` / `kimi-k2.6` / `qwen3.7-plus` /
`qwen3.8-max` / `qwen3.8-omni-flash`.

> `qwen3.8-max` is experimental: unstable under web proxying (occasional
> incoherent output, a model-side issue). Prefer `qwen3.7-plus` when quality
> matters.

**OpenAI-compatible clients** (opencode, Cherry Studio, ...):

- Base URL: `http://127.0.0.1:8800/v1`
- API key: a `sk-...` key generated in the panel

`127.0.0.1` means "the machine running webai2api" — everyone runs their own
instance and points their clients at their own.

**Anthropic-native clients** (Claude Code / ZCode) use `/v1/messages`:

- `ANTHROPIC_BASE_URL`: `http://127.0.0.1:8800`
- `ANTHROPIC_AUTH_TOKEN` (or `ANTHROPIC_API_KEY`): your `sk-...` key
- Example: `claude -p "hi" --model webai2api/GLM-5.3-Flash`

**opencode example** — the panel's key page can write the provider config
into opencode with one click (with backup and removal). Manual setup in
`~/.config/opencode/opencode.jsonc` (V2 format):

```jsonc
"webai2api": {
  "package": "@opencode/ai/providers/openai-compatible",
  "name": "webai2api",
  "settings": {
    "baseURL": "http://127.0.0.1:8800/v1",
    "apiKey": "sk-your-key",
    "maxRetries": 0
  },
  "compatibility": { "streamUsage": true },
  "models": {
    "deepseek-chat": { "name": "DeepSeek Chat (web)",
      "capabilities": { "input": ["image", "text"], "output": ["text"],
        "reasoning": { "supported": false } },
      "limit": { "context": 1000000, "output": 384000 } },
    "deepseek-reasoner": { "name": "DeepSeek Reasoner (web)",
      "capabilities": { "input": ["image", "text"], "output": ["text"],
        "reasoning": { "supported": true } },
      "limit": { "context": 1000000, "output": 384000 } }
  }
}
```

## Building the exe

```bash
build-exe.cmd
```

Output: `dist/webai2api/webai2api.exe` (~60 MB). Proxying itself is pure
HTTP and needs no browser at runtime; only credential login uses the
system-installed WebView2 in an embedded window.

## Maintainer guide

```powershell
# 1. Bump the version in 4 places: webai2api/__init__.py / pyproject.toml / web/index.html (?v= twice)
# 2. Full test suite (must be green)
.venv\Scripts\python.exe -m pytest tests -q
# 3. Local build (must use python -m; calling pyinstaller.exe directly fails silently)
.venv\Scripts\python.exe -m PyInstaller webai2api.spec --noconfirm --distpath dist --workpath build
# 4. Deploy to the dev instance 8800: stop the process before overwriting
# 5. Smoke test: all four platforms, minimal chat + full tool chain
.venv\Scripts\python.exe scripts\smoke_test.py
# 6. push main → tag vX.Y.Z → CI builds and creates the GitHub Release
```

## Directory layout

```
webai2api/
├── webai2api/
│   ├── app.py            # FastAPI application assembly
│   ├── gateway.py        # OpenAI-compatible gateway (auth + routing + usage)
│   ├── admin_api.py      # Panel backend API
│   ├── anthropic_adapter.py  # Anthropic Messages adapter (/v1/messages)
│   ├── login_webview.py  # Embedded WebView2 credential login
│   ├── store.py          # SQLite storage (accounts/keys/usage)
│   ├── web/index.html    # Admin panel (single file)
│   └── adapters/
│       ├── base.py          # Account client / pool / two-tier selection
│       ├── tool_bridge.py   # Tool-calling bridge (inject/parse/seam/degraded forms)
│       ├── reasoning_bridge.py  # Reasoning fallback bridge
│       ├── deepseek/     # DeepSeek core (based on deepseek2api, MIT)
│       ├── chatglm/      # Zhipu (delta+finish calibration, native tool_calls)
│       ├── kimi/         # Kimi (Connect protocol, DSML XML+CDATA)
│       └── qwen/         # Qwen (per-account hidden-browser relay)
├── data/                 # database / browser profiles (gitignored)
└── webai2api.spec        # PyInstaller spec
```

## Version history

See [GitHub Releases](https://github.com/duruoxian/webai2api/releases).
v0.1.50: four-platform parity (tool-call validation, empty-response retry,
reasoning dual-channel, required enforcement, continuation seam trimming),
multi-account auth failover + two-tier selection, unified usage accounting,
panel test grid, and removal of the legacy paid/lite editions (single free
edition since v0.1.36).

## Acknowledgements & license

The DeepSeek adapter core is based on
[deepseek2api](https://github.com/safe049/deepseek2api) /
[deepseek2api-fixed](https://github.com/duruoxian/deepseek2api-fixed)
(MIT License). The gateway, panel and multi-account layers are MIT as well —
see `LICENSE`.

Free code signing provided by [SignPath Foundation](https://signpath.org):
*Free code signing provided by SignPath.io, certificate by SignPath
Foundation.* Signing policy: [CODE_SIGNING.md](CODE_SIGNING.md).
