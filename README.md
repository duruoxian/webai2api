# webai2api

把各家 AI 的**网页版**反向代理成**统一的 OpenAI 兼容 API**，自带中文管理面板。

- 一个软件里管理**多个账号**（每个账号独立凭证、互不干扰，也不影响你自己浏览器里的登录）
- 生成**多个 API Key**，发给朋友，填进 opencode / Cherry Studio 等任何 OpenAI 兼容客户端即可
- 按 **模型 / 账号 / API Key** 统计 token 使用量
- 目前已接入：**DeepSeek 网页版**（chat.deepseek.com）

> 仅供个人学习与本地自用，请勿公开分发或对外提供服务。

## 特点

| 功能 | 说明 |
|---|---|
| 多账号 | 每个账号一套独立登录凭证，轮询调度，互不冲突 |
| 多 Key | 给不同的人/用途发不同的 Key，可随时启用/停用/删除 |
| 用量统计 | 每个模型、每个账号、每个 Key 的输入/输出 token 明细 |
| 中文面板 | 浏览器打开 `/admin` 可视化操作 |
| OpenAI 兼容 | `POST /v1/chat/completions`、`GET /v1/models`，支持流式 |

## 快速开始

### 1. 环境

- Windows / Linux / macOS
- Python **3.12+**

```bash
git clone <本仓库>
cd webai2api
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e .
```

### 2. 准备 PoW 文件（DeepSeek 必需）

DeepSeek 网页端有 PoW 反爬，需要一个 WASM 文件。

1. 打开 https://chat.deepseek.com ，按 F12 → Network，刷新页面
2. 找到 `sha3_wasm_bg.xxxxxxxx.wasm`，下载
3. 重命名为 **`sha3_wasm_bg.wasm`**，放到项目根目录

### 3. 启动

```bash
python -m webai2api            # 默认 http://127.0.0.1:8800
python -m webai2api --port 8800
```

### 4. 打开面板并添加账号

浏览器打开 **http://127.0.0.1:8800/admin**，默认密码 `admin`。

**方式一（推荐）：内置浏览器登录**

1. 在「账号管理」点「添加账号」，填个名称保存
2. 点该账号的「**浏览器登录**」按钮——软件会弹出一个**独立窗口的浏览器**
3. 在这个窗口里正常登录 DeepSeek（手机号/微信/Apple 均可）
4. 登录成功后凭证会**自动抓取并保存**，无需手动复制

> 每个账号使用**独立的浏览器配置目录**（`data/profiles/<账号id>`），
> 所以：不同账号的登录互不干扰，也**完全不会**影响你自己平时浏览器的登录状态。

**方式二：手动粘贴凭证**

在账号弹窗里手动填入 4 项凭证。提取方法：在 DeepSeek 页面按 F12 → Console，粘贴执行：

```javascript
copy(JSON.stringify({
  t: localStorage.getItem('userToken'),
  d: localStorage.getItem('deepseek-device-id:chat'),
  s: localStorage.getItem('settingsJwt'),
  c: document.cookie
}));
```

（Chrome/Edge 会弹「允许粘贴」，输入"允许粘贴"回车放行。）对应填入：

| 面板字段 | 取值 |
|---|---|
| auth_token | `t` 的 `value` 字段（去掉 `Bearer ` 前缀） |
| device_id | `d` |
| settings_token | `s` 的 `value.jwt` 字段 |
| cookie | `c` |

### 5. 生成 API Key 给客户端用

在「API Key」页生成一个 Key。然后任意 OpenAI 兼容客户端填：

- **Base URL**：`http://127.0.0.1:8800/v1`
- **API Key**：你生成的 `sk-...`
- **模型**：`deepseek-chat` 或 `deepseek-reasoner`

## opencode 接入示例

在 `~/.config/opencode/opencode.jsonc` 的 `provider` 里加：

```jsonc
"webai2api": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "webai2api",
  "options": {
    "baseURL": "http://127.0.0.1:8800/v1",
    "apiKey": "sk-你的key",
    "maxRetries": 0
  },
  "models": {
    "deepseek-chat": { "name": "DeepSeek Chat (网页版)", "attachment": true, "reasoning": true,
      "modalities": { "input": ["image", "text"], "output": ["text"] },
      "limit": { "context": 1000000, "output": 384000 } },
    "deepseek-reasoner": { "name": "DeepSeek Reasoner (网页版)", "attachment": true, "reasoning": true,
      "modalities": { "input": ["image", "text"], "output": ["text"] },
      "limit": { "context": 1000000, "output": 384000 } }
  }
}
```

## 打包成 exe

把整个服务打成一个 Windows 可执行程序（约 56 MB，无需再装 Python）：

```bash
# 双击运行，或命令行：
build-exe.cmd
```

产物：`dist/webai2api/webai2api.exe`。双击即用：

- 自动启动服务（`127.0.0.1:8800`）并**自动打开管理面板**
- 无黑框窗口，后台常驻（日志写在 exe 同级的 `webai2api.log`）
- 数据（数据库、浏览器 profile）存在 exe 同级的 `data/` 目录
- 首次使用若要"浏览器登录"功能，需要本机已装 Edge 或 Chrome（Windows 一般自带 Edge）

> **为什么这么轻？** 反代本身是纯 HTTP 请求，运行时**不需要浏览器进程**。
> 只有"登录拿凭证"那一下需要浏览器，因此这里直接用**系统自带的 Edge**（通过 CDP 协议），
> 而不是打包一个上百 MB 的 Chromium。

## 目录结构

```
webai2api/
├── webai2api/
│   ├── app.py            # FastAPI 应用组装
│   ├── __main__.py       # 启动入口
│   ├── gateway.py        # OpenAI 兼容网关（鉴权 + 路由 + 用量）
│   ├── admin_api.py      # 面板后端 API
│   ├── login.py          # 内置浏览器登录（Playwright，独立 profile）
│   ├── store.py          # SQLite 存储（账号/Key/用量）
│   ├── web/index.html    # 中文面板（单文件）
│   └── adapters/
│       └── deepseek/     # DeepSeek 适配内核（基于 deepseek2api，MIT）
├── data/                 # 数据库（运行时生成，已 gitignore）
└── sha3_wasm_bg.wasm     # PoW 文件（自行放置，已 gitignore）
```

## 路线图

- [x] DeepSeek 网页版适配
- [x] 多账号 / 多 API Key / 用量统计 / 中文面板
- [x] 内置浏览器登录（调用系统 Edge，独立 profile，免手动抓凭证）
- [x] 打包成 Windows exe（PyInstaller）
- [ ] 接入更多平台：GLM / Kimi / Qwen …
- [ ] 打包成 Tauri 桌面应用

## 更新记录

**v0.1.1（稳定性修复）**
- 修复：网关在同步调用 DeepSeek 客户端时**阻塞 asyncio 事件循环**，导致并发请求排队卡死
  （现改为线程池 + 生产/消费队列，并发实测 3 请求并行完成）
- 修复：`windowed` exe 因 `sys.stdout/stderr` 为 `None` 导致 uvicorn 启动卡死
- 改进：账号池增加**冷却机制**——某账号失败自动跳转其他账号重试（最多换一个）
- 改进：账号池减少每请求的数据库查询（改为 5 秒节流同步）
- 安全：面板 token 增加 12 小时有效期 + 登录失败限速 + 恒定时比较
- 健壮：账号/Key/设置接口增加请求体校验；浏览器登录结束后清理整个浏览器进程树

## 致谢与开源许可

DeepSeek 适配内核基于开源项目
[deepseek2api](https://github.com/safe049/deepseek2api) /
[deepseek2api-fixed](https://github.com/duruoxian/deepseek2api-fixed)（**MIT License**），
在其基础之上做了多账号、用量统计与网关层面的集成改造。

本项目的网关、面板、多账号等外层代码同为 MIT。详见 `LICENSE`。
