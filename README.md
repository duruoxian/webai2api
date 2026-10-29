# webai2api

> English documentation: [README_EN.md](README_EN.md)

把 DeepSeek、智谱清言、Kimi、通义千问四家 AI 的**网页版**反向代理成统一的
OpenAI 兼容 API，自带中文管理面板。Windows 下解压即用，全程本机运行。

## 功能

- **多平台**：DeepSeek 网页版、智谱清言（chatglm.cn）、Kimi（kimi.com）、通义千问（chat.qwen.ai），共十个模型
- **多账号**：每个账号一套独立登录凭证，轮询调度，互不冲突，也不影响你自己浏览器的登录
- **多 API Key**：给不同客户端/用途发不同的 Key，可随时启用、停用、删除
- **工具调用**：四平台全部支持——opencode、Claude Code、ZCode 等编码代理接进来能读写文件、跑命令
- **图片输入**：DeepSeek 与 GLM 支持在消息里附带图片（自动上传到对应平台）
- **用量统计**：按模型、账号、Key 统计输入/输出 token
- **中文面板**：浏览器打开 `/admin` 可视化管理账号、Key、用量
- **Anthropic 兼容**：`POST /v1/messages`（含工具调用与流式），Claude Code / ZCode 原生接入
- **完全开源免费**：MIT 协议，自由使用、修改；不售卖、不出租、不需要授权码

> 使用声明：本项目倡导仅供个人学习与本地自用，请勿公开分发或对外提供服务，
> 超出自用范围的一切后果由使用者自负。完整说明见 [DISCLAIMER.md](DISCLAIMER.md)。

## 快速开始（Windows exe）

1. 到 [Releases](https://github.com/duruoxian/webai2api/releases) 下载
   `webai2api-win64.zip`，解压后双击 `webai2api.exe`（无需安装 Python）。
2. 首次运行如弹出 SmartScreen 提示「Windows 已保护你的电脑」：这是系统对
   未签名程序的默认提醒，不是病毒。点「更多信息」→「仍要运行」即可。
   解压后如提示「已从 Internet 下载」，可右键 exe → 属性 → 勾选「解除锁定」。
3. 服务启动后自动打开管理面板（`http://127.0.0.1:8800/admin`，默认密码 `admin`），
   关闭面板窗口即退出服务。

不想用 exe 也可以源码运行（Python 3.12+，Windows/Linux/macOS）：

```bash
git clone <本仓库>
cd webai2api
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e .
python -m webai2api            # 默认 http://127.0.0.1:8800
```

## 添加账号

在面板「账号管理」点「添加账号」，两种方式：

**方式一（推荐）：内置浏览器登录**

1. 新建账号时只填「名称」保存，凭证四项留空；
2. 点该账号的「浏览器登录」——会弹出一个软件内嵌的浏览器窗口（WebView2，
   不是你日常的浏览器）；
3. 在窗口里正常登录对应平台（验证码/密码/扫码均可）；
4. 登录成功后凭证自动抓取并保存，无需手动复制。

每个账号使用独立的浏览器数据目录（`data/browser_profiles/<账号id>`），
不同账号互不干扰，也不影响你平时浏览器的登录状态。

**方式二：手动粘贴凭证**

在对应平台页面按 F12 打开 Console，粘贴执行（以 DeepSeek 为例）：

```javascript
copy(JSON.stringify({
  t: localStorage.getItem('userToken'),
  d: localStorage.getItem('deepseek-device-id:chat'),
  s: localStorage.getItem('settingsJwt'),
  c: document.cookie
}));
```

（Chrome/Edge 会弹「允许粘贴」，输入"允许粘贴"回车放行。）对应填入面板：

| 面板字段 | 取值 |
|---|---|
| auth_token | `t` 的 `value` 字段（去掉 `Bearer ` 前缀） |
| device_id | `d` |
| settings_token | `s` 的 `value.jwt` 字段 |
| cookie | `c` |

## 接入客户端

可用模型：`deepseek-chat` / `deepseek-reasoner` / `GLM-5.3` / `GLM-5.3-Flash` /
`kimi-k3` / `kimi-k2.8` / `kimi-k2.6` / `qwen3.7-plus` / `qwen3.8-max` /
`qwen3.8-omni-flash`。

> `qwen3.8-max` 目前为实验性：网页反代下稳定性差（偶发语序混乱，属模型自身
> 问题，适配层无法修复）。对输出质量有要求时建议用 `qwen3.7-plus`。

**OpenAI 兼容客户端**（opencode、Cherry Studio 等）填：

- Base URL：`http://127.0.0.1:8800/v1`
- API Key：面板「API Key」页生成的 `sk-...`

`127.0.0.1` 指「运行本软件的那台电脑」：你在自己电脑上跑就填这个；
别人在他自己的电脑上跑本软件，也填这个，各用各的。

**Anthropic 原生客户端**（Claude Code / ZCode）直接接 `/v1/messages`：

- `ANTHROPIC_BASE_URL`：`http://127.0.0.1:8800`
- `ANTHROPIC_AUTH_TOKEN`（或 `ANTHROPIC_API_KEY`）：你生成的 `sk-...`
- 例：`claude -p "你好" --model webai2api/GLM-5.3-Flash`

**opencode 接入示例**

方式 A（推荐）：面板「API Key」页点「接入」，一键写入 opencode 配置
（含备份与一键移除）。也可以在 opencode 的 设置 → 提供商 → 自定义提供商
里手填：提供商 ID `webai2api`、基础 URL `http://127.0.0.1:8800/v1`、
API 密钥填面板生成的 Key、模型 id 一字不差（如 `deepseek-chat`）。

方式 B：直接改 `~/.config/opencode/opencode.jsonc`（opencode ≥2.x 的 V2 格式）：

```jsonc
"webai2api": {
  "package": "@opencode/ai/providers/openai-compatible",
  "name": "webai2api",
  "settings": {
    "baseURL": "http://127.0.0.1:8800/v1",
    "apiKey": "sk-你的key",
    "maxRetries": 0
  },
  "compatibility": { "streamUsage": true },
  "models": {
    "deepseek-chat": { "name": "DeepSeek Chat (网页版)",
      "capabilities": { "input": ["image", "text"], "output": ["text"],
        "reasoning": { "supported": false } },
      "limit": { "context": 1000000, "output": 384000 } },
    "deepseek-reasoner": { "name": "DeepSeek Reasoner (网页版)",
      "capabilities": { "input": ["image", "text"], "output": ["text"],
        "reasoning": { "supported": true } },
      "limit": { "context": 1000000, "output": 384000 } }
  }
}
```

## 打包成 exe（自建）

```bash
build-exe.cmd
```

产物：`dist/webai2api/webai2api.exe`（约 60 MB）。双击即用：自动启动服务并
弹出原生桌面窗口（pywebview + 系统 WebView2）打开面板，关窗即退出；数据存在
exe 同级的 `data/` 目录。

反代本身是纯 HTTP 请求，运行时不需要浏览器进程；只有「登录拿凭证」那一下
需要浏览器，用的是系统已装的 WebView2 内嵌窗口，不打包 Chromium，也不调用
你日常浏览器的 profile。

## 维护者指南

```powershell
# 1. 版本号四处拉齐：webai2api/__init__.py / pyproject.toml / web/index.html 的两处 ?v=
# 2. 全量测试（必须全绿）
.venv\Scripts\python.exe -m pytest tests -q
# 3. 本地构建（必须用 python -m，直接调 pyinstaller.exe 会静默失败）
.venv\Scripts\python.exe -m PyInstaller webai2api.spec --noconfirm --distpath dist --workpath build
# 4. 部署开发端 8800：先停进程再覆盖 exe 与 _internal（否则文件占用静默跳过）
# 5. 真机冒烟：四平台最小对话 + 完整工具链（真实写文件、回填结果、继续对话）
.venv\Scripts\python.exe scripts\smoke_test.py
# 6. push main → 打 tag vX.Y.Z → CI 自动构建并创建 GitHub Release → gh release edit 补标题/说明
```

冒烟会消耗少量真实额度；任一平台失败退出码为 1。

## 目录结构

```
webai2api/
├── webai2api/
│   ├── app.py            # FastAPI 应用组装
│   ├── __main__.py       # 启动入口
│   ├── gateway.py        # OpenAI 兼容网关（鉴权 + 路由 + 用量）
│   ├── admin_api.py      # 面板后端 API
│   ├── anthropic_adapter.py  # Anthropic Messages 协议适配（/v1/messages）
│   ├── login.py          # 内置浏览器登录管理（启动子进程 + 读状态）
│   ├── login_webview.py  # 内嵌 WebView2 登录并抓取凭证
│   ├── store.py          # SQLite 存储（账号/Key/用量）
│   ├── web/index.html    # 中文面板（单文件）
│   └── adapters/
│       ├── base.py          # 账号客户端 / 池 / 两级选号
│       ├── tool_bridge.py   # 工具调用桥接（注入/增量/解析/接缝剪除/降级形态清洗）
│       ├── reasoning_bridge.py  # 思考块兜底桥（空正文转正、<think> 兼容模式）
│       ├── deepseek/     # DeepSeek 适配内核（基于 deepseek2api，MIT；DSML 工具协议）
│       ├── chatglm/      # 智谱清言（delta+finish 校准累积器、服务端原生 tool_calls）
│       ├── kimi/         # Kimi（Connect 协议、DSML XML+CDATA）
│       └── qwen/         # 通义千问（每账号隐藏浏览器中继、多路候选过滤）
├── data/                 # 数据库 / 浏览器 profile（运行时生成，已 gitignore）
├── webai2api.spec        # PyInstaller 打包配置
```

## 版本历史

完整发布说明见 [GitHub Releases](https://github.com/duruoxian/webai2api/releases)。

**v0.1.50（2026-10-29）**：四平台适配拉平（工具调用必填校验/空响应重试/
思考双通道/required 强校验/续写接缝剪除）、多账号 auth 自动换号 + 两级选号、
用量口径四家统一、面板四层测试网、lite 客户版与授权体系物理删除（单版本化）。

> 历史存档：v0.1.36 之前本项目曾有专业版/客户版（Lite）双版本与授权码体系，
> v0.1.36 起合并为单一开源版本，相关实现已于 2026-10-29 从代码中物理删除。

## 开发记录

踩坑与实现细节的完整记录（博客）：

- [把网页版做成 API](https://duruoxian-blog.pages.dev/posts/webai2api)
- [让网页模型真的会调工具](https://duruoxian-blog.pages.dev/posts/webai2api-v3)

## 致谢与许可

DeepSeek 适配内核基于开源项目
[deepseek2api](https://github.com/safe049/deepseek2api) /
[deepseek2api-fixed](https://github.com/duruoxian/deepseek2api-fixed)
（MIT License），在其基础之上做了多账号、用量统计与网关层面的集成改造。

本项目的网关、面板、多账号等外层代码同为 MIT，详见 `LICENSE`。

发布产物的代码签名由 [SignPath Foundation](https://signpath.org) 免费提供：
*Free code signing provided by SignPath.io, certificate by SignPath Foundation.*
代码签名政策见 [CODE_SIGNING.md](CODE_SIGNING.md)。
