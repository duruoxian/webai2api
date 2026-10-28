# webai2api 🌳

> **把各家 AI 的网页版 🌲，变成一片会说话的森林 🦜🐒**

一个 Windows 本地小工具，把四家 AI 的**网页版**反向代理成**统一的
OpenAI 兼容 API**，自带中文管理面板 😃

| | |
|---|---|
| 🌳 **四平台一家亲** | DeepSeek 🐋 / 智谱清言 ✨ / Kimi 🌙 / 通义千问 🦚 —— 十个模型随便挑 |
| 👀 **看得见的管理** | 中文面板：账号、Key、用量，一切尽在眼底 |
| 🐒 **真的会干活** | 四平台全部支持工具调用——opencode / Claude Code / ZCode 接进来能读写文件、跑命令、写文档 |
| 🛡️ **自己的数据自己守** | 全程本机运行，账号凭证不上传任何第三方 ♥️ |
| 🌕 **开箱即用** | Windows 免装 Python，解压双击就跑 |
| 🆓 **完全开源免费** | MIT 协议，自由使用、修改、分发；不售卖、不出租、不需要授权码 |

- 一个软件里管理**多个账号**（每个账号独立凭证、互不干扰，也不影响你自己浏览器里的登录）
- 生成**多个 API Key**，发给朋友，填进 opencode / Cherry Studio 等任何 OpenAI 兼容客户端即可
- 按 **模型 / 账号 / API Key** 统计 token 使用量
- 想折腾？也支持源码运行（Python 3.12+）

> 🥺 仅供个人学习与本地自用，请勿公开分发或对外提供服务——
> 风控 ⛈️ 面前，低调是最长的保险。

> 🥹 这个项目从「一个周末的反代脚本」长成了现在这棵树，
> 中间被真实使用打碎又重装了好几轮。完整踩坑记录（博客系列）：
> [一、把网页版做成 API](https://duruoxian-blog.pages.dev/posts/webai2api) ｜
> [二、从玩具到能卖的客户端](https://duruoxian-blog.pages.dev/posts/webai2api-v2) ｜
> [三、让网页模型真的会调工具](https://duruoxian-blog.pages.dev/posts/webai2api-v3)

## 下载（Windows）

不想折腾环境？直接到 **[Releases](https://github.com/duruoxian/webai2api/releases)** 下载 `webai2api-win64.zip`，解压后双击 `webai2api.exe` 即可（无需安装 Python）。详见下方「打包成 exe」。

## 特点

| 功能 | 说明 |
|---|---|
| 多账号 | 每个账号一套独立登录凭证，轮询调度，互不冲突 |
| 多平台 | 支持 DeepSeek 网页版、智谱清言（chatglm.cn）、Kimi（kimi.com）、通义千问（chat.qwen.ai） |
| 多 Key | 给不同的人/用途发不同的 Key，可随时启用/停用/删除 |
| 用量统计 | 每个模型、每个账号、每个 Key 的输入/输出 token 明细 |
| 中文面板 | 浏览器打开 `/admin` 可视化操作（单文件面板 `webai2api/web/index.html`） |
| OpenAI 兼容 | `POST /v1/chat/completions`、`GET /v1/models`，支持流式 |
| Anthropic 兼容 🆕 | `POST /v1/messages`（含工具调用与流式事件）——Claude Code / ZCode 等原生客户端直接接入 |

> **自 v0.1.36 起，本项目合并为单一版本，完全开源免费，不再提供授权码、不再售卖或出租。**
> 旧的专业版 / 客户版（Lite）双版本方案已停止维护，下方「双版本构建」「授权体系」等章节仅作历史存档。

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

### 2. 启动

```bash
python -m webai2api            # 默认 http://127.0.0.1:8800
python -m webai2api --port 8800
```

### 3. 打开面板并添加账号

浏览器打开 **http://127.0.0.1:8800/admin**，默认密码 `admin`。

**方式一（推荐）：内置浏览器登录**

1. 在「账号管理」点「添加账号」，**只填「名称」**保存（下面 4 项凭证**留空即可**）
2. 点该账号的「**浏览器登录**」按钮——软件会弹出一个**内嵌的浏览器窗口**（WebView2，在软件内部，**不调用你日常的浏览器**）
3. 在这个窗口里正常登录 DeepSeek（手机号验证码 / 密码 / 微信 / Apple 均可）
4. 登录成功后凭证会**自动抓取并保存**，账号行显示「登录成功」，无需手动复制

> 每个账号使用**独立的浏览器数据目录**（`data/browser_profiles/<账号id>`），
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

**Anthropic 原生客户端（Claude Code / ZCode 等）** 也能直接接——
webai2api 提供 `/v1/messages` 端点（含工具调用与流式）：

- **ANTHROPIC_BASE_URL**：`http://127.0.0.1:8800`
- **ANTHROPIC_AUTH_TOKEN**（或 ANTHROPIC_API_KEY）：你生成的 `sk-...`
- **模型**：与上表相同；以 Claude Code 为例：
  `claude -p "你好" --model webai2api/GLM-5.3-Flash`
  （或在 `~/.claude/settings.json` 里配置 env 后正常启动）
- **模型**：`deepseek-chat` / `deepseek-reasoner` / `GLM-5.3` / `GLM-5.3-Flash` / `kimi-k3` / `kimi-k2.8` / `kimi-k2.6` / `qwen3.7-plus` / `qwen3.8-max` / `qwen3.8-omni-flash`

> ⚠️ `qwen3.8-max` 目前为**实验性**：网页反代下稳定性差（偶发语序混乱、逻辑不连贯，属模型自身问题，适配层无法修复）。对输出质量有要求时建议用 `qwen3.7-plus`。

## opencode 接入示例

### 方式 A：用 opencode 图形界面添加（推荐，无需改文件）

opencode 主页 → **设置 → 提供商 → 自定义提供商**，按下表填：

| 字段 | 填什么 |
|---|---|
| 提供商 ID | `webai2api` |
| 显示名称 | `DeepSeek 网页版 (webai2api)`（随意） |
| 基础 URL | `http://127.0.0.1:8800/v1` |
| API 密钥 | 面板「API Key」页生成的那个 `sk-…` |
| 模型① model-id | `deepseek-chat` |
| 模型① 显示名称 | `DeepSeek V4.1-Flash` |
| 模型② model-id | `deepseek-reasoner` |
| 模型② 显示名称 | `DeepSeek V4.1-Flash 思考` |

保存后，在模型列表选 `webai2api/deepseek-chat`（或 `…/deepseek-reasoner`）即可。
模型 id 必须**一字不差**：`deepseek-chat` / `deepseek-reasoner`。

> 说明：**`base URL` 永远是 `http://127.0.0.1:8800/v1`**。
> 因为 `127.0.0.1` 指"运行本软件的那台电脑"——你在自己电脑上跑，就填这个；
> 别人在**他自己电脑**上跑本软件，**也填这个**（各用各的，互不依赖）。

其它平台把 model-id 换成对应值即可：GLM-5.3 / GLM-5.3-Flash（智谱）、
kimi-k3 / kimi-k2.8 / kimi-k2.6（Kimi）、qwen3.7-plus / qwen3.8-max /
qwen3.8-omni-flash（千问）。懒得手填的话，面板「API Key」页点「接入」
可一键写入 opencode 配置（含备份与一键移除）。

### 方式 B：直接改配置文件

> opencode ≥2.x 用 V2 格式：顶层键 `providers`，供应商用 `package` + `settings`，
> 模型能力用 `capabilities`。V1（`provider` + `npm` + `options`）已被 V2 取代。

在 `~/.config/opencode/opencode.jsonc` 的 `providers` 里加：

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

## 打包成 exe

把整个服务打成一个 Windows 可执行程序（约 60 MB，无需再装 Python）：

```bash
# 双击运行，或命令行：
build-exe.cmd
```

产物：`dist/webai2api/webai2api.exe`。双击即用：

- 自动启动服务（`127.0.0.1:8800`）并弹出**原生桌面窗口**（pywebview + 系统 WebView2）打开管理面板
- 关闭窗口即退出；无黑框窗口，后台常驻（日志写在 exe 同级的 `webai2api.log`）
- 若原生窗口不可用，自动回退系统默认浏览器
- 数据（数据库、浏览器 profile）存在 exe 同级的 `data/` 目录
- 「浏览器登录」用**内嵌 WebView2**（Win10/11 一般自带，随 Edge 一起安装），无需另装浏览器，也不影响你日常浏览器

> **为什么这么轻？** 反代本身是纯 HTTP 请求，运行时**不需要浏览器进程**。
> 只有"登录拿凭证"那一下需要浏览器，这里用**系统已装的 WebView2**（pywebview 内嵌窗口），
> 不打包上百 MB 的 Chromium，**也不调用你日常浏览器的 profile**（零冲突）。

## 发布（维护者）

```powershell
# 1. 版本号四处拉齐：webai2api/__init__.py / pyproject.toml / web/index.html 的两处 ?v=
# 2. 全量测试（必须全绿，当前基线 514+）
.venv\Scripts\python.exe -m pytest tests -q
# 3. 本地构建（必须用 python -m，直接调 pyinstaller.exe 会静默失败）
.venv\Scripts\python.exe -m PyInstaller webai2api.spec --noconfirm --distpath dist --workpath build
# 4. 部署开发端 8800：先停进程再覆盖 exe 与 _internal（否则文件占用静默跳过）
# 5. 真机冒烟：四平台最小对话 + 完整工具链（真实写文件、回填结果、继续对话）
.venv\Scripts\python.exe scripts\smoke_test.py
# 6. push main → 打 tag vX.Y.Z → CI 自动构建并创建 GitHub Release → gh release edit 补标题/说明
```

冒烟会消耗少量真实额度；任一平台失败退出码为 1。Key 自动从本机开发端数据库读取，也可用 `--key` 指定。

## 版本合并与授权体系（历史存档 · 已物理删除）

> ⚠️ **历史记录，当前代码里已不存在对应实现。**
> - v0.1.36 起不再使用授权码（完全开源免费，无激活/到期/机器绑定）；
> - v0.1.43 起 lite 面板代码层退役（单面板）；
> - 2026-10-29 起 **lite 版物理删除**：`edition.py`、`webai2api-lite.spec`、
>   `lite_panel.html`、`installer/webai2api-lite.iss`、lite 图标与 `usage-lite.txt`
>   全部移除，CI 不再有 `-lite` tag 分支，永远只构建开发端（8800）。
>
> 旧版差异机制（仅存档）：专业版与客户版曾出自同一份代码，由构建时的 `edition.txt`
> 决定运行形态——端口（8802/8800）、面板文件、托盘图标、数据目录
> （lite 装在 Program Files 时回退 `%LOCALAPPDATA%\webai2api-lite\data`）。
> 授权体系曾为 Ed25519 签名码（`WB2.<payload>.<sig>`），激活记录三副本防删、
> 防时钟回拨，签发走开发端「授权签发」卡，私钥四不原则（不入库/不进包/不发客户/不进日志）。

## 目录结构

```
webai2api/
├── webai2api/
│   ├── app.py            # FastAPI 应用组装
│   ├── __main__.py       # 启动入口
│   ├── gateway.py        # OpenAI 兼容网关（鉴权 + 路由 + 用量）
│   ├── admin_api.py      # 面板后端 API
│   ├── login.py          # 内置浏览器登录管理（启动子进程 + 读状态）
│   ├── login_webview.py  # 内置浏览器（内嵌 WebView2）登录并抓取凭证
│   ├── store.py          # SQLite 存储（账号/Key/用量）
│   ├── anthropic_adapter.py  # Anthropic Messages 协议适配（/v1/messages，Claude Code / ZCode 原生接入）
│   ├── web/index.html    # 中文面板（单文件；lite 面板已于 2026-10-29 物理删除，永远只有开发端）
│   └── adapters/
│       ├── base.py          # 账号客户端 / 池 / 两级选号
│       ├── tool_bridge.py   # 网页版工具调用桥接（注入/增量/解析/接缝剪除，kimi·qwen·glm 共用）
│       ├── reasoning_bridge.py  # 思考块兜底桥（空正文转正、<think> 兼容模式）
│       ├── deepseek/     # DeepSeek 适配内核（基于 deepseek2api，MIT；自带 DSML 工具协议）
│       ├── chatglm/      # 智谱清言（delta+finish 校准累积器、服务端原生 tool_calls）
│       ├── kimi/         # Kimi（Connect 协议、DSML XML+CDATA）
│       └── qwen/         # 通义千问（每账号隐藏浏览器中继、多路候选过滤）
├── data/                 # 数据库 / 浏览器 profile（运行时生成，已 gitignore）
├── tools/                # 历史授权体系的遗留公钥（代码已删，仅存档）
├── webai2api.spec        # PyInstaller 打包配置（唯一版本）
```

## 路线图

- [x] DeepSeek 网页版适配
- [x] 多账号 / 多 API Key / 用量统计 / 中文面板
- [x] 内置浏览器登录（**内嵌 WebView2**，不调用系统浏览器，独立 profile，免手动抓凭证）
- [x] 打包成 Windows exe（PyInstaller）
- [x] 原生桌面窗口（pywebview + 系统 WebView2，关闭窗口即退出）
- [x] 接入智谱清言（chatglm.cn）：GLM-5.3 / GLM-5.3-Flash
- [x] 接入 Kimi（网页版 Connect 协议）：`kimi-k3` / `kimi-k2.8` / `kimi-k2.6`
- [x] 接入通义千问（每账号隐藏浏览器中继，绕阿里风控）：`qwen3.7-plus` / `qwen3.8-max` / `qwen3.8-omni-flash`
- [x] Anthropic Messages 端点（`/v1/messages`，Claude Code / ZCode 原生接入，含流式与工具调用）
- [x] 四平台工具调用拉平（必填校验/空响应重试/思考双通道/required 强校验/续写接缝剪除）
- [x] 可靠性：多账号 auth 自动换号、两级选号、会话粘性 + 压缩检测、用量口径四家统一
- [x] 面板四层测试网（契约 / 主题矩阵 / 健康防线 / 真浏览器走查）+ O(n²) 渲染修复
- [x] 单版本化：lite 客户版与授权体系物理删除，永远只有开发端

## 测试

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

没有 pytest 时先 `pip install -r requirements-dev.txt`（或 `pip install pytest pytest-asyncio`）。

## 版本历史

完整发布说明见 **[GitHub Releases](https://github.com/duruoxian/webai2api/releases)**。

**v0.1.50（当前，2026-10-29）—— 四平台适配拉平 + 可靠性修复 + 面板测试网**

- 工具调用必填校验、空响应弃会话重试、思考双通道提取、required 强校验、续写接缝剪除——deepseek / GLM / kimi / qwen 四家同规格
- 多账号 auth 自动换号 + 两级选号：单个账号凭证过期不再拖垮请求
- 用量口径四家统一（deepseek 修会话累计虚高、kimi/qwen 补思考计入）
- 面板修复 4 处 O(n²) 渲染；新增契约 / 主题矩阵 / 健康防线 / 真浏览器四层测试（全量 514 项）
- lite 客户版与授权体系物理删除，永远只有开发端单版本

更早的历史版本（v0.1.36 单版本化、v0.1.32 流链重构、v0.1.28 适配层大修等）
见对应 tag 的 Release 说明与本仓库提交记录。

## 致谢与开源许可

DeepSeek 适配内核基于开源项目
[deepseek2api](https://github.com/safe049/deepseek2api) /
[deepseek2api-fixed](https://github.com/duruoxian/deepseek2api-fixed)（**MIT License**），
在其基础之上做了多账号、用量统计与网关层面的集成改造。

本项目的网关、面板、多账号等外层代码同为 MIT。详见 `LICENSE`。
