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

## 双版本构建（历史存档 · v0.1.35 及以前）

> ⚠️ **本章节为历史存档。** 自 v0.1.36 起项目合并为单一版本，不再构建 Lite 客户版，
> 也不再使用授权码。以下内容仅用于理解旧版本的产物结构。

v0.1.26 起，专业版与客户版出自**同一份代码、同一个分支**，版本差异由构建时的 `edition.txt` 决定，
运行期 `webai2api.edition.is_lite()` 自动分派端口（8802/8800）、面板（`lite_panel.html`/`index.html`）、
授权卡/签发卡、托盘图标与数据目录（lite 装在 Program Files 时数据自动回退到
`%LOCALAPPDATA%\webai2api-lite\data`）。

| 版本 | 默认端口 | 面板 | 安装/部署位置 | 产物 |
|---|---|---|---|---|
| **Pro**（spec：`webai2api.spec`） | `127.0.0.1:8800` | `index.html`（全部功能 + 签发卡） | `%LOCALAPPDATA%\Programs\webai2api`（pyinstaller onedir） | `dist/webai2api/webai2api.exe` |
| **Lite**（spec：`webai2api-lite.spec`） | `127.0.0.1:8802` | `lite_panel.html`（授权卡激活制） | `{localappdata}\Programs\webai2api-lite`（Inno Setup 安装） | `dist-lite/webai2api/webai2api.exe` → `dist-installer/webai2api-lite-setup-X.Y.Z.exe` |

### 版本 bump 点

| 位置 | 内容 |
|---|---|
| `webai2api/__init__.py` | `__version__ = "x.y.z"` |
| `pyproject.toml` | `version = "x.y.z"` |
| `webai2api/web/index.html` | 静态资源版本串 `?v=x.y.z` |
| `installer/webai2api-lite.iss` | `#define MyAppVersion "x.y.z"`（决定安装包文件名） |

### 构建与发布

```powershell
# 专业版
.venv\Scripts\pyinstaller.exe webai2api.spec --noconfirm --distpath dist --workpath build
robocopy "dist\webai2api" "$env:LOCALAPPDATA\Programs\webai2api" /E /XD data /XF webai2api.log current_port.txt

# 客户版（同一棵源码树）
.venv\Scripts\pyinstaller.exe webai2api-lite.spec --noconfirm --distpath dist-lite --workpath build-lite
& "C:\Users\<你>\AppData\Local\Programs\Inno Setup 6\ISCC.exe" installer\webai2api-lite.iss
```

`/XD data` 是关键：否则新包里的空 `data` 会盖掉账号库与授权记录。
CI 按 tag 后缀自动选版本：推 `vX.Y.Z` → 构建专业版并发 Release；推 `vX.Y.Z-lite` → 客户版。

### 发版前冒烟测试（必跑）

```powershell
.venv\Scripts\python.exe scripts\smoke_test.py
```

对本地实例逐个验证四家平台：最小对话 + 完整工具链（真实执行文件写入、回填结果、继续对话）。
任一平台失败退出码为 1。Key 自动从本机开发端数据库读取，也可用 `--key` 指定。

### 授权体系简表（历史存档 · 已停用）

> ⚠️ **自 v0.1.36 起不再使用授权码。** 软件完全开源免费，无激活、无到期、无机器绑定。
> 以下为旧版本实现记录。

| 项 | 实现 |
|---|---|
| 码格式 | `WB2.<payload>.<sig>`，Ed25519 签名（纯 Python `webai2api/ed25519.py`，无 cryptography 依赖） |
| payload | 固定 **10 字节**：版本 1B / 码 ID 4B / 有效天数 uint16 / 签发日 uint16 / edition 1B |
| 码内机器号 | **不绑定**。首次激活时才在本地绑定机器码并记录激活时间；同码换机 → `other_device`（「此授权码已在其他设备激活」） |
| 计时 | 从**首次激活时刻**起算，精确到分钟；面板倒计时 `X天 X小时 X分`，到期即 `license.key` 失效、`/v1/*` 返回 403（面板本身仍可用） |
| 记录保护 | 激活记录带防篡改校验，手改记录文件会被判为「被修改」 |
| 面板接口 | `GET/POST /admin/api/license`（查状态）、`POST /admin/api/license/install`（激活/续期，旧别名 `/license/activate`） |
| 三副本 | 激活记录在数据目录、隐藏影子文件、注册表三处各存一份并交叉自愈；删 `data/` 不会重置授权 |
| 防时钟回拨 | `license.check_clock()` 记录上次运行时间，回拨累计 `license_clock_strikes`，`strikes >= 3` 判可疑并拒绝 |
| 机器码 | 硬件指纹（**刻意不含硬盘**，换盘不失效），形如 `ABCD-EFGH-IJKL-MNOP` |

### 签发流程（作者侧）

1. 私钥查找顺序：环境变量 `WEBAI2API_AUTHOR_KEY` → `%LOCALAPPDATA%\webai2api-author\private_key.txt` → 仓库 `tools\private_key.txt`。
   私钥**四不原则**：不入库、不进包、不发客户、不进日志。
2. 打开开发端（8800）→「设置 → 授权签发」卡：选天数 / 备注 → 生成 → 拿到授权码，同时写入签发记录。
3. 签发记录落盘在数据目录：`data/issued_codes.json`，导出 CSV 为 `data/issued_codes.csv`（含有效授权码，属**敏感文件**）。
4. 客户侧激活接口是 `POST /admin/api/license/install`（旧名 `/admin/api/license/activate` 保留为别名），
   面板「设置 → 授权」卡片就是调它。
5. 客户续期直接粘贴新码覆盖；换机报 `other_device`，让客户提供授权卡上的**授权编号**（码 ID）再补发。

### 安全红线

1. `tools\private_key.txt`、`%LOCALAPPDATA%\webai2api-author\` **绝不入库、绝不进包、绝不发客户**。
2. 公钥内嵌 `license.py`（验签用）；更换私钥必须同步换公钥并**重发客户版**。
3. 签发记录（`issued_codes.json` / `issued_codes.csv`）含有效授权码，属敏感文件，不要随包分发。
4. 客户版永不显示签发卡，`/license/issue` 在 lite 下一律拒绝；面板与日志都不要回显私钥。
5. 交付给客户的内容只有：安装包 + `使用说明.txt`。不要把 8800 开发端、签发器、`tools\` 一并带出去。

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
│   ├── web/index.html    # 中文面板（单文件，两分支共用，lite 版按 edition 隐藏签发卡）
│   └── adapters/
│       └── deepseek/     # DeepSeek 适配内核（基于 deepseek2api，MIT）
├── data/                 # 数据库 / 浏览器 profile / license.key / issued_codes.*（运行时生成，已 gitignore）
├── tools/                # 作者私钥 keygen CLI（禁入库、禁进包）
├── webai2api.spec        # PyInstaller 打包配置（Pro）
└── webai2api-lite.spec   # PyInstaller 打包配置（lite 分支）
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

## 测试

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

没有 pytest 时先 `pip install -r requirements-dev.txt`（或 `pip install pytest pytest-asyncio`）。

## 更新记录

**v0.1.10（授权精确到分钟 + 一键接入 opencode）**
- **授权到期精确到分钟**：倒计时按 `expires_ts` 在前端自算，显示「X天 X小时 X分」，激活 / 到期时间同为 `时:分` 精度，到期那一分钟即停，不再是「按天」算。
- **API Key 页「接入方式」合并**：原来散在设置页的客户端说明收进 API Key 页一张卡——顶部接口地址 / 可用模型 / Key 下拉与「接入」「移除」，下面依次是「OpenCode 配置片段」和「其他 OpenAI 兼容客户端」对照表（提供商 ID、显示名称、基础 URL、模型、请求头，逐项可复制）。
- **选 Key 一键接入 / 移除 opencode**：选中 Key 点「接入」即自动写入 `~/.config/opencode/opencode.jsonc`（写前备份为同名 `.bak_时间戳`），「移除」只删自己那段 `provider.webai2api`，不影响已有配置；目标路径可用环境变量 `WEBAI2API_OPENCODE_CONFIG` 覆盖。
- **Key 生成防连点**：生成按钮提交即禁用（「生成中…」→「已生成」），重复点击不会一次多建出几个 Key。

**v0.1.9（授权签发 + 界面修复）**
- **内置授权签发**：设置页多了「授权签发」卡片，只有本机存在作者私钥时才显示。可选 7 / 14 / 30 / 自定义天数，支持批量生成与备注。
- **签发记录表**：生成时间、有效天数、参考到期日、备注、授权码一目了然，支持搜索、单条复制、导出 CSV（写入数据目录并自动定位文件）。
- **签发器不进分发包**：私钥不参与打包，客户版永不显示签发卡片，接口也会拒绝。
- **日期选择修复**：自定义日期范围与 Key 到期日的日历图标改为 CSS 绘制（WebView2 下 SVG 渲染成空白）。
- **告警条改版**：去掉 emoji 三角，改为圆点标记，深色模式配色同步调整。
- **筛选下拉排序**：最近请求的 Key / 账号下拉里，已删除项统一排到最后。

**v0.1.8（稳定性修复 + 用量统计重构）**
- **会话粘性**：同一段对话固定在同一账号上（按「Key + 模型 + 对话开头」指纹，6 小时有效），多轮对话不再因轮询换号导致上游会话缓存全部失效、反复重传上下文。
- **账号冷却真正生效**：此前冷却代码没有调用点。现在按错误类型分类处理——凭证失效冷却 10 分钟、限流 60 秒起指数退避（最长 15 分钟）、5xx/网络 30 秒、内容审核不冷却直接返回。全部账号都在冷却时直接返回「请约 N 分钟后重试」，不再硬撞上游。账号页会显示「冷却 N 分」。
- **重试策略细分**：401 / 内容审核立即失败不等待；限流直接换号；只有网络和 5xx 才退避重试。
- **Token 估算兜底**：上游不返回 usage 时，输入按 messages 估算、输出按实际文本估算（原来 completion 恒为 0，导致额度统计严重偏低）。
- **额度并发预占**：并发请求不再同时挤过同一个额度检查；异常路径有 30 分钟自动清理，不会卡死额度。
- **客户端断开取消上游**：流式响应中断后立刻停止拉取上游并关闭连接，日志记为 `cancelled`，不再白烧额度。
- **无凭证账号不参与选号**：避免选了也必然失败、白白重试 5 次。
- **用量统计去冗余**：
  - 「每日 Token 消耗」改为按 Key 切换查看（全部 Key / 单个 Key），7/14/30/90 天与滑块照常可用；「全部 Key」只统计仍然存在的 Key。
  - 概览卡片从 7 个合并为 4 个（总请求含成功率、总 Token 含输入输出、近 24h、可用账号）。
  - Key 额度并入「按 Key 用量」表（请求数 / Token 合计 / 今日 / 本月 / 额度进度条）。
  - Key 详情弹窗去掉重复的每日小柱图，改为「在图表中查看每日消耗」一键跳转。
- **用量数据清理**：保留天数改为每 6 小时检查一次（原来只在启动时读一次，长期运行不清理）；设置页显示当前条数、数据范围、数据库体积，并支持「立即清理历史用量」。
- **接口地址跟随实际端口**：面板里写死的 `127.0.0.1:8800` 改为按当前访问地址生成，端口被占用自动漂移时也不会填错。
- 新增 `(api_key_id, ts)` 复合索引，Key 日/月额度查询不再全表扫描。

**v0.1.7（Apple 液态玻璃面板 + 智谱登录隔离修复）**
- 面板视觉整体重做：Apple「液态玻璃」风格，支持**亮色 / 暗色一键切换**（记住偏好）；顶栏、卡片、弹窗、下拉框、日期选择器统一为同一套设计语言。玻璃效果只用于顶栏、登录卡片等轻量区域，避免长列表滚动卡顿。
- 弹窗重构：标题与底部按钮固定，内容区独立滚动，修掉「添加账号」弹窗滚动条压住圆角、看起来像穿模的问题。
- **修复智谱（chatglm）添加账号会直接登进旧账号**：chatglm.cn 打开页面 1~2 秒内会先下发一个「游客」refresh_token（JWT 中 `is_guest: true`），旧逻辑会把它当成登录成功并保存；现在会识别并忽略游客凭证，抓到其它账号的 token 时还会自动清空 cookie / localStorage / sessionStorage 后重新加载。
- 登录结束后的进程清理改用 PowerShell CIM 精确匹配，兼容没有 `wmic` 的 Windows 11，不再残留 WebView2 进程占用账号目录。
- 设置页新增两块内容：
  - **关于**：作者、GitHub 仓库、版本号、使用声明与免责条款。
  - **在其他客户端中接入**：提供商 ID、显示名称、基础 URL、API 密钥、模型、请求头怎么填，逐项带「复制」按钮；基础 URL 会按当前访问地址自动生成。
- `/admin/api/state` 返回版本号；`/admin/static` 静态资源改为 `no-cache` 并加上版本参数，升级后不会继续吃旧 CSS。
- 其他：登录页与设置页响应式适配；弹窗打开时暂停后台轮询，进一步降低卡顿。

**v0.1.6（顶栏 / 设置页重构 + 性能优化）**
- 面板顶栏改为单行吸顶，设置页重构为三张卡片，登录页去掉多余滚动条。
- 性能：首 token `read1`、会话缓存开关、keep-alive、SQLite 复合索引 + WAL。
- 单账号指数退避重试；用量数据按保留期自动清理；管理员密码可在面板内修改。

**v0.1.5（思考强度对齐智谱三档）**
- 修正思考强度映射：智谱官网只有 **快速 / 深度 / 极致** 三档，客户端下拉自动收敛为
  `low/不选 → 快速`、`medium/high → 深度`、`max → 极致`，不再把 `medium` 当独立档位发给上游。
- 面板设置页写明三档对应关系。

**v0.1.4（新增智谱清言模型）**
- **新增平台：智谱清言（chatglm.cn）反代**，新增两个模型：
  - **`GLM-5.3`**：旗舰模型
  - **`GLM-5.3-Flash`**：快速模型，回复更快
- **思考强度**：智谱官网只有三档，客户端下拉会自动收敛：

  | 客户端 `reasoning_effort` | 智谱档位 | 请求体 |
  |---|---|---|
  | 不传 / `low` | 快速 | 普通对话 |
  | `medium` / `high` | 深度 | `chat_mode=zero` + `reasoning_effort=high` |
  | `max` | 极致 | `chat_mode=zero` + `reasoning_effort=max` |
- 账号添加支持选「智谱清言」；内置浏览器登录会自动抓取 `chatglm_refresh_token`，与 DeepSeek 账号独立、互不影响。
- 网关按模型名自动路由：`GLM-*` 走智谱账号池，`deepseek-*` 走 DeepSeek 账号池。
- 设置页新增智谱分组（默认模型 / 智能体 ID / 防风控延迟 / 联网搜索）。

**v0.1.3（应用图标）**
- 新增应用图标（深色圆角方块 + 白色 ∧，与面板 logo 一致）：同时作为 **exe 文件图标**和**运行窗口 / 任务栏图标**，不再是默认的 Python 图标。

**v0.1.2（内置浏览器 + 更易接入）**
- 新增：**内置浏览器改用内嵌 WebView2**（pywebview）。点「浏览器登录」在软件**内部**弹窗登录 DeepSeek，
  **完全不再调用系统 Edge/Chrome**，与用户日常浏览器**零冲突**（旧版会带出系统浏览器、甚至影响日常登录态）。
- 面板「添加账号」弹窗与账号卡片**直接写明用法**：只填名称 → 保存 → 点「浏览器登录」，凭证自动抓取，4 项凭证留空即可。
- 文档：补充 opencode **图形界面「自定义提供商」**该填什么，以及"自己用 / 别人用都填 `http://127.0.0.1:8800/v1`"。

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
