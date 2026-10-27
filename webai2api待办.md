# webai2api 面板页面模块设计待办

> 创建：2026-10-27 ｜ 基于当前版本（v0.1.38 前后、单版本免费合并后）的面板整体评审
> 范围：`webai2api/web/index.html` + `app.css`（页面结构 / 信息组织 / 使用流程 / 视觉）
> 纪律：改完跑全量测试；面板改动如 lite_panel.html 仍在用需双面板同步；只 commit+push main，发版等用户明说

---

## P0 页面小 Bug（改动小、无风险，可先修）

- [ ] **5 秒轮询覆盖正在编辑的设置表单**：`refresh()` → `renderSettings(s)` 每次轮询无条件重置输入框 value（`setWasm`、`setDsContext`、各平台防风控延迟等）。修法：更新前判 `document.activeElement === el` 跳过，或值没变才写。
- [ ] **Escape 只能关 2 个弹窗**：keydown 处理器只写了 `['accModal','keyModal']`，实际有 6 个 modal（keyDetail / keyView / keyEdit / logModal / ocResultModal）。改为遍历 `.modal-bg.show` 全关。
- [ ] **ocResultModal 关闭按钮是空的**：动态创建时 `<button class="modal-x">` 没放 × SVG，右上角不可见。补上图标。

## P1 「接入方式」独立成 tab + 首次使用引导

> 对「免费给别人用」收益最大：拿到软件的人第一个里程碑就是接上客户端，路径越短越好。

- [ ] **接入独立 tab**：opencode 片段、Anthropic 片段（Claude Code / ZCode）、其他 OpenAI 兼容客户端指南从 API Key 页下半屏拆出，平铺到新 tab；API Key 页只管 Key 的增删改查。五 tab 变六 tab。
- [ ] **首次使用三步引导卡**（放总览页顶部）：① 添加账号 → ② 浏览器登录 → ③ 生成 Key 接入客户端，每步显示完成状态，全部完成后自动收起。替代现在空态只有一句「还没有账号，点右上角添加」。

## P2 「总览」与「用量」重新分工（消除重复）

- [ ] 现状：总览放用量图表（统计卡 + 模型环形图 + 每日柱状图），用量页又有「今日概览」，数字两处重复；且总览没有「系统现在健康吗」的视角。
- [ ] 方案（二选一）：
  - **A（推荐）**：总览改成驾驶舱——顶部一行平台健康摘要 chips（如「DeepSeek 3 可用 · GLM 2 可用 · Kimi 凭证失效 1 个」，可点跳转）+ 告警条 + 核心用量图；「今日概览」的昨日对比并入统计卡 foot 文案，删用量页重复卡。
  - **B**：两页合并成一个「用量」，五 tab 变四 tab。

## P3 设置页拆分整合

- [ ] **外观独立**：主题 / 强调色 / 背景纹理单独一张卡，或放进顶栏主题按钮下拉（现在换强调色要进设置页滚到底）。
- [ ] **四平台模型设置折叠**：四块结构完全重复（额度/性能/风控/思考强度 ×4，约 150 行重复 HTML），用 `<details>` 折叠面板，默认只展开当前用的平台。
- [ ] **关于 + 免责声明折叠**：设置页能短一半。
- [ ] （远期）四平台设置块参数化合并成一个区块，平台 seg 只切标签文案和默认值；加新平台只改一处。

## P4 账号页平台维度汇总

- [ ] 账号表上方加一行平台状态 chips（可点筛选：平台 × 可用数）。「浏览器登录 + ⋯菜单」的行结构已理顺，不动。

## P5 视觉美化小项

- [ ] 原生 `confirm()` 换成主题化弹窗，与玻璃风统一（涉及删账号/删 Key/清痕迹/退出程序等多处）。
- [ ] Key 详情弹窗「模型分布」从纯文字行改成小条形图（复用环形图 DONUT_COLORS 配色）。
- [ ] Key 行 6 按钮（查看/编辑/停用/接入/移除/删除）降噪成「查看 + 编辑 + ⋯菜单」，对齐账号行样式。

## 用户提出的更改需求

- [ ] **【Key 编辑】删除「归属人 / 到期时间 / 每日 Token 上限 / 每月 Token 上限」四项**（2026-10-27）：
  - **页面**：`index.html` 编辑 Key 弹窗（keyEditModal）删掉 `editKeyOwner`、到期时间日期按钮（`editKeyExpiresBtn`/`editKeyExpiresText` 及相关日期选择逻辑 `DP_TARGET==='keyexp'` 分支、`KEY_EXPIRES`）、`editKeyQuotaDay`、`editKeyQuotaMonth`；`openKeyEdit` / `saveKeyEdit` 对应字段同步删除；Key 列表行 scope 文案里的归属人与到期显示删除。
  - **代码**：`admin_api.py`（或 store 层）`PUT /keys/<id>` 的 owner / note(保留与否按现状) / expires_at / quota_daily / quota_monthly 字段、`GET /state` 返回中的对应字段、用量页「按 Key 用量」表的「今日 / 本月 / 额度」列与额度进度条渲染、总览的额度告警逻辑（quota 80% 提醒）、数据库字段处置（留列不用或迁移，需确认）一并清理。
  - **注意**：lite_panel.html 若仍打包需同步；跑全量测试。

## 顺手清理（重构时一并做）

- [ ] `setModelSort` 定义了两次（index.html 约 1137 / 1142 行），删一个。
- [ ] `refresh()` 里 `remove('loading')` 写了两遍。
- [ ] `empty()` 的 colspan 硬编码 9，改成按表列数。
- [ ] 三个 `DOMContentLoaded` 监听（renderThemeChips / renderAccentChips / renderPatternChips）可合并。
- [ ] 「其他 OpenAI 兼容客户端」的模型列表写死 4 个（deepseek×2 + GLM×2），改用 `window.__lastModels` 动态渲染，避免以后加模型必漏。
- [ ] lite_panel.html 去留决策：单版本合并后若旧 lite 用户归零，退役 lite_panel.html + edition.py 分支，app.py 恒走 index.html；近期不删则冻结、不再加新功能。

---

## 建议节奏

| 批次 | 内容 | 理由 |
|---|---|---|
| 第一批 | P0 三项 + 顺手清理 | 小改动、无风险、马上做 |
| 第二批 | P1 接入 tab + 引导卡 | 对新用户收益最大 |
| 第三批 | P2 总览/用量分工 | 涉及两页结构，需整体想清楚 |
| 之后再排 | P3 / P4 / P5 | 逐步打磨 |

> 2026-10-27 夜：以上全部为评审结论，**未动任何代码**。
