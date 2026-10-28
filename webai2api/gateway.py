"""OpenAI 兼容网关：/v1/chat/completions、/v1/models。

流程：
  1. 校验客户端 API Key（sk-...）
  2. 按 model 选对应 provider 的可用账号（或请求头指定 account_id）
  3. 调对应客户端（流式/非流式）
  4. 从结果提取 token 用量，写库

注意：客户端是同步阻塞实现，必须放到线程里执行，
否则会阻塞 asyncio 事件循环，导致并发请求全部卡住。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from logging import Logger
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from .providers import DEFAULT_MODEL, ProviderRegistry
from .store import Store

router = APIRouter()


def _fresh_router() -> "APIRouter":
    """每次 build_app 都用全新 router：模块级单例在多个 app 实例共存时
    （测试/热重载）会让旧路由的闭包捕获旧 registry，先注册先匹配，
    新 app 的请求全部落到旧闭包上（测试实测的 pick 泄漏）。"""
    return APIRouter()


def _clean_key(raw: str) -> str:
    """容忍客户端常见的多余格式：引号、重复 Bearer、首尾空白。"""
    v = (raw or "").strip()
    for _ in range(3):
        v = v.strip().strip('"').strip("'").strip()
        if v.lower().startswith("bearer "):
            v = v[7:].strip()
            continue
        break
    return v


def _extract_api_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return _clean_key(auth[7:])
    for name in ("x-api-key", "api-key"):
        value = request.headers.get(name, "")
        if value:
            return _clean_key(value)
    return ""


def _mask_key(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 12:
        return f"{value!r} (len={len(value)})"
    return f"{value[:8]}...{value[-4:]} (len={len(value)})"


def _err_code(exc: Exception | None) -> str:
    if exc is None:
        return ""
    text = str(exc)
    m = re.search(r"\b(4\d\d|5\d\d)\b", text)
    if m:
        return m.group(1)
    low = text.lower()
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if "token" in low or "登录" in low or "凭证" in low:
        return "auth"
    return type(exc).__name__


def _key_scope(row: dict) -> tuple[set[str], set[str]]:
    """Key 的可用账号 / 可用模型范围（空集合表示不限）。"""
    accounts = {str(a) for a in (row.get("accounts") or []) if str(a).strip()}
    models = {
        str(m).strip().lower() for m in (row.get("models") or []) if str(m).strip()
    }
    return accounts, models


def _error(status: int, message: str, code: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def _friendly_error(exc: Exception) -> tuple[int, str]:
    """把上游异常翻译成中文可读提示。"""
    text = str(exc)
    low = text.lower()

    def has(*words: str) -> bool:
        return any(w in low for w in words)

    if has("refresh_token", "refresh token", "40102", "token 已过期", "重新登录"):
        return 401, "账号凭证已失效，请到面板重新登录该账号"
    if has("401", "unauthorized", "authentication"):
        return 401, "账号未登录或登录已过期，请到面板重新登录"
    if has("403", "forbidden", "risk", "风控", "verify"):
        return 403, "账号被上游风控拦截，请稍后再试或更换账号"
    if has("429", "rate limit", "too many", "限流", "频繁"):
        return 429, "请求过于频繁，已被上游限流，请稍后重试"
    if has("timeout", "timed out", "超时"):
        return 504, "上游响应超时，请重试或换用其他模型"
    if has("connection", "connect", "network", "unreachable", "ssl", "握手"):
        return 503, "无法连接上游服务，请检查网络后重试"
    if has("余额", "quota", "insufficient", "额度"):
        return 402, "账号额度不足"
    if has("content", "sensitive", "审核", "违规"):
        return 400, "内容被上游审核拦截，请调整提问后重试"
    return 502, f"上游服务异常：{text[:200]}"


def _estimate_tokens(text: str) -> int:
    """中文约 1.5 字/token，其余约 4 字符/token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


def _messages_text(payload: dict) -> str:
    msgs = payload.get("messages") or []
    return "".join(
        _msg_text(m.get("content")) if isinstance(m, dict) else "" for m in msgs
    )


def _estimate_prompt(payload: dict) -> int:
    """上游没返回 usage 时，按 messages 估算输入 token。"""
    return _estimate_tokens(_messages_text(payload))


def _estimate_completion(payload: dict, text: str = "") -> int:
    """上游没返回 usage 时，按实际输出估算 completion；没有输出再退回 max_tokens 的一半。

    注意：reasoner 的 reasoning_content 也是真实计费输出，必须计入，
    否则取消/断流记录会严重低估。
    """
    if text:
        return _estimate_tokens(text)
    try:
        return max(1, int(payload.get("max_tokens") or 0) // 2)
    except Exception:
        return 0


def _usage_tokens_from_openai(resp: dict) -> tuple[int, int, int]:
    """从上游 usage 取数；缺 completion 时用 DeepSeek 的
    completion_tokens_details.reasoning_tokens 兜底（思考 + 正文都是计费输出）。"""
    u = resp.get("usage") or {}
    p = int(u.get("prompt_tokens") or 0)
    c = int(u.get("completion_tokens") or 0)
    if not c:
        det = u.get("completion_tokens_details") or {}
        try:
            c = int(det.get("reasoning_tokens") or 0)
        except Exception:
            c = 0
    t = int(u.get("total_tokens") or (p + c))
    return p, c, t


def _response_text(resp: dict) -> str:
    """从非流式响应里拼出输出文本（用于估算 completion）。"""
    out = []
    for ch in resp.get("choices") or []:
        msg = ch.get("message") or {}
        out.append(str(msg.get("content") or ""))
        out.append(str(msg.get("reasoning_content") or ""))
    return "".join(out)


def _classify_error(exc: Exception | None) -> str:
    """把上游异常分类，决定「换号 / 冷却 / 直接失败」。"""
    if exc is None:
        return "unknown"
    text = str(exc)
    low = text.lower()

    def has(*words: str) -> bool:
        return any(w in low for w in words)

    if has("429", "rate limit", "too many", "限流", "频繁"):
        return "rate"
    if has(
        "封禁",
        "被封",
        "封号",
        "封停",
        "账号已被限制",
        "账号异常",
        "banned",
        "suspended",
    ):
        return "banned"
    if re.search(r"\b401\b", text) or has(
        "unauthorized",
        "refresh_token",
        "40102",
        "token 已过期",
        "token expired",
        "invalid token",
        "access token",
        "登录",
        "凭证",
    ):
        return "auth"
    if has("content", "sensitive", "审核", "违规"):
        return "content"
    if has(
        "timeout",
        "timed out",
        "超时",
        "connection",
        "connect",
        "network",
        "unreachable",
        "ssl",
        "握手",
    ):
        return "network"
    if re.search(r"\b5\d\d\b", text):
        return "server"
    return "unknown"


def _apply_failure(registry: ProviderRegistry, ac, exc: Exception) -> str:
    """按错误类型更新账号状态，返回分类结果。"""
    kind = _classify_error(exc)
    if kind == "auth":
        registry.note_auth_failure(ac.account["id"])
        ac.mark_bad(600, exc=exc)
    elif kind == "rate":
        registry.note_auth_ok(ac.account["id"])
        ac.mark_bad(exc=exc)
    elif kind == "banned":
        # 封禁类：mark_bad 内部按关键词给 7 天长冷却；不计 auth 失败
        # （note_auth_failure 可能把账号在库里误禁用，封禁应保持可观察）
        ac.mark_bad(exc=exc)
    elif kind == "content":
        ac.mark_ok()
    else:
        registry.note_auth_ok(ac.account["id"])
        ac.mark_bad(30, exc=exc)
    return kind


# ── 会话粘性：同一段对话尽量固定在同一账号，避免会话缓存失效 ──
_STICKY_MAX = 2000
_STICKY_TTL = 6 * 3600
_STICKY: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
_STICKY_LOCK = threading.Lock()


def _msg_text(content: Any) -> str:
    """把 message.content 规整成纯文本（兼容 OpenAI 的多段 content 数组）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, str):
                parts.append(seg)
            elif isinstance(seg, dict):
                txt = seg.get("text")
                if txt is None and isinstance(seg.get("content"), str):
                    txt = seg.get("content")
                if txt is not None:
                    parts.append(str(txt))
        return "\n".join(parts)
    return str(content)


def _conv_key(payload: dict, key_id: str) -> str:
    """用「Key + 模型 + 第一条 user + 第一条 assistant 文本」做指纹。

    跳过所有 system 消息：客户端（如 opencode）每轮的 system 内容可能变化，
    若计入指纹会让账号粘性失效、会话缓存跨账号 miss。
    """
    msgs = payload.get("messages") or []
    first_user = ""
    first_assistant = ""
    asst_seen = False
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system" or role == "tool":
            continue
        if m.get("tool_calls"):
            # 工具调用轮同样要计入指纹：assistant.content 为 null 时若跳过，
            # 工具轮与首轮算出同一个 key，上游会新建会话、丢掉工具结果。
            if not asst_seen:
                names = [
                    str((c.get("function") or {}).get("name") or "")
                    for c in m["tool_calls"]
                    if isinstance(c, dict)
                ]
                first_assistant = "tool_call:" + ",".join(names)
                asst_seen = True
            if first_user:
                break
            continue
        text = _msg_text(m.get("content"))
        if role == "user" and not first_user:
            first_user = text[:2000]
        elif role == "assistant" and not asst_seen:
            first_assistant = text[:500]
            asst_seen = True
        if first_user and asst_seen:
            break
    raw = json.dumps(
        [key_id, payload.get("model"), first_user, first_assistant], ensure_ascii=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _sticky_get(ck: str) -> str:
    if not ck:
        return ""
    with _STICKY_LOCK:
        item = _STICKY.get(ck)
        if not item:
            return ""
        aid, ts = item
        if time.time() - ts > _STICKY_TTL:
            _STICKY.pop(ck, None)
            return ""
        _STICKY.move_to_end(ck)
        return aid


def _sticky_set(ck: str, account_id: str) -> None:
    if not ck:
        return
    with _STICKY_LOCK:
        _STICKY[ck] = (account_id, time.time())
        _STICKY.move_to_end(ck)
        while len(_STICKY) > _STICKY_MAX:
            _STICKY.popitem(last=False)


# ── 额度预占：并发请求不会同时挤过同一个额度检查 ──
_QUOTA_INFLIGHT: dict[str, tuple[int, float]] = {}
_QUOTA_LOCK = threading.Lock()
_QUOTA_STALE = 30 * 60


def _quota_sweep() -> None:
    """清理异常路径（客户端秒断、进程内异常）遗留的预占。调用方需持锁。"""
    now = time.time()
    for kid in [k for k, (_, ts) in _QUOTA_INFLIGHT.items() if now - ts > _QUOTA_STALE]:
        _QUOTA_INFLIGHT.pop(kid, None)


def _quota_inflight(key_id: str) -> int:
    with _QUOTA_LOCK:
        _quota_sweep()
        item = _QUOTA_INFLIGHT.get(key_id)
        return item[0] if item else 0


def _quota_reserve(key_id: str, amount: int) -> None:
    amount = max(0, int(amount))
    if not amount:
        return
    with _QUOTA_LOCK:
        _quota_sweep()
        cur = _QUOTA_INFLIGHT.get(key_id, (0, 0.0))[0]
        _QUOTA_INFLIGHT[key_id] = (cur + amount, time.time())


def _quota_release(key_id: str, amount: int) -> None:
    amount = max(0, int(amount))
    if not amount:
        return
    with _QUOTA_LOCK:
        cur = _QUOTA_INFLIGHT.get(key_id, (0, 0.0))[0] - amount
        if cur > 0:
            _QUOTA_INFLIGHT[key_id] = (cur, time.time())
        else:
            _QUOTA_INFLIGHT.pop(key_id, None)


MAX_ACCOUNT_TRIES = 5


def create_gateway_router(
    store: Store, registry: ProviderRegistry, logger: Logger
) -> APIRouter:
    router = _fresh_router()

    @router.get("/v1/models")
    async def list_models(request: Request):
        key = _extract_api_key(request)
        row = await run_in_threadpool(store.find_key, key) if key else None
        if not key or not row:
            logger.warning(
                "鉴权失败 path=/v1/models key=%s auth=%s ua=%s",
                _mask_key(key),
                _mask_key(request.headers.get("authorization", "")),
                (request.headers.get("user-agent") or "")[:120],
            )
            return _error(401, "缺少或无效的 API Key", "authentication_error")
        catalog = await run_in_threadpool(registry.model_catalog)
        _, allowed_models = _key_scope(row)
        if allowed_models:
            catalog = [
                m
                for m in catalog
                if str(m.get("id", "")).strip().lower() in allowed_models
            ]
        return {"object": "list", "data": catalog}

    async def _chat_core(request: Request, internal: dict | None = None):
        """chat 链路的实现体。

        `internal` 非空表示来自 /v1/messages 的内部调用：直接复用已在内存
        里的 payload。绝不能改写 `request._body` 再递归调用——Starlette 会
        把 body 缓存住，改写后 request.json() 仍返回旧内容（Anthropic 原始
        body），导致整条链路按错误形状跑。
        """
        key = _extract_api_key(request)
        key_row = await run_in_threadpool(store.find_key, key) if key else None
        if not key_row:
            logger.warning(
                "鉴权失败 path=/v1/chat/completions key=%s auth=%s api-key=%s ua=%s",
                _mask_key(key),
                _mask_key(request.headers.get("authorization", "")),
                _mask_key(
                    request.headers.get("x-api-key", "")
                    or request.headers.get("api-key", "")
                ),
                (request.headers.get("user-agent") or "")[:120],
            )
            return _error(401, "缺少或无效的 API Key", "authentication_error")
        if not key_row.get("enabled"):
            return _error(403, "该 API Key 已被禁用", "permission_error")

        try:
            payload = dict(internal) if internal is not None else await request.json()
        except Exception:
            return _error(400, "请求体不是合法 JSON")
        if not isinstance(payload, dict):
            return _error(400, "请求体格式错误")
        if not payload.get("messages"):
            return _error(400, "缺少 messages 字段")

        model = payload.get("model") or DEFAULT_MODEL
        stream = bool(payload.get("stream"))
        # 内部字段（下划线前缀）：仅本进程使用，translator 白名单取字段不会转发到上游。
        payload["_wapi_key_id"] = key_row["id"]

        if key_row.get("expires_at") and float(key_row["expires_at"]) < time.time():
            return _error(403, "该 API Key 已过期", "permission_error")

        daily_quota = int(key_row.get("quota_daily") or 0)
        monthly_quota = int(key_row.get("quota_monthly") or 0)
        # 预占额度：并发请求在「检查通过但还没记账」的窗口里不会同时挤过去。
        # 预占量取「估算输入」与「剩余额度」的较小值，保证最后一个请求仍能放行。
        reserve = 0
        if daily_quota or monthly_quota:
            used_today, used_month = await run_in_threadpool(
                store.key_quota_status, key_row["id"]
            )
            inflight = _quota_inflight(key_row["id"])
            left_day = (daily_quota - used_today - inflight) if daily_quota else None
            left_month = (
                (monthly_quota - used_month - inflight) if monthly_quota else None
            )
            if left_day is not None and left_day <= 0:
                return _error(
                    429,
                    f"该 Key 今日额度已用完（{used_today}/{daily_quota} tokens）",
                    "rate_limit_error",
                )
            if left_month is not None and left_month <= 0:
                return _error(
                    429,
                    f"该 Key 本月额度已用完（{used_month}/{monthly_quota} tokens）",
                    "rate_limit_error",
                )
            lefts = [x for x in (left_day, left_month) if x is not None]
            reserve = max(1, min(_estimate_prompt(payload), min(lefts)))
            _quota_reserve(key_row["id"], reserve)

        allowed_accs, allowed_models = _key_scope(key_row)
        if allowed_models and str(model).strip().lower() not in allowed_models:
            _quota_release(key_row["id"], reserve)
            return _error(403, f"该 API Key 无权使用模型 {model}", "permission_error")

        started = time.perf_counter()
        want_acc = request.headers.get("x-account-id", "").strip()
        if want_acc and allowed_accs and want_acc not in allowed_accs:
            _quota_release(key_row["id"], reserve)
            return _error(403, "该 API Key 无权使用指定账号", "permission_error")
        conv = _conv_key(payload, key_row["id"])
        sticky = _sticky_get(conv) if not want_acc else ""
        try:
            ac = (
                registry.pick_by_id(want_acc)
                if want_acc
                else registry.pick(
                    model, allowed=allowed_accs or None, prefer=sticky or None
                )
            )
        except Exception as exc:  # noqa: BLE001
            _quota_release(key_row["id"], reserve)
            logger.error("选号失败: %s", exc)
            return _error(500, "内部错误：账号池异常", "internal_error")
        if ac is None:
            _quota_release(key_row["id"], reserve)
            cds = registry.cooldowns()
            wait = max(cds.values()) if cds else 0
            if wait:
                mins = max(1, int(wait // 60))
                return _error(
                    429, f"账号都在冷却中，请约 {mins} 分钟后重试", "rate_limit_error"
                )
            msg = (
                "该 Key 绑定的账号暂无可用，请检查账号状态"
                if allowed_accs
                else "没有可用的账号，请先在面板添加并启用账号"
            )
            return _error(503, msg, "unavailable")

        try:
            await run_in_threadpool(store.touch_key, key_row["id"])
        except Exception as exc:  # noqa: BLE001
            _quota_release(key_row["id"], reserve)
            logger.error("touch_key 失败: %s", exc)
            return _error(500, "内部错误：状态更新失败", "internal_error")
        base = dict(
            api_key_id=key_row["id"],
            api_key_name=key_row["name"],
            account_id=ac.account["id"],
            account_name=ac.account["name"],
            provider=ac.account["provider"],
            model=model,
        )

        if not stream:
            resp = None
            last_exc: Exception | None = None
            tried: set[str] = set()
            for attempt in range(MAX_ACCOUNT_TRIES):
                try:
                    resp = await run_in_threadpool(ac.client.chat_completion, payload)
                    ac.mark_ok()
                    registry.note_auth_ok(ac.account["id"])
                    _sticky_set(conv, ac.account["id"])
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "非流式请求失败(账号 %s, 第%d次): %s",
                        ac.account["name"],
                        attempt + 1,
                        exc,
                    )
                    kind = _apply_failure(registry, ac, exc)
                    # 内容审核 / 凭证失效：换号也没用，直接返回
                    if want_acc or kind in ("content", "auth"):
                        break
                    tried.add(ac.account["id"])
                    nxt = registry.pick(
                        model, exclude=tried, allowed=allowed_accs or None
                    )
                    if nxt is None:
                        if attempt + 1 >= MAX_ACCOUNT_TRIES:
                            break
                        # 没有别的账号：只有网络/5xx 才值得原地重试
                        if kind not in ("network", "server"):
                            break
                        await asyncio.sleep(min(2**attempt, 8))
                        continue
                    ac = nxt
                    base.update(
                        account_id=ac.account["id"],
                        account_name=ac.account["name"],
                        provider=ac.account["provider"],
                    )
            if resp is None:
                _quota_release(key_row["id"], reserve)
                await run_in_threadpool(
                    store.log_usage,
                    status=f"error:{type(last_exc).__name__}",
                    **base,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error_code=_err_code(last_exc),
                )
                code, msg = (
                    _friendly_error(last_exc) if last_exc else (502, "上游服务异常")
                )
                return _error(code, msg, "upstream_error")
            p, c, t = _usage_tokens_from_openai(resp)
            if not p:
                p = _estimate_prompt(payload)
            if not c:
                c = _estimate_completion(payload, _response_text(resp))
            if not t:
                t = p + c
            _quota_release(key_row["id"], reserve)
            await run_in_threadpool(
                store.log_usage,
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=t,
                **base,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            return JSONResponse(resp)

        # ── 流式 ──
        # 先在异步上下文里尝试拿到第一块数据，成功后再开始吐流；
        # 这样上游失败时可以换账号重试（而不是已经 200 了才发现失败）。
        # ttft 从这里单独起算；latency 仍用链路起点 started（含鉴权/选号）。
        t_pick = time.perf_counter()
        first_chunk, ac, base = await _stream_with_retry(
            registry,
            store,
            payload,
            model,
            ac,
            base,
            want_acc,
            allowed_accs or None,
            logger,
            conv,
        )
        ttft_ms = int((time.perf_counter() - t_pick) * 1000)
        if first_chunk is None:
            _quota_release(key_row["id"], reserve)
            return _error(502, "所有可用账号均请求失败，请稍后重试", "upstream_error")

        async def gen():
            prompt_t = completion_t = total_t = 0
            status = "ok"
            out_parts: list[str] = []
            aq: "asyncio.Queue" = asyncio.Queue()
            loop = asyncio.get_event_loop()
            _DONE = object()
            cancelled = threading.Event()

            def _push(item):
                try:
                    loop.call_soon_threadsafe(aq.put_nowait, item)
                except RuntimeError:
                    pass

            def _producer():
                # 继续消费「已经拿到前缀」的那个流（不要重新发起请求）。
                try:
                    for chunk in first_chunk:
                        if cancelled.is_set():
                            break
                        _push(chunk)
                except Exception as exc:  # noqa: BLE001
                    _push(("error", exc))
                finally:
                    # 客户端断开时在这里关闭上游流（必须由迭代线程自己关）
                    close = getattr(first_chunk, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
                    _push(_DONE)

            threading.Thread(target=_producer, daemon=True).start()
            try:
                while True:
                    item = await aq.get()
                    if item is _DONE:
                        break
                    if isinstance(item, tuple):
                        status = f"error:{type(item[1]).__name__}"
                        logger.warning("流式请求失败: %s", item[1])
                        _apply_failure(registry, ac, item[1])
                        _, msg = _friendly_error(item[1])
                        err = {"error": {"message": msg, "type": "upstream_error"}}
                        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    line = item
                    if line.startswith("data:") and "usage" in line:
                        try:
                            data = json.loads(line[5:].strip() or "{}")
                            u = data.get("usage") or {}
                            if u:
                                prompt_t = int(u.get("prompt_tokens") or prompt_t)
                                completion_t = int(
                                    u.get("completion_tokens") or completion_t
                                )
                                total_t = int(u.get("total_tokens") or total_t)
                        except Exception:
                            pass
                    elif line.startswith("data:"):
                        # 攒输出文本，供上游不给 usage 时估算 completion
                        # （正文与 reasoning_content 都要攒：思考同样是计费输出）
                        body = line[5:].strip()
                        if body and body != "[DONE]" and len(out_parts) < 200000:
                            try:
                                obj = json.loads(body)
                                for ch in obj.get("choices") or []:
                                    d = ch.get("delta") or {}
                                    piece = (
                                        d.get("content")
                                        or d.get("reasoning_content")
                                        or ""
                                    )
                                    if piece:
                                        out_parts.append(str(piece))
                            except Exception:
                                pass
                    yield line
            except asyncio.CancelledError:
                # 客户端断开：通知上游线程停止拉取，避免继续消耗额度
                cancelled.set()
                status = "cancelled"
                raise
            finally:
                _quota_release(key_row["id"], reserve)
                # 上游没给 usage 时用估算兜底，保证面板/客户端都有数
                if not prompt_t:
                    prompt_t = _estimate_prompt(payload)
                if not completion_t:
                    completion_t = _estimate_completion(payload, "".join(out_parts))
                if not total_t:
                    total_t = prompt_t + completion_t
                _log = dict(
                    prompt_tokens=prompt_t,
                    completion_tokens=completion_t,
                    total_tokens=total_t,
                    status=status,
                    **base,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    first_token_ms=ttft_ms,
                    error_code=(
                        ""
                        if status == "ok"
                        else ("cancelled" if status == "cancelled" else "upstream")
                    ),
                )
                if status == "cancelled":
                    # 任务已被取消，不能再 await，交给后台线程落库
                    threading.Thread(
                        target=lambda: store.log_usage(**_log), daemon=True
                    ).start()
                else:
                    try:
                        await run_in_threadpool(store.log_usage, **_log)
                    except Exception:
                        pass

        return StreamingResponse(gen(), media_type="text/event-stream")

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _chat_core(request)

    # ── Anthropic Messages 协议（Claude Code / ZCode 等客户端）────
    @router.post("/v1/messages")
    async def anthropic_messages(request: Request):
        """Anthropic /v1/messages 端点：入口翻译成内部 OpenAI 形状后
        复用 /v1/chat/completions 的完整链路（鉴权/额度/选号/重试/用量），
        响应与非流式/流式输出再翻译回 Anthropic 格式。"""
        from .anthropic_adapter import (
            AnthropicStreamTranslator,
            anthropic_error,
            anthropic_to_openai,
            openai_error_to_anthropic,
            openai_to_anthropic,
        )

        # Anthropic 用 x-api-key 鉴权（_extract_api_key 已兼容），同时
        # 接受 Authorization Bearer——两种客户端习惯都照顾。
        if not _extract_api_key(request):
            status, content = anthropic_error(
                401, "authentication_error", "缺少或无效的 API Key"
            )
            return JSONResponse(status_code=status, content=content)

        try:
            abody = await request.json()
        except Exception:
            status, content = anthropic_error(
                400, "invalid_request_error", "请求体不是合法 JSON"
            )
            return JSONResponse(status_code=status, content=content)
        if not isinstance(abody, dict) or not abody.get("messages"):
            status, content = anthropic_error(
                400, "invalid_request_error", "缺少 messages 字段"
            )
            return JSONResponse(status_code=status, content=content)

        payload = anthropic_to_openai(abody)
        payload["_wapi_key_id"] = ""  # 占位：chat_completions 会写入真实值
        model = payload.get("model") or DEFAULT_MODEL
        want_stream = bool(payload.get("stream"))
        # Anthropic 的 system 已由 anthropic_to_openai 转成 messages[0]；
        # 只有当转换结果里确实没有任何 system 消息时才补占位——
        # 占位恒插入会把客户端真实 system 挤到第二位，而 Kimi/Qwen 只取
        # 第一条 system，真实指令会整个丢失（审查发现的真 bug）。
        _has_system = any(
            isinstance(m, dict) and m.get("role") == "system"
            for m in (payload.get("messages") or [])
        )
        if not _has_system and not payload.get("system"):
            # 缺失时 chat 链路没有 system 消息，三家网页端会以「无视工具
            # 说明」的口吻作答（实测：工具型对话回填结果后不认账）。
            # 补一句占位 system，工具注入落在这里。
            payload["system"] = "你是 webai2api 网关转发的助手。"
            payload["messages"].insert(
                0, {"role": "system", "content": payload["system"]}
            )

        # 复用 chat 链路的全部实现（鉴权/额度/选号/重试/用量）：payload 直接
        # 以内存交接，不经过 request body（见 _chat_core 的说明）。
        resp = await _chat_core(request, payload)

        # 错误（鉴权/额度/选号/上游）统一转 Anthropic 错误格式
        if resp.status_code != 200:
            try:
                data = json.loads(resp.body)
            except Exception:
                data = {}
            oai = data.get("error") or {}
            err_type = str(oai.get("type") or "api_error")
            if resp.status_code == 401:
                err_type = "authentication_error"
            elif resp.status_code == 429:
                err_type = "rate_limit_error"
            elif resp.status_code == 403:
                err_type = "permission_error"
            elif err_type == "upstream_error":
                err_type = "api_error"
            aerr = anthropic_error(
                resp.status_code, err_type, str(oai.get("message") or "上游错误")
            )
            return JSONResponse(status_code=aerr[0], content=aerr[1])

        if not want_stream:
            data = json.loads(resp.body)
            return JSONResponse(openai_to_anthropic(data, model))

        # 流式：把 OpenAI SSE 块流翻译成 Anthropic 事件流
        async def agen():
            tr = AnthropicStreamTranslator(model)
            async for line in _iter_sse_lines(resp):
                body_txt = line[5:].strip()
                if not body_txt or body_txt == "[DONE]":
                    continue
                try:
                    obj = json.loads(body_txt)
                except Exception:
                    continue
                for ev in tr.feed(obj):
                    yield ev
            for ev in tr.finish():
                yield ev

        return StreamingResponse(agen(), media_type="text/event-stream")

    return router


async def _stream_with_retry(
    registry, store, payload, model, ac, base, want_acc, allowed, logger, conv=""
):
    """流式请求：先取到第一块「有内容」的数据，失败则换账号重试。

    注意：客户端生成器一开始只发一个空的 role chunk，所以不能拿第一块就判定成功，
    必须等到真正有 content / reasoning / tool_calls 的块，才能确认上游正常。
    """
    tried: set[str] = set()
    last_exc: Exception | None = None
    for attempt in range(MAX_ACCOUNT_TRIES):
        try:
            it = ac.client.stream_chat_completion(payload)
            prefix = await run_in_threadpool(_take_until_content, it)
            if prefix is None:
                raise RuntimeError("上游未返回任何内容")
            ac.mark_ok()
            registry.note_auth_ok(ac.account["id"])
            _sticky_set(conv, ac.account["id"])
            return _ChainPrefix(prefix, it), ac, base
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "流式请求失败(账号 %s, 第%d次): %s",
                ac.account["name"],
                attempt + 1,
                exc,
            )
            kind = _apply_failure(registry, ac, exc)
            if want_acc or kind in ("content", "auth"):
                break
            tried.add(ac.account["id"])
            nxt = registry.pick(model, exclude=tried, allowed=allowed)
            if nxt is None:
                if attempt + 1 >= MAX_ACCOUNT_TRIES:
                    break
                if kind not in ("network", "server", "unknown"):
                    break
                await asyncio.sleep(min(2**attempt, 8))
                continue
            ac = nxt
            base.update(
                account_id=ac.account["id"],
                account_name=ac.account["name"],
                provider=ac.account["provider"],
            )
    if last_exc:
        logger.warning("流式请求最终失败: %s", last_exc)
    return None, ac, base


def _split_sse_events(line: str) -> list[str]:
    """把一个流片段拆成若干条完整 SSE 事件（每条一行）。

    上游可能把多个 `data: {...}` 一次性刷进同一个 chunk；若整段只解析第一条，
    后续的 tool_calls / finish_reason 会被静默丢掉（Anthropic 端点收不到
    工具调用与结束原因，表现为「回填后再调一次工具」）。
    """
    if not isinstance(line, str):
        line = line.decode("utf-8", "replace") if isinstance(line, bytes) else str(line)
    out: list[str] = []
    for raw in line.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("data:"):
            out.append(raw)
        elif out:
            # 多行 data 的续行（SSE 规范），拼回上一条
            out[-1] = out[-1] + "\n" + raw
    return out


async def _iter_sse_lines(resp):
    """把上游流片段逐条泵成完整 SSE 事件行（不丢块、不留残片）。"""
    buf = ""
    async for chunk in resp.body_iterator:
        buf += chunk if isinstance(chunk, str) else chunk.decode("utf-8", "replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if line.startswith("data:"):
                yield line
    tail = buf.strip()
    if tail.startswith("data:"):
        yield tail


def _chunk_has_payload(line: str) -> bool:
    """判断一个 SSE 片段是否包含真实内容（而不是空 role / 心跳）。

    上游一个片段里可能挤了多条 data 事件，逐个解析，避免只看第一条就
    误判「这一块没内容」。
    """
    for ev in _split_sse_events(line):
        if _event_has_payload(ev):
            return True
    return False


def _event_has_payload(line: str) -> bool:
    if not isinstance(line, str) or not line.startswith("data:"):
        return False
    body = line[5:].strip()
    if not body or body == "[DONE]":
        return False
    try:
        obj = json.loads(body)
    except Exception:
        return False
    if obj.get("error"):
        return True
    for ch in obj.get("choices") or []:
        delta = ch.get("delta") or {}
        if (
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("tool_calls")
        ):
            return True
        if ch.get("finish_reason"):
            return True
    return False


def _take_until_content(it, limit: int = 200):
    """消费生成器直到出现有内容的块，返回已消费的块列表。"""
    prefix: list[str] = []
    for _ in range(limit):
        try:
            line = next(it)
        except StopIteration:
            return prefix or None
        prefix.append(line)
        if _chunk_has_payload(line):
            return prefix
    return prefix or None


class _ChainPrefix:
    """把「已消费的前缀块」和剩余迭代器拼成完整迭代器。"""

    def __init__(self, prefix, rest):
        self._prefix = prefix
        self._rest = rest

    def __iter__(self):
        yield from self._prefix
        yield from self._rest

    def close(self) -> None:
        close = getattr(self._rest, "close", None)
        if callable(close):
            close()
