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
import json
import re
import time
from logging import Logger

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from .providers import DEFAULT_MODEL, ProviderRegistry
from .store import Store

router = APIRouter()


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
    models = {str(m).strip().lower() for m in (row.get("models") or []) if str(m).strip()}
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


def _estimate_usage(payload: dict, status: str) -> tuple[int, int]:
    """上游没返回 usage 时的兜底估算。"""
    msgs = payload.get("messages") or []
    text = "".join(
        str(m.get("content") or "") for m in msgs if isinstance(m, dict)
    )
    return _estimate_tokens(text), 0


def _usage_from_openai(resp: dict) -> tuple[int, int, int]:
    u = resp.get("usage") or {}
    p = int(u.get("prompt_tokens") or 0)
    c = int(u.get("completion_tokens") or 0)
    t = int(u.get("total_tokens") or (p + c))
    return p, c, t


MAX_ACCOUNT_TRIES = 5


def create_gateway_router(store: Store, registry: ProviderRegistry, logger: Logger) -> APIRouter:
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
            catalog = [m for m in catalog
                       if str(m.get("id", "")).strip().lower() in allowed_models]
        return {"object": "list", "data": catalog}

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        key = _extract_api_key(request)
        key_row = await run_in_threadpool(store.find_key, key) if key else None
        if not key_row:
            logger.warning(
                "鉴权失败 path=/v1/chat/completions key=%s auth=%s api-key=%s ua=%s",
                _mask_key(key),
                _mask_key(request.headers.get("authorization", "")),
                _mask_key(request.headers.get("x-api-key", "") or request.headers.get("api-key", "")),
                (request.headers.get("user-agent") or "")[:120],
            )
            return _error(401, "缺少或无效的 API Key", "authentication_error")
        if not key_row.get("enabled"):
            return _error(403, "该 API Key 已被禁用", "permission_error")

        try:
            payload = await request.json()
        except Exception:
            return _error(400, "请求体不是合法 JSON")
        if not isinstance(payload, dict):
            return _error(400, "请求体格式错误")
        if not payload.get("messages"):
            return _error(400, "缺少 messages 字段")

        model = payload.get("model") or DEFAULT_MODEL
        stream = bool(payload.get("stream"))

        allowed_accs, allowed_models = _key_scope(key_row)
        if allowed_models and str(model).strip().lower() not in allowed_models:
            return _error(403, f"该 API Key 无权使用模型 {model}", "permission_error")

        started = time.perf_counter()
        want_acc = request.headers.get("x-account-id", "").strip()
        if want_acc and allowed_accs and want_acc not in allowed_accs:
            return _error(403, "该 API Key 无权使用指定账号", "permission_error")
        ac = registry.pick_by_id(want_acc) if want_acc else registry.pick(
            model, allowed=allowed_accs or None)
        if ac is None:
            msg = ("该 Key 绑定的账号暂无可用，请检查账号状态"
                   if allowed_accs else "没有可用的账号，请先在面板添加并启用账号")
            return _error(503, msg, "unavailable")

        await run_in_threadpool(store.touch_key, key_row["id"])
        base = dict(
            api_key_id=key_row["id"], api_key_name=key_row["name"],
            account_id=ac.account["id"], account_name=ac.account["name"],
            provider=ac.account["provider"], model=model,
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
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "非流式请求失败(账号 %s, 第%d次): %s",
                        ac.account["name"], attempt + 1, exc,
                    )
                    low = str(exc).lower()
                    if any(w in low for w in ("401", "unauthorized", "token", "凭证", "登录")):
                        registry.note_auth_failure(ac.account["id"])
                    else:
                        registry.note_auth_ok(ac.account["id"])
                    if want_acc:
                        break
                    # 先尝试换一个「没试过」的账号；没有别的账号时，
                    # 就等一会儿重试同一个账号（单账号也能重试）。
                    tried.add(ac.account["id"])
                    nxt = registry.pick(model, exclude=tried,
                                        allowed=allowed_accs or None)
                    if nxt is None:
                        if attempt + 1 >= MAX_ACCOUNT_TRIES:
                            ac.mark_bad(exc=exc)
                            break
                        await asyncio.sleep(min(2 ** attempt, 8))
                        ac.mark_bad(exc=exc)
                        continue
                    ac.mark_bad(exc=exc)
                    ac = nxt
                    base.update(
                        account_id=ac.account["id"], account_name=ac.account["name"],
                        provider=ac.account["provider"],
                    )
            if resp is None:
                await run_in_threadpool(
                    store.log_usage,
                    status=f"error:{type(last_exc).__name__}", **base,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error_code=_err_code(last_exc),
                )
                code, msg = _friendly_error(last_exc) if last_exc else (502, "上游服务异常")
                return _error(code, msg, "upstream_error")
            p, c, t = _usage_from_openai(resp)
            await run_in_threadpool(
                store.log_usage, prompt_tokens=p, completion_tokens=c,
                total_tokens=t, **base,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            return JSONResponse(resp)

        # ── 流式 ──
        # 先在异步上下文里尝试拿到第一块数据，成功后再开始吐流；
        # 这样上游失败时可以换账号重试（而不是已经 200 了才发现失败）。
        started = time.perf_counter()
        first_chunk, ac, base = await _stream_with_retry(
            registry, store, payload, model, ac, base, want_acc, allowed_accs or None, logger
        )
        ttft_ms = int((time.perf_counter() - started) * 1000)
        if first_chunk is None:
            return _error(502, "所有可用账号均请求失败，请稍后重试", "upstream_error")
        async def gen():
            import asyncio
            import threading

            prompt_t = completion_t = total_t = 0
            status = "ok"
            aq: "asyncio.Queue" = asyncio.Queue()
            loop = asyncio.get_event_loop()
            _DONE = object()

            def _producer():
                # 继续消费「已经拿到前缀」的那个流（不要重新发起请求）。
                try:
                    for chunk in first_chunk:
                        loop.call_soon_threadsafe(aq.put_nowait, chunk)
                except Exception as exc:  # noqa: BLE001
                    loop.call_soon_threadsafe(aq.put_nowait, ("error", exc))
                finally:
                    loop.call_soon_threadsafe(aq.put_nowait, _DONE)

            threading.Thread(target=_producer, daemon=True).start()
            try:
                while True:
                    item = await aq.get()
                    if item is _DONE:
                        break
                    if isinstance(item, tuple):
                        status = f"error:{type(item[1]).__name__}"
                        logger.warning("流式请求失败: %s", item[1])
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
                                completion_t = int(u.get("completion_tokens") or completion_t)
                                total_t = int(u.get("total_tokens") or total_t)
                        except Exception:
                            pass
                    yield line
            finally:
                # 上游没给 usage 时用估算兜底，保证面板/客户端都有数
                if not prompt_t and not completion_t:
                    prompt_t, completion_t = _estimate_usage(payload, status)
                if not total_t:
                    total_t = prompt_t + completion_t
                try:
                    await run_in_threadpool(
                        store.log_usage, prompt_tokens=prompt_t,
                        completion_tokens=completion_t, total_tokens=total_t,
                        status=status, **base,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        first_token_ms=ttft_ms,
                        error_code=("upstream" if status != "ok" else ""),
                    )
                except Exception:
                    pass

        return StreamingResponse(gen(), media_type="text/event-stream")

    return router


async def _stream_with_retry(registry, store, payload, model, ac, base, want_acc, allowed, logger):
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
            return _ChainPrefix(prefix, it), ac, base
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "流式请求失败(账号 %s, 第%d次): %s", ac.account["name"], attempt + 1, exc
            )
            if want_acc:
                break
            tried.add(ac.account["id"])
            nxt = registry.pick(model, exclude=tried, allowed=allowed)
            if nxt is None:
                if attempt + 1 >= MAX_ACCOUNT_TRIES:
                    ac.mark_bad(exc=exc)
                    break
                await asyncio.sleep(min(2 ** attempt, 8))
                ac.mark_bad(exc=exc)
                continue
            ac.mark_bad(exc=exc)
            ac = nxt
            base.update(
                account_id=ac.account["id"], account_name=ac.account["name"],
                provider=ac.account["provider"],
            )
    if last_exc:
        logger.warning("流式请求最终失败: %s", last_exc)
    return None, ac, base


def _chunk_has_payload(line: str) -> bool:
    """判断一个 SSE 块是否包含真实内容（而不是空 role / 心跳）。"""
    if not isinstance(line, str):
        return False
    if not line.startswith("data:"):
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
        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
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

