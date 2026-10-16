"""OpenAI 兼容网关：/v1/chat/completions、/v1/models。

流程：
  1. 校验客户端 API Key（sk-...）
  2. 选一个可用账号（或请求头指定 account_id）
  3. 调 DeepSeek 客户端（流式/非流式）
  4. 从结果提取 token 用量，写库

注意：DeepSeek 客户端是同步阻塞实现，必须放到线程里执行，
否则会阻塞 asyncio 事件循环，导致并发请求全部卡住。
"""
from __future__ import annotations

import json
from logging import Logger

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters.deepseek.pool import ClientPool
from .store import Store

router = APIRouter()

MODELS = [
    {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek-web"},
    {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek-web"},
]


def _extract_api_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _error(status: int, message: str, code: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def _usage_from_openai(resp: dict) -> tuple[int, int, int]:
    u = resp.get("usage") or {}
    p = int(u.get("prompt_tokens") or 0)
    c = int(u.get("completion_tokens") or 0)
    t = int(u.get("total_tokens") or (p + c))
    return p, c, t


def create_gateway_router(store: Store, pool: ClientPool, logger: Logger) -> APIRouter:
    @router.get("/v1/models")
    async def list_models(request: Request):
        key = _extract_api_key(request)
        row = await run_in_threadpool(store.find_key, key) if key else None
        if not key or not row:
            return _error(401, "缺少或无效的 API Key", "authentication_error")
        return {"object": "list", "data": MODELS}

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        key = _extract_api_key(request)
        key_row = await run_in_threadpool(store.find_key, key) if key else None
        if not key_row:
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

        model = payload.get("model") or "deepseek-chat"
        stream = bool(payload.get("stream"))

        want_acc = request.headers.get("x-account-id", "").strip()
        ac = pool.pick_by_id(want_acc) if want_acc else pool.pick()
        if ac is None:
            return _error(503, "没有可用的账号，请先在面板添加并启用账号", "unavailable")

        await run_in_threadpool(store.touch_key, key_row["id"])
        base = dict(
            api_key_id=key_row["id"], api_key_name=key_row["name"],
            account_id=ac.account["id"], account_name=ac.account["name"],
            provider=ac.account["provider"], model=model,
        )

        if not stream:
            resp = None
            last_exc: Exception | None = None
            for _ in range(2):  # 失败时换账号重试一次
                try:
                    resp = await run_in_threadpool(ac.client.chat_completion, payload)
                    ac.mark_ok()
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    ac.mark_bad()
                    logger.warning("非流式请求失败(账号 %s): %s", ac.account["name"], exc)
                    if want_acc:
                        break
                    nxt = pool.pick(exclude={ac.account["id"]})
                    if nxt is None:
                        break
                    ac = nxt
                    base.update(account_id=ac.account["id"], account_name=ac.account["name"])
            if resp is None:
                await run_in_threadpool(
                    store.log_usage,
                    status=f"error:{type(last_exc).__name__}", **base
                )
                return _error(502, f"上游错误: {last_exc}", "upstream_error")
            p, c, t = _usage_from_openai(resp)
            await run_in_threadpool(
                store.log_usage, prompt_tokens=p, completion_tokens=c,
                total_tokens=t, **base
            )
            return JSONResponse(resp)

        # ── 流式 ──
        async def gen():
            import asyncio
            import queue as _queue

            prompt_t = completion_t = total_t = 0
            status = "ok"
            q: "_queue.Queue" = _queue.Queue(maxsize=256)
            _DONE = object()

            def _producer():
                # 独立线程里跑同步生成器，逐块塞进队列
                try:
                    for chunk in ac.client.stream_chat_completion(payload):
                        q.put(("data", chunk))
                except Exception as exc:  # noqa: BLE001
                    q.put(("error", exc))
                finally:
                    q.put(("done", _DONE))

            import threading
            threading.Thread(target=_producer, daemon=True).start()

            loop = asyncio.get_event_loop()
            try:
                while True:
                    try:
                        kind, item = await loop.run_in_executor(None, q.get)
                    except Exception:
                        break
                    if kind == "done":
                        break
                    if kind == "error":
                        status = f"error:{type(item).__name__}"
                        logger.warning("流式请求失败: %s", item)
                        err = {"error": {"message": f"上游错误: {item}",
                                         "type": "upstream_error"}}
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
                if not total_t:
                    total_t = prompt_t + completion_t
                try:
                    await run_in_threadpool(
                        store.log_usage, prompt_tokens=prompt_t,
                        completion_tokens=completion_t, total_tokens=total_t,
                        status=status, **base
                    )
                except Exception:
                    pass

        return StreamingResponse(gen(), media_type="text/event-stream")

    return router
