"""智谱清言图片上传（chat_upload）。

上游流程（2026-10-30 真机抓帧确认）：
  1. POST /chatglm/productivity-api/file/chat_upload
     multipart: file + from=chat + assistant_id
     → {status:0, result:{file_id, file_url(带签名的临时链接), file_name,
        file_size, file_type, ...}}
  2. file_id/file_url 进消息 content 的 image part（字段名必须逐字对齐网页端）：
     {"type":"image","image":[{"file_name","file_id","image_url","file_size","order"}]}

注意：/chatglm/backend-api/assistant/file/upload 是智能体文档上传，聊天账号会回
「权限不足」；聊天传图只走 productivity-api 的 chat_upload。
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
import uuid
from logging import Logger
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .client import ChatGlmWebClient

CHAT_UPLOAD_PATH = "/chatglm/productivity-api/file/chat_upload"
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_CACHE = 256

_DATA_URL_RE = re.compile(r"^data:([^;,]*?)(;base64)?,(.*)$", re.DOTALL)


def extract_image_urls(messages: list[dict]) -> list[str]:
    """提取消息里的 image_url（兼容对象与字符串两种写法），保持出现顺序。"""
    urls: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            u = part.get("image_url")
            if isinstance(u, dict):
                u = u.get("url")
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    return urls


class GlmImageUploader:
    """把 OpenAI image_url（data URL / http URL）上传成 GLM 文件引用。

    内部维护 image_url hash → 上传结果的 LRU 缓存：同一张图在会话多轮里
    反复出现时只上传一次。
    """

    def __init__(self, client: "ChatGlmWebClient", logger: Logger) -> None:
        self._client = client
        self._logger = logger
        self._cache: dict[str, dict] = {}

    # ── 公共入口 ─────────────────────────────────────────

    def upload(self, image_url: str, *, name_hint: str = "") -> Optional[dict]:
        """上传一张图，返回 image part 条目；失败返回 None（调用方降级纯文本）。"""
        if not image_url:
            return None
        key = hashlib.sha256(image_url.encode("utf-8", "ignore")).hexdigest()
        cached = self._cache.get(key)
        if cached:
            return dict(cached)
        try:
            content, mime, filename = self._decode_image_url(image_url, name_hint)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("GlmUpload | 解析 image_url 失败: %s", exc)
            return None
        if len(content) > _MAX_IMAGE_BYTES:
            self._logger.warning(
                "GlmUpload | 图片过大（%d 字节），跳过", len(content)
            )
            return None
        result = self._upload_bytes(content, mime, filename)
        if not result:
            return None
        item = {
            "file_name": result.get("file_name") or filename,
            "file_id": result.get("file_id") or "",
            "image_url": result.get("file_url") or "",
            "file_size": result.get("file_size") or len(content),
        }
        if not item["image_url"]:
            self._logger.warning("GlmUpload | 上传响应缺少 file_url，跳过")
            return None
        if len(self._cache) >= _MAX_CACHE:
            self._cache.clear()
        self._cache[key] = item
        return dict(item)

    # ── 内部实现 ─────────────────────────────────────────

    def _decode_image_url(
        self, image_url: str, name_hint: str
    ) -> tuple[bytes, str, str]:
        image_url = image_url.strip()
        m = _DATA_URL_RE.match(image_url)
        if m:
            mime = (m.group(1) or "image/png").strip() or "image/png"
            payload = m.group(3)
            if m.group(2):
                content = base64.b64decode(payload)
            else:
                from urllib.parse import unquote_to_bytes

                content = unquote_to_bytes(payload)
            ext = mimetypes.guess_extension(mime) or ".png"
            filename = name_hint or f"image_{uuid.uuid4().hex[:8]}{ext}"
            if not filename.lower().endswith(ext):
                filename += ext
            return content, mime, filename
        if image_url.startswith(("http://", "https://")):
            content, mime = self._download(image_url)
            ext = mimetypes.guess_extension(mime) or ".png"
            name = name_hint or f"image_{uuid.uuid4().hex[:8]}{ext}"
            return content, mime, name
        raise ValueError(f"不支持的 image_url 形式: {image_url[:80]}")

    def _download(self, url: str) -> tuple[bytes, str]:
        """下载 http(s) 图片——走 client._conn 以对齐账号出口代理。"""
        conn = self._client._conn(url, 60)
        try:
            _, path = self._client._split(url)
            conn.request("GET", path, headers={"Accept": "image/*"})
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}")
            mime = (resp.getheader("Content-Type") or "image/png").split(";")[0]
            return raw, mime
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _upload_bytes(
        self, content: bytes, mime: str, filename: str
    ) -> Optional[dict]:
        client = self._client
        cfg = client.config
        boundary = f"----glmupload{uuid.uuid4().hex}"
        out = io_bytes_multipart(
            {
                "from": "chat",
                "assistant_id": cfg.assistant_id,
            },
            "file",
            filename,
            mime,
            content,
            boundary,
        )
        headers = client.upload_headers()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        url = f"{cfg.base_url}{CHAT_UPLOAD_PATH}"
        conn = client._conn(url, cfg.request_timeout)
        try:
            _, path = client._split(url)
            conn.request("POST", path, body=out, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                self._logger.warning(
                    "GlmUpload | HTTP %d: %s",
                    resp.status,
                    raw[:200].decode("utf-8", "replace"),
                )
                return None
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:  # noqa: BLE001
                self._logger.warning("GlmUpload | 响应不是 JSON: %r", raw[:120])
                return None
            if data.get("status") != 0:
                self._logger.warning(
                    "GlmUpload | 上传失败: %s", str(data.get("message"))[:200]
                )
                return None
            result = data.get("result")
            return result if isinstance(result, dict) else None
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("GlmUpload | 上传请求异常: %s", exc)
            return None
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def io_bytes_multipart(
    fields: dict, file_field: str, filename: str, mime: str, content: bytes, boundary: str
) -> bytes:
    """构造 multipart/form-data 请求体。"""
    out = bytearray()
    for k, v in fields.items():
        out += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
        ).encode("utf-8")
    out += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    out += content
    out += f"\r\n--{boundary}--\r\n".encode("utf-8")
    return bytes(out)
