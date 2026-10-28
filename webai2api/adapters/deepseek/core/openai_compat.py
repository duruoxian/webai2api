"""OpenAI 兼容字段的生成器（chatcmpl id / system fingerprint）。"""

from __future__ import annotations

import hashlib
import os
import platform
import time
import uuid

_FINGERPRINT_SALT = f"{platform.python_version()}-{platform.system()}-{os.getpid()}"
_FINGERPRINT_HASH = hashlib.md5(_FINGERPRINT_SALT.encode("utf-8")).hexdigest()[:6]


def gen_chatcmpl_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def system_fingerprint(model: str = "") -> str:
    if model:
        date_str = time.strftime("%Y%m%d")
        base = hashlib.md5(f"{model}:{date_str}".encode("utf-8")).hexdigest()[:8]
        return f"fp_{base}"
    return f"fp_{_FINGERPRINT_HASH}"
