from __future__ import annotations

import logging
from typing import Any


def serialize_for_debug(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, str):
        return value
    try:
        import json
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return repr(value)


def debug_dump(logger: logging.Logger, enabled: bool, title: str, value: Any) -> None:
    if not enabled:
        return
    logger.debug("%s\n%s", title, serialize_for_debug(value))