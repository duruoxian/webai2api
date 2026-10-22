"""通义千问账号客户端池。"""
from __future__ import annotations

from logging import Logger

from ..base import GenericClientPool
from ...store import Store
from . import config as qwen_config
from .client import QwenWebClient

PROVIDER = "qwen"


def build_config(store: Store, account: dict) -> qwen_config.QwenConfig:
    return qwen_config.build_config(store, account)


def build_pool(store: Store, logger: Logger) -> GenericClientPool:
    return GenericClientPool(
        store, PROVIDER, logger,
        build_config=build_config,
        client_factory=lambda cfg, lg: QwenWebClient(cfg, lg),
    )
