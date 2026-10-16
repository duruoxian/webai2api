"""智谱清言账号客户端池。"""
from __future__ import annotations

from logging import Logger

from ..base import GenericClientPool
from ...store import Store
from . import config as glm_config
from .client import ChatGlmWebClient

PROVIDER = "chatglm"


def build_config(store: Store, account: dict) -> glm_config.GlmConfig:
    return glm_config.build_config(store, account)


def build_pool(store: Store, logger: Logger) -> GenericClientPool:
    return GenericClientPool(
        store, PROVIDER, logger,
        build_config=build_config,
        client_factory=lambda cfg, lg: ChatGlmWebClient(cfg, lg),
    )
