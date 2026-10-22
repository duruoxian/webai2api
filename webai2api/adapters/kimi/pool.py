"""Kimi 账号客户端池。"""
from __future__ import annotations

from logging import Logger

from ..base import GenericClientPool
from ...store import Store
from . import config as kimi_config
from .client import KimiWebClient

PROVIDER = "kimi"


def build_config(store: Store, account: dict) -> kimi_config.KimiConfig:
    return kimi_config.build_config(store, account)


def build_pool(store: Store, logger: Logger) -> GenericClientPool:
    return GenericClientPool(
        store, PROVIDER, logger,
        build_config=build_config,
        client_factory=lambda cfg, lg: KimiWebClient(cfg, lg),
    )
