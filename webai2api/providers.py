"""多 provider 注册表：把 deepseek / chatglm 的账号池统一起来，按模型路由。"""
from __future__ import annotations

from logging import Logger

from .adapters.base import AccountClient, GenericClientPool
from .adapters.deepseek.pool import build_pool as build_deepseek_pool
from .adapters.chatglm.pool import build_pool as build_chatglm_pool
from .store import Store

# 对外暴露的模型清单
MODELS = [
    {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek-web"},
    {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek-web"},
    {"id": "GLM-5.3", "object": "model", "owned_by": "chatglm-web"},
    {"id": "GLM-5.3-Flash", "object": "model", "owned_by": "chatglm-web"},
]

DEFAULT_MODEL = "deepseek-chat"


class ProviderRegistry:
    def __init__(self, store: Store, logger: Logger) -> None:
        self.store = store
        self.logger = logger
        self.deepseek = build_deepseek_pool(store, logger)
        self.chatglm = build_chatglm_pool(store, logger)
        self._pools: dict[str, GenericClientPool] = {
            "deepseek": self.deepseek,
            "chatglm": self.chatglm,
        }

    def pool_for_model(self, model: str | None) -> GenericClientPool:
        m = (model or "").strip().lower()
        if m.startswith("glm") or (len(m) >= 24 and m.isalnum()):
            return self.chatglm
        return self.deepseek

    def pick(self, model: str | None, exclude: set[str] | None = None) -> AccountClient | None:
        return self.pool_for_model(model).pick(exclude)

    def pick_by_id(self, account_id: str) -> AccountClient | None:
        for pool in self._pools.values():
            ac = pool.pick_by_id(account_id)
            if ac:
                return ac
        return None

    def refresh(self) -> None:
        for pool in self._pools.values():
            try:
                pool.refresh()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("刷新账号池失败: %s", exc)

    def available(self) -> list[AccountClient]:
        out: list[AccountClient] = []
        for pool in self._pools.values():
            out.extend(pool.available())
        return out
