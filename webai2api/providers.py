"""多 provider 注册表：把 deepseek / chatglm 的账号池统一起来，按模型路由。"""
from __future__ import annotations

from logging import Logger

from .adapters.base import AccountClient, GenericClientPool
from .adapters.deepseek.pool import build_pool as build_deepseek_pool
from .adapters.chatglm.pool import build_pool as build_chatglm_pool
from .store import Store

DEFAULT_MODEL = "deepseek-chat"

# 各 provider 的推荐额度 + 硬上限（面板可改，但不能超过上限）
DEFAULT_LIMITS: dict[str, dict[str, int]] = {
    "deepseek": {"context": 1_000_000, "output": 384_000},
    "chatglm": {"context": 1_000_000, "output": 128_000},
}
MAX_LIMITS: dict[str, dict[str, int]] = {
    "deepseek": {"context": 1_000_000, "output": 384_000},
    "chatglm": {"context": 1_000_000, "output": 128_000},
}

# 每个 provider 的默认模型清单（账号侧没配置时的兜底）
PROVIDER_MODELS: dict[str, list[str]] = {
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "chatglm": ["GLM-5.3", "GLM-5.3-Flash"],
}

# 兼容旧引用
MODELS = [
    {"id": mid, "object": "model", "owned_by": f"{prov}-web"}
    for prov, ids in PROVIDER_MODELS.items()
    for mid in ids
]


class ProviderRegistry:
    def __init__(self, store: Store, logger: Logger) -> None:
        self.store = store
        self.logger = logger
        self._auth_fails: dict[str, int] = {}
        self.deepseek = build_deepseek_pool(store, logger)
        self.chatglm = build_chatglm_pool(store, logger)
        self._pools: dict[str, GenericClientPool] = {
            "deepseek": self.deepseek,
            "chatglm": self.chatglm,
        }

    def model_catalog(self) -> list[dict]:
        """动态模型清单：默认清单 ∪ 账号配置里出现过的模型。

        以后新增模型只需在账号/设置里配置，无需改代码。
        """
        seen: set[str] = set()
        out: list[dict] = []
        # 1) 默认清单，但有可用账号的 provider 才展示
        active = {p for p in self._pools if self.store.list_accounts(p)}
        active = active or set(self._pools)
        for prov in active:
            for mid in PROVIDER_MODELS.get(prov, []):
                if mid not in seen:
                    seen.add(mid)
                    out.append({"id": mid, "object": "model", "owned_by": f"{prov}-web"})
        # 2) 用量里实际出现过的模型（说明真的在用）
        try:
            for row in self.store.list_used_models(limit=100):
                mid = (row.get("model") or "").strip()
                if len(mid) >= 3 and mid not in seen:
                    seen.add(mid)
                    prov = "chatglm" if mid.lower().startswith("glm") else "deepseek"
                    out.append({"id": mid, "object": "model", "owned_by": f"{prov}-web"})
        except Exception:
            pass
        return out or MODELS

    def pool_for_model(self, model: str | None) -> GenericClientPool:
        m = (model or "").strip().lower()
        if m.startswith("glm") or (len(m) >= 24 and m.isalnum()):
            return self.chatglm
        return self.deepseek

    def pick(self, model: str | None, exclude: set[str] | None = None,
             allowed: set[str] | None = None) -> AccountClient | None:
        return self.pool_for_model(model).pick(exclude, allowed=allowed)

    def note_auth_failure(self, account_id: str) -> bool:
        """连续两次鉴权失败 → 自动停用账号，避免继续浪费请求。"""
        fails = self._auth_fails.get(account_id, 0) + 1
        self._auth_fails[account_id] = fails
        if fails >= 2:
            try:
                self.store.update_account(account_id, enabled=0, health="expired")
            except Exception:
                pass
            self.logger.warning("账号 %s 连续鉴权失败，已自动停用", account_id)
            self.refresh()
            return True
        return False

    def note_auth_ok(self, account_id: str) -> None:
        self._auth_fails.pop(account_id, None)

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
