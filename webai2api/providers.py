"""多 provider 注册表：把 deepseek / chatglm 的账号池统一起来，按模型路由。"""
from __future__ import annotations

from logging import Logger

from .adapters.base import AccountClient, GenericClientPool
from .adapters.deepseek.pool import build_pool as build_deepseek_pool
from .adapters.chatglm.pool import build_pool as build_chatglm_pool
from .adapters.kimi.pool import build_pool as build_kimi_pool
from .adapters.qwen.pool import build_pool as build_qwen_pool
from .store import Store

DEFAULT_MODEL = "deepseek-chat"

# 各 provider 的推荐额度 + 硬上限（面板可改，但不能超过上限）
DEFAULT_LIMITS: dict[str, dict[str, int]] = {
    "deepseek": {"context": 1_000_000, "output": 384_000},
    "chatglm": {"context": 1_000_000, "output": 128_000},
    "kimi": {"context": 262_144, "output": 32_768},
    "qwen": {"context": 262_144, "output": 65_536},
}
MAX_LIMITS: dict[str, dict[str, int]] = {
    "deepseek": {"context": 1_000_000, "output": 384_000},
    "chatglm": {"context": 1_000_000, "output": 128_000},
    "kimi": {"context": 1_000_000, "output": 65_536},
    "qwen": {"context": 1_000_000, "output": 65_536},
}

# 每个 provider 的默认模型清单（账号侧没配置时的兜底）
PROVIDER_MODELS: dict[str, list[str]] = {
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "chatglm": ["GLM-5.3", "GLM-5.3-Flash"],
    "kimi": ["kimi-k3", "kimi-k2.8", "kimi-k2.6"],
    "qwen": ["qwen3.7-plus", "qwen3.8-max", "qwen3.8-omni-flash"],
}

# 模型名前缀 → provider（全仓唯一权威表；admin_api / 面板兜底都从这里取值）
MODEL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("glm", "chatglm"),
    ("kimi", "kimi"),
    ("moonshot", "kimi"),
    ("qwen", "qwen"),
    ("deepseek", "deepseek"),
)

DEFAULT_PROVIDER = "deepseek"

# 对外可用的平台：DeepSeek / 智谱 / Kimi / 千问。
# Kimi 走网页版 Connect 协议；千问走每个账号一个隐藏 WebView2 中继（绕阿里风控）。
PUBLIC_PROVIDERS: tuple[str, ...] = ("deepseek", "chatglm", "kimi", "qwen")


def infer_provider(model: str | None) -> str:
    m = (model or "").strip().lower()
    for prefix, prov in MODEL_PREFIXES:
        if m.startswith(prefix):
            return prov
    return DEFAULT_PROVIDER

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
        self.kimi = build_kimi_pool(store, logger)
        self.qwen = build_qwen_pool(store, logger)
        self._pools: dict[str, GenericClientPool] = {
            "deepseek": self.deepseek,
            "chatglm": self.chatglm,
            "kimi": self.kimi,
            "qwen": self.qwen,
        }

    def model_catalog(self) -> list[dict]:
        """动态模型清单：默认清单 ∪ 账号配置里出现过的模型。

        以后新增模型只需在账号/设置里配置，无需改代码。
        """
        seen: set[str] = set()
        out: list[dict] = []
        # 1) 默认清单，但有可用账号的 provider 才展示
        active = {p for p in self._pools if self.store.list_accounts(p)}
        active = active or set(PUBLIC_PROVIDERS)
        active &= set(PUBLIC_PROVIDERS)
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
                    prov = infer_provider(mid)
                    out.append({"id": mid, "object": "model", "owned_by": f"{prov}-web"})
        except Exception:
            pass
        return out or MODELS

    def pool_for_model(self, model: str | None) -> GenericClientPool:
        m = (model or "").strip().lower()
        # 24 位以上纯小写字母数字 = 智谱智能体 ID
        if len(m) >= 24 and m.isalnum():
            return self.chatglm
        return self._pools.get(infer_provider(m), self.deepseek)

    def pick(self, model: str | None, exclude: set[str] | None = None,
             allowed: set[str] | None = None,
             prefer: str | None = None) -> AccountClient | None:
        return self.pool_for_model(model).pick(exclude, allowed=allowed, prefer=prefer)

    def cooldowns(self) -> dict[str, int]:
        """各账号剩余冷却秒数（跨 provider 合并）。"""
        out: dict[str, int] = {}
        for pool in self._pools.values():
            try:
                out.update(pool.cooldowns())
            except Exception:  # noqa: BLE001
                pass
        return out

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

    def refresh(self, provider: str | None = None) -> None:
        """刷新账号池。指定 provider 时只重建该平台的池（增删账号只需一个）。"""
        pools = ([self._pools[provider]] if provider in self._pools
                 else self._pools.values())
        for pool in pools:
            try:
                pool.refresh()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("刷新账号池失败: %s", exc)

    def available(self) -> list[AccountClient]:
        out: list[AccountClient] = []
        for pool in self._pools.values():
            out.extend(pool.available())
        return out

    def all_accounts(self) -> list[AccountClient]:
        """跨 provider 汇总所有账号客户端（不过滤健康/冷却，供后台清理线程用）。"""
        out: list[AccountClient] = []
        for pool in self._pools.values():
            try:
                out.extend(pool.available())
            except Exception:  # noqa: BLE001
                pass
        return out
