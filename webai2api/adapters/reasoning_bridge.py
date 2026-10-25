"""网页版模型的 reasoning 兜底桥接（Kimi / 智谱清言 / 通义千问共用）。

网页版经常把最终答案写进思考块（Kimi 全系实测：正文空白、思考块里
是完整答案），客户端拿到 content 为空就等于白问一次。这里统一做两件事：

1. 空正文兜底（始终启用）：流结束时 content 为空或仅空白而 reasoning
   非空 → 把 reasoning 转成 content 下发，同时清空 reasoning_content
   （非流式直接置换；流式思考增量已发出不可撤回，只能在结尾补一发
   content 增量，保证最终正文非空）。

2. LEGACY_REASONING_IN_CONTENT 开关（默认 false，命名参考 Qwen2API）：
   true 时思考内容以 <think>...</think> 包进 content 流、不再单独发
   reasoning_content——给不认识 reasoning_content 字段的旧客户端用。
   默认 false 保持现有行为（reasoning 走独立字段）。
"""
from __future__ import annotations

__all__ = [
    "finalize_reasoning",
    "ReasoningStream",
]


def finalize_reasoning(content: str, reasoning: str, *,
                       legacy: bool = False) -> tuple[str, str]:
    """非流式收尾：返回 (content, reasoning) 的最终形态。

    - legacy=True：思考以 <think> 块并入 content，reasoning 清空；
    - 否则保持两字段原样，仅当 content 空白而 reasoning 非空时兜底置换。
    """
    content = content or ""
    reasoning = reasoning or ""
    if legacy and reasoning.strip():
        merged = f"<think>\n{reasoning}\n</think>\n\n{content}".strip()
        return merged, ""
    if not content.strip() and reasoning.strip():
        return reasoning.strip(), ""
    return content, reasoning


class ReasoningStream:
    """流式增量分发器：统一处理思考增量的两种去向。

    正常模式：思考增量走 reasoning_content，正文增量走 content，
    与改造前行为一致；收尾时若正文全程为空而思考非空，补发一发
    content 增量（思考已流出不可撤回，但保证最终正文非空）。

    legacy 模式：思考以 <think>...</think> 包进 content 流——首段思考
    前发 "<think>\\n"，末尾由 finish() 闭合 "</think>"，顺序保证
    思考块永远在正文前面。
    """

    def __init__(self, legacy: bool = False) -> None:
        self._legacy = legacy
        self._think_open = False
        self._think_closed = False
        self._reasoning_total = ""
        self._content_total = ""

    def feed(self, d_reasoning: str, d_content: str) -> tuple[str, str]:
        """喂入本事件的思考/正文增量，返回 (reasoning增量, content增量)。"""
        out_r = ""
        out_c = ""
        if d_reasoning:
            self._reasoning_total += d_reasoning
            if self._legacy:
                if not self._think_open:
                    self._think_open = True
                    out_c += "<think>\n"
                out_c += d_reasoning
            else:
                out_r = d_reasoning
        if d_content:
            self._content_total += d_content
            if self._legacy and self._think_open and not self._think_closed:
                self._think_closed = True
                out_c += "\n</think>\n"
            out_c += d_content
        return out_r, out_c

    def close_think(self) -> str:
        """只闭合未闭的 </think>（工具调用收尾用），不做空正文兜底。

        工具调用路径 content 为空是正常形态，不能触发兜底。
        """
        if self._legacy and self._think_open and not self._think_closed:
            self._think_closed = True
            return "\n</think>\n"
        return ""

    def finish(self) -> tuple[str, str]:
        """收尾：闭合 <think>；空正文兜底把思考补发为 content。

        返回 (补发的 reasoning增量, 补发的 content增量)。
        """
        out_c = self.close_think()
        if not out_c and not self._legacy \
                and not self._content_total.strip() and self._reasoning_total.strip():
            # 正文空白而思考非空：思考里就是答案，转成正文补发
            out_c += self._reasoning_total
        return "", out_c
