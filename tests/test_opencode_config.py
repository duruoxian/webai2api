"""admin_api 里 opencode 配置改写相关的纯函数。

全部用 tmp_path / 纯字符串，不触碰用户真实的 ~/.config/opencode。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from webai2api import admin_api as api

SPAN = api._find_provider_span


def _provider_obj(text: str) -> dict:
    """把 provider 段（去掉注释后）解析出来，验证结果仍是合法 JSON 对象。"""
    span = SPAN(text)
    assert span is not None, text
    seg = api._strip_jsonc_comments(text[span[0]:span[1] + 1])
    return json.loads(seg)


def _whole_doc(text: str) -> dict:
    return json.loads(api._strip_jsonc_comments(text))


# ── _strip_jsonc_comments ──────────────────────────
def test_strip_line_and_block_comments():
    src = '{\n  // 行注释\n  "a": 1, /* 块\n注释 */ "b": 2\n}'
    out = api._strip_jsonc_comments(src)
    assert "//" not in out and "/*" not in out and "*/" not in out
    assert json.loads(out) == {"a": 1, "b": 2}


def test_strip_keeps_http_url_inside_string():
    src = '{"schema": "https://opencode.ai/config.json"}'
    assert api._strip_jsonc_comments(src) == src
    assert json.loads(api._strip_jsonc_comments(src))["schema"].startswith("https://")


def test_strip_keeps_provider_literal_inside_string():
    src = '{"note": "provider", "x": "// not a comment"}'
    assert api._strip_jsonc_comments(src) == src
    assert json.loads(api._strip_jsonc_comments(src))["x"] == "// not a comment"


def test_strip_respects_escaped_quote():
    src = '{"q": "a\\"//b"}'
    out = api._strip_jsonc_comments(src)
    assert out == src
    assert json.loads(out)["q"] == 'a"//b'


def test_strip_is_idempotent():
    src = ('{\n  // c1\n  "u": "http://x//y", /* c2 */\n'
           '  "provider": { "a": 1 } // tail\n}\n')
    once = api._strip_jsonc_comments(src)
    twice = api._strip_jsonc_comments(once)
    assert once == twice
    assert "c1" not in once and "c2" not in once and "tail" not in once
    assert json.loads(once)["provider"] == {"a": 1}


def test_strip_unterminated_block_comment():
    assert api._strip_jsonc_comments('{"a":1} /* never closed').strip() == '{"a":1}'


# ── _find_provider_span ────────────────────────────
def test_find_provider_span_basic():
    text = '{\n  "provider": {\n    "x": {}\n  },\n  "model": "a"\n}\n'
    span = SPAN(text)
    assert span is not None
    start, end = span
    assert text[start] == "{" and text[end] == "}"
    assert _provider_obj(text) == {"x": {}}


def test_find_provider_span_ignores_comment_and_string_literal():
    text = (
        '{\n'
        '  // "provider": { fake: 1 }\n'
        '  "note": "provider",\n'
        '  "provider": { "real": {} }\n'
        '}\n'
    )
    span = SPAN(text)
    assert span is not None
    assert _provider_obj(text) == {"real": {}}
    assert span[0] > text.index("// \"provider\"")


def test_find_provider_span_missing():
    assert SPAN('{"model": "x"}') is None
    assert SPAN('// only comment "provider": {\n') is None


def test_find_provider_span_inner_unbalanced_still_finds_provider():
    # provider 的 {} 自身配平（字符串/注释内的括号不计）
    assert SPAN('{"provider": {"a": {"b": 1}}}') == (13, 27)


def test_find_provider_span_unbalanced():
    assert SPAN('{"provider": { "a": 1') is None


# ── _provider_has_entry ────────────────────────────
def test_provider_has_entry():
    text = '{"provider": {"webai2api": {"npm": "x"}, "other": {}}}'
    assert api._provider_has_entry(text, "webai2api") is True
    assert api._provider_has_entry(text, "missing") is False


def test_provider_has_entry_not_fooled_by_comment_or_sibling():
    text = (
        '{"provider": {\n'
        '  // "webai2api": {}\n'
        '  "other": {"note": "webai2api"}\n'
        '}}'
    )
    assert api._provider_has_entry(text, "webai2api") is False


def test_provider_has_entry_without_provider_section():
    assert api._provider_has_entry('{"model": "x"}', "webai2api") is False


# ── disabled_providers ─────────────────────────────
def test_remove_disabled_does_not_touch_provider_block():
    text = (
        '{\n'
        '  "disabled_providers": ["webai2api", "foo"],\n'
        '  "provider": { "webai2api": { "npm": "x" } }\n'
        '}\n'
    )
    out = api._remove_disabled_provider(text, "webai2api")
    doc = _whole_doc(out)
    assert doc["disabled_providers"] == ["foo"]
    assert "webai2api" in doc["provider"]  # provider 里的块不能被动


def test_remove_disabled_last_element():
    text = '{"disabled_providers": ["a", "webai2api"], "x": 1}'
    doc = _whole_doc(api._remove_disabled_provider(text, "webai2api"))
    assert doc["disabled_providers"] == ["a"]
    assert doc["x"] == 1


def test_remove_disabled_only_element():
    text = '{"disabled_providers": ["webai2api"]}'
    doc = _whole_doc(api._remove_disabled_provider(text, "webai2api"))
    assert doc["disabled_providers"] == []


def test_remove_disabled_noop_when_absent():
    text = '{"disabled_providers": ["a"], "provider": {}}'
    assert api._remove_disabled_provider(text, "webai2api") == text


# ── _remove_provider_block ─────────────────────────
def test_remove_provider_block_between_siblings():
    text = (
        '{\n'
        '  "provider": {\n'
        '    "before": { "a": 1 },\n'
        '    "webai2api": { "npm": "x", "models": { "m": {"limit": {"context": 1}} } },\n'
        '    "after": { "b": 2 }\n'
        '  }\n'
        '}\n'
    )
    out = api._remove_provider_block(text, "webai2api")
    assert api._opencode_structure_ok(out)
    doc = _whole_doc(out)
    assert set(doc["provider"]) == {"before", "after"}
    assert doc["provider"]["after"] == {"b": 2}


def test_remove_provider_block_last_without_trailing_comma():
    text = '{\n  "provider": {\n    "a": {}, \n    "webai2api": { "x": 1 }\n  }\n}\n'
    out = api._remove_provider_block(text, "webai2api")
    doc = _whole_doc(out)
    assert doc["provider"] == {"a": {}}
    assert ",}" not in out and ", }" not in out


def test_remove_provider_block_was_last_inside_object():
    """webai2api 是 provider 里最后一项且前面有逗号：删掉后不能留下 , }。"""
    text = '{"provider": {"webai2api": {"a": 1}}}'
    out = api._remove_provider_block(text, "webai2api")
    assert _whole_doc(out) == {"provider": {}}


def test_remove_provider_block_only_entry():
    text = '{\n  "provider": {\n    "webai2api": { "npm": "x" }\n  }\n}\n'
    out = api._remove_provider_block(text, "webai2api")
    assert _whole_doc(out)["provider"] == {}


def test_remove_provider_block_string_values_with_provider_name():
    """文本值里出现 webai2api 不能被当成 provider 键删掉。"""
    text = (
        '{\n  "provider": {\n'
        '    "real": { "name": "webai2api", "note": "see webai2api docs" }\n'
        '  }\n}\n'
    )
    out = api._remove_provider_block(text, "webai2api")
    doc = _whole_doc(out)
    assert doc["provider"]["real"] == {"name": "webai2api", "note": "see webai2api docs"}


def test_remove_provider_block_inside_sibling_breaks_json(monkeypatch, tmp_path):
    """真实 bug 回归：另一个 provider 的 models 里若有同名键，期望只删顶层块。

    期望行为：只删 provider 顶层的 "webai2api": {...}，留下 real.models 里的同名键。
    走 _remove_opencode_provider 完整链路（含 _opencode_structure_ok 前置检查）。
    """
    p = _use_config(monkeypatch, tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text(
        '{\n  "provider": {\n'
        '    "real": { "models": { "webai2api": { "a": 1 } } },\n'
        '    "webai2api": { "npm": "x" },\n'
        '    "last": { "npm": "y" }\n'
        '  }\n}\n',
        encoding="utf-8",
    )
    api._remove_opencode_provider()
    doc = _whole_doc(p.read_text(encoding="utf-8"))
    assert "webai2api" not in doc["provider"]
    assert doc["provider"]["real"]["models"]["webai2api"] == {"a": 1}


def test_remove_provider_block_no_provider_section_returns_unchanged():
    text = '{\n  "model": "x"  // "provider": {}\n}\n'
    assert api._remove_provider_block(text, "webai2api") == text


def test_remove_provider_block_not_present_unchanged():
    text = '{"provider": {"other": {"a": 1}}}'
    assert api._remove_provider_block(text, "webai2api") == text


# ── _build_opencode_snippet ────────────────────────
def test_snippet_placeholder_is_parseable_object():
    snip = api._build_opencode_snippet("http://127.0.0.1:8802", "sk-123",
                                       ["deepseek-chat", "deepseek-reasoner"])
    obj = json.loads("{" + snip + "}")
    block = obj["webai2api"]
    assert block["npm"] == "@ai-sdk/openai-compatible"
    assert block["options"]["baseURL"] == "http://127.0.0.1:8802/v1"
    assert block["options"]["apiKey"] == "sk-123"
    assert set(block["models"]) == {"deepseek-chat", "deepseek-reasoner"}
    assert block["models"]["deepseek-reasoner"]["reasoning"] is True
    assert block["models"]["deepseek-chat"]["limit"] == {"context": 1000000, "output": 384000}


def test_snippet_glm_variants_and_limits():
    snip = api._build_opencode_snippet("http://127.0.0.1:8802", "k", ["GLM-5.3"])
    block = json.loads("{" + snip + "}")["webai2api"]
    m = block["models"]["GLM-5.3"]
    assert m["reasoning"] is True
    assert m["limit"] == {"context": 1000000, "output": 128000}
    assert set(m["variants"]) == {"快速", "深度", "极致"}


def test_snippet_model_id_with_quotes_is_safe():
    weird = 'my"model\\x'
    snip = api._build_opencode_snippet("http://127.0.0.1:8802", 'key"with\\quote', [weird])
    block = json.loads("{" + snip + "}")["webai2api"]
    assert list(block["models"]) == [weird]
    assert block["models"][weird]["name"] == weird
    assert block["options"]["apiKey"] == 'key"with\\quote'


# ── 与 _write_opencode_config 组合（仍只用 tmp_path） ──
def _use_config(monkeypatch, tmp_path: Path) -> Path:
    p = tmp_path / "opencode" / "opencode.jsonc"
    monkeypatch.setenv("WEBAI2API_OPENCODE_CONFIG", str(p))
    return p


def test_write_then_remove_roundtrip(monkeypatch, tmp_path):
    p = _use_config(monkeypatch, tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text(
        '{\n  "$schema": "https://opencode.ai/config.json",\n'
        '  // 用户自己的配置\n'
        '  "disabled_providers": ["webai2api"],\n'
        '  "provider": {\n    "other": { "npm": "@x" }\n  }\n}\n',
        encoding="utf-8",
    )
    snip = api._build_opencode_snippet("http://127.0.0.1:8802", "sk-ab", ["deepseek-chat"])
    res = api._write_opencode_config(snip)
    assert res["ok"] is True and Path(res["path"]) == p

    text = p.read_text(encoding="utf-8")
    assert api._opencode_structure_ok(text)
    doc = _whole_doc(text)
    assert "webai2api" in doc["provider"]
    assert doc["provider"]["other"] == {"npm": "@x"}
    assert doc["disabled_providers"] == []
    assert doc["$schema"].startswith("https://")

    # 再写一次：幂等，不应出现重复块
    api._write_opencode_config(snip)
    doc2 = _whole_doc(p.read_text(encoding="utf-8"))
    assert len([k for k in doc2["provider"] if k == "webai2api"]) == 1

    res = api._remove_opencode_provider()
    assert res["ok"] is True
    doc3 = _whole_doc(p.read_text(encoding="utf-8"))
    assert "webai2api" not in doc3["provider"]
    assert doc3["provider"]["other"] == {"npm": "@x"}


def test_write_creates_config_when_absent(monkeypatch, tmp_path):
    """配置不存在时先建模板再写入，产出必须是合法 JSONC。"""
    p = _use_config(monkeypatch, tmp_path)
    snip = api._build_opencode_snippet("http://127.0.0.1:8802", "k", ["deepseek-chat"])
    api._write_opencode_config(snip)
    doc = _whole_doc(p.read_text(encoding="utf-8"))
    assert "webai2api" in doc["provider"]


def test_write_creates_provider_when_missing(monkeypatch, tmp_path):
    """没有 provider 段时自动创建（一键导入的关键）。"""
    p = _use_config(monkeypatch, tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text('{\n  "model": "x"\n}\n', encoding="utf-8")
    api._write_opencode_config('"webai2api": {}')
    doc = _whole_doc(p.read_text(encoding="utf-8"))
    assert "webai2api" in doc["provider"]
    assert doc["model"] == "x"


def test_remove_is_idempotent_when_absent(monkeypatch, tmp_path):
    p = _use_config(monkeypatch, tmp_path)
    res = api._remove_opencode_provider()
    assert res["ok"] is True and res["not_present"] is True
    assert not p.exists()  # 不应为了"断开"去创建文件


def test_remove_detects_broken_jsonc(monkeypatch, tmp_path):
    """JSONC 已损坏（括号不配对）时，移除操作必须拒绝而不是越改越坏。"""
    p = _use_config(monkeypatch, tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text('{\n  "provider": {\n    "webai2api": {"a": 1},\n    "z": {"b": 2\n  }\n}\n',
                 encoding="utf-8")
    with pytest.raises(RuntimeError):
        api._remove_opencode_provider()


def test_remove_truncated_jsonc_is_never_destroyed(monkeypatch, tmp_path):
    """文件被截断（最后的 } 丢失）时，处理完不能把整份配置变成空文件。

    这是个弱断言：只要求结果仍是可解析 JSON 或原样未动。
    """
    p = _use_config(monkeypatch, tmp_path)
    p.parent.mkdir(parents=True)
    original = '{"provider": {"webai2api": {"a": 1}'
    p.write_text(original, encoding="utf-8")
    try:
        res = api._remove_opencode_provider()
        assert res["ok"] is True
    except RuntimeError:
        return
    after = p.read_text(encoding="utf-8")
    assert after in (original, "{}", '{"provider": {}}') or _parses(after)


def _parses(text: str) -> bool:
    try:
        _whole_doc(text)
        return True
    except Exception:
        return False





# ── _opencode_status_io ────────────────────────────
def test_status_io_missing_file(monkeypatch, tmp_path):
    p = _use_config(monkeypatch, tmp_path)
    info = api._opencode_status_io()
    assert info["config_path"] == str(p)
    assert info["has_provider"] is False
    assert info["installed"] == bool(__import__("shutil").which("opencode"))


def test_status_io_with_provider(monkeypatch, tmp_path):
    p = _use_config(monkeypatch, tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text('{"provider": {"webai2api": {}}}', encoding="utf-8")
    info = api._opencode_status_io()
    assert info["installed"] is True
    assert info["has_provider"] is True


def test_status_io_provider_only_in_comment(monkeypatch, tmp_path):
    p = _use_config(monkeypatch, tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text('{\n  // "provider": {"webai2api": {}}\n  "a": 1\n}', encoding="utf-8")
    assert api._opencode_status_io()["has_provider"] is False


def test_config_path_honours_env(monkeypatch, tmp_path):
    p = _use_config(monkeypatch, tmp_path)
    assert api._opencode_config_path() == p
    monkeypatch.delenv("WEBAI2API_OPENCODE_CONFIG")
    assert api._opencode_config_path() == Path.home() / ".config" / "opencode" / "opencode.jsonc"
