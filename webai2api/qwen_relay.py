"""通义千问浏览器会话中继。

阿里风控会拦截纯 HTTP 请求，但**在已登录的网页里**发 fetch 不受影响
（baxia SDK 自动补 bx-ua / bx-umidtoken 等头）。因此每个千问账号挂一个
隐藏的 WebView2 窗口，由它执行 fetch，再通过本机 HTTP 把 SSE 转给主程序。

子进程入口：<exe> --relay-qwen <account_id>
状态文件：data/relay_state/<account_id>.json
"""
from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .paths import user_data_dir
from .store import Store

_QWEN_URL = "https://chat.qwen.ai/"


def _jwt_exp(token: str) -> int:
    """解析 JWT 的 exp（秒）；失败返回 0。"""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        return int(data.get("exp") or 0)
    except Exception:
        return 0


def _state_dir() -> Path:
    d = user_data_dir() / "relay_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path(acc_id: str) -> Path:
    return _state_dir() / f"{acc_id}.json"


def read_state(acc_id: str) -> dict:
    try:
        return json.loads(state_path(acc_id).read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(acc_id: str, data: dict) -> None:
    p = state_path(acc_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def clear_state(acc_id: str) -> None:
    try:
        state_path(acc_id).unlink()
    except Exception:
        pass


# ── 注入到千问页面的 JS ─────────────────────────────────────

_RELAY_JS = r"""
(function(){
  if(window.__qwenRelayStart) return 'already';

  function readToken(){
    try{
      const raw = localStorage.getItem('qwen_access_token_state');
      if(raw){ const o = JSON.parse(raw); const s = (o && o.state) ? o.state : o;
        if(s && typeof s.token === 'string' && s.token) return s.token; }
    }catch(e){}
    for(let i=0;i<localStorage.length;i++){
      const k = localStorage.key(i), v = localStorage.getItem(k);
      if(!v) continue;
      if(k === 'token' && v.split('.').length === 3) return v;
      try{ const o = JSON.parse(v); const s = (o && o.state) ? o.state : o;
        if(s && typeof s.token === 'string' && s.token.split('.').length === 3) return s.token; }catch(e){}
    }
    return '';
  }

  function uuid(){
    if(crypto.randomUUID) return crypto.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c){
      const r = Math.random()*16|0, v = c==='x' ? r : (r&0x3|0x8);
      return v.toString(16);
    });
  }

  function jwtExp(t){
    try{
      const p = t.split('.')[1].replace(/-/g,'+').replace(/_/g,'/');
      return JSON.parse(atob(p)).exp || 0;
    }catch(e){ return 0; }
  }

  // accessToken 只有 15 分钟。页面**加载时** SPA 会自己换一张新票；隐藏窗口的
  // 定时器会被浏览器节流，所以过期时最可靠的自愈手段是「重载页面」让 SPA 重新
  // 换票（由 Python worker 统一执行并重新注入本脚本）。
  function tokenFresh(t){ return jwtExp(t) > Math.floor(Date.now()/1000) + 30; }
  function looksAuthExpired(code, msg){
    const c = String(code || '').toUpperCase();
    const m = String(msg || '');
    return c === 'UNAUTHORIZED' || c === 'TOKEN_INVALID' || c === 'AUTH_FAILED'
      || /token has expired|please log in again|未授权|登录.*(过期|失效)|401/.test(m);
  }
  function bodyOf(req, chatId, parentId){
    const now = Math.floor(Date.now()/1000);
    const model = req.model || 'qwen3.7-plus';
    const prompt = (req.system ? (req.system + '\n\n') : '') + (req.prompt || '');
    const feature = {
      thinking_enabled: !!req.thinking_enabled,
      output_schema: 'phase',
      research_mode: 'normal',
      auto_thinking: !!req.auto_thinking,
      thinking_mode: req.thinking_mode || (req.auto_thinking ? 'Auto' : (req.thinking_enabled ? 'Thinking' : 'Fast')),
      thinking_format: 'summary',
      auto_search: !!req.auto_search,
    };
    const msg = {
      id: null,
      fid: uuid(),
      parentId: parentId || null,
      childrenIds: [uuid()],
      role: 'user',
      content: prompt,
      user_action: 'chat',
      files: [],
      timestamp: now,
      models: [model],
      model: '',
      chat_type: 't2t',
      sub_chat_type: 't2t',
      feature_config: feature,
      extra: {meta: {subChatType: 't2t'}},
      parent_id: parentId || null,
    };
    return {
      stream: true,
      version: '2.1',
      incremental_output: true,
      chatId: chatId,
      parentId: parentId || '',
      chat_id: chatId,
      chat_mode: 'normal',
      model: model,
      parent_id: parentId || null,
      messages: [msg],
      timestamp: now,
    };
  }

  window.__qwenRelayStart = async function(reqId, req){
    const push = async function(kind, data){
      try{
        await window.pywebview.api.relay_push(reqId, kind, JSON.stringify(data || {}));
      }catch(e){}
    };
    try{
      const token = readToken();
      if(!token){ await push('error', {message: '未登录通义千问，请在中继窗口里完成登录'}); return; }
      if(!tokenFresh(token)){
        // 已过期：只上报，由 Python worker 统一整页重载换票
        await push('error', {message: '登录态即将到期，已自动重连，请稍后重试', reload: true});
        return;
      }
      const model = req.model || 'qwen3.7-plus';
      const base = (req.base_url || '').replace(/\/+$/,'');

      function httpHeaders(extra){
        return Object.assign({
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'X-Accel-Buffering': 'no',
          'X-Request-Id': uuid(),
          'Version': '0.3.22',
          'source': 'web',
          'Timezone': new Date().toString().replace(/[^\x20-\x7E]/g, ''),
          'Authorization': 'Bearer ' + token,
        }, extra || {});
      }

      async function createChat(){
        const r = await fetch(base + '/api/v2/chats/new', {
          method: 'POST',
          credentials: 'include',
          headers: httpHeaders({'Accept': 'application/json, text/plain, */*'}),
          body: JSON.stringify({
            chatId: '', models: [model], project_id: '',
            timestamp: Date.now(), chat_type: 't2t', chat_mode: 'normal',
          }),
        });
        let text = '';
        try{ text = await r.text(); }catch(e){}
        let j = null;
        try{ j = JSON.parse(text); }catch(e){}
        if(!j || j.success === false){
          const d = (j && j.data) || {};
          throw new Error((d && (d.details || d.code || d.message)) || text || ('HTTP ' + r.status));
        }
        const d = j.data || {};
        if(!d.id) throw new Error('创建对话失败：缺少 id');
        return d.id;
      }

      let chatId = req.chat_id || '';
      let parentId = req.parent_id || '';
      if(!chatId){
        chatId = await createChat();
        parentId = '';
      }
      await push('meta', {chat_id: chatId, response_id: '', parent_id: parentId});

      let resp = null;
      for(let attempt = 0; attempt < 2; attempt++){
        const body = bodyOf(req, chatId, parentId);
        const url = base + '/api/v2/chat/completions?chat_id=' + encodeURIComponent(chatId);
        if(req.debug) await push('debug', {body: body, url: url, chat_id: chatId});
        try{
          resp = await fetch(url, {
            method: 'POST', credentials: 'include',
            headers: httpHeaders(), body: JSON.stringify(body),
          });
        }catch(e){
          await push('error', {message: '页面 fetch 失败: ' + String(e)});
          return;
        }
        const ctype = resp.headers.get('content-type') || '';
        if(ctype.includes('text/event-stream')) break;
        let text = '';
        try{ text = await resp.text(); }catch(e){}
        let code = '';
        let msg = text || ('HTTP ' + resp.status);
        try{
          const j = JSON.parse(text);
          const d = (j && j.data) || {};
          code = String(d.code || '');
          msg = (d && (d.details || d.code || d.message)) || (j && (j.detail || j.message)) || msg;
        }catch(e){}
        if(code === 'CHAT_NOT_FOUND' && attempt === 0){
          chatId = await createChat();
          parentId = '';
          await push('meta', {chat_id: chatId, response_id: '', parent_id: ''});
          continue;
        }
        if(looksAuthExpired(code, msg)){
          await push('error', {message: '登录态已过期，正在自动重连，请稍后重试', reload: true});
          return;
        }
        await push('error', {message: String(msg).slice(0, 500), status: resp.status,
                            body: String(text || '').slice(0, 1200)});
        return;
      }
      let responseId = '';
      let primaryResponseId = '';
      let answerChars = 0;
      let sawFinish = false;
      const seenReasoning = {};
      // ── 多路候选回答过滤（移植自 Rfym21/Qwen2API）─────────────
      // 上游偶尔对同一次请求开多路候选回答：多个 response.created
      // （chat_id/parent_id 相同、response_index 0/1、response_id 不同），
      // 随后两路增量帧交错到达。不区分 response_id 一律拼接，两路内容
      // 就会搅在一起——qwen3.8-max「语无伦次」的根源（官方网页端只展示
      // index=0 的主回答）。旧协议无 created 元数据时锁定第一路带
      // response_id 的正文；完全不带 response_id 时一律放行。
      let acceptedResponseId = null;
      let sawCreatedEvent = false;
      const responseIndexes = {};
      function acceptEvent(ev){
        const created = ev['response.created'] || (ev.response && ev.response.created);
        if(created && typeof created === 'object'){
          sawCreatedEvent = true;
          const rid = created.response_id || created.responseId || null;
          const ridx = Number(created.response_index != null ? created.response_index : created.responseIndex);
          if(rid && Number.isFinite(ridx)) responseIndexes[rid] = ridx;
          if(rid && ridx === 0) acceptedResponseId = rid;
          else if(rid && acceptedResponseId === null && !Number.isFinite(ridx)) acceptedResponseId = rid;
          return !rid || acceptedResponseId === null || rid === acceptedResponseId;
        }
        const rid = ev.response_id;
        if(!rid) return true;
        if(acceptedResponseId !== null) return rid === acceptedResponseId;
        const ridx = responseIndexes[rid];
        if(ridx === 0){ acceptedResponseId = rid; return true; }
        if(Number.isFinite(ridx) || sawCreatedEvent) return false;
        acceptedResponseId = rid;
        return true;
      }
      if(!resp.body || !resp.body.getReader){
        await push('error', {message: '当前浏览器不支持流式读取'}); return;
      }
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      let finished = false;
      while(!finished){
        const chunk = await reader.read();
        if(chunk.value) buf += dec.decode(chunk.value, {stream: true});
        let idx;
        while((idx = buf.indexOf('\n')) >= 0){
          let line = buf.slice(0, idx); buf = buf.slice(idx + 1);
          line = line.replace(/\r$/, '').trim();
          if(!line.startsWith('data:')) continue;
          const raw = line.slice(5).trim();
          if(!raw || raw === '[DONE]') continue;
          let ev;
          try{ ev = JSON.parse(raw); }catch(e){ continue; }
          // 多路候选过滤：非主回答流的帧整条丢弃（错误帧不带 response_id，不受影响）
          if(!acceptEvent(ev)) continue;
          if(ev.error || ev.success === false){
            const d = (ev && ev.data) || {};
            const code = d.code || '';
            const msg = d.details || d.code || d.message || ev.message || ev.error || raw;
            if(looksAuthExpired(code, msg)){
              await push('error', {message: '登录态已过期，正在自动重连，请稍后重试', reload: true});
              return;
            }
            await push('error', {message: String(msg).slice(0, 500), body: raw.slice(0, 1200)});
            return;
          }
          if(ev['response.created']){
            const c = ev['response.created'];
            if(!primaryResponseId) primaryResponseId = c.response_id || '';
            responseId = primaryResponseId || c.response_id || responseId;
            await push('meta', {chat_id: c.chat_id || chatId, response_id: responseId, parent_id: c.parent_id || parentId});
            continue;
          }
          if(ev.response_id){
            if(!primaryResponseId) primaryResponseId = ev.response_id;
            responseId = responseId || primaryResponseId;
          }
          // 多路候选已在 acceptEvent 里按 response_index=0 锁定主回答，
          // 走到这里的事件都属于主回答流（思考/搜索在 phase 字段里区分，
          // 见下方 thinking_summary / answer 分支）。
          const choices = ev.choices || [];
          for(const ch of choices){
            const d = ch.delta || {};
            const phase = d.phase || '';
            if(phase === 'thinking_summary'){
              const extra = d.extra || {};
              const thought = (extra.summary_thought && extra.summary_thought.content) || [];
              const title = (extra.summary_title && extra.summary_title.content) || [];
              for(const t of title.concat(thought)){
                const s = String(t);
                if(seenReasoning[s]) continue;
                seenReasoning[s] = true;
                await push('reasoning', {content: s});
              }
            }else if(phase === 'answer' || phase === '' || !phase){
              let piece = '';
              if(typeof d.content === 'string') piece = d.content;
              else if(Array.isArray(d.content)) piece = d.content.map(x => typeof x === 'string' ? x : (x && x.text) || '').join('');
              else if(d.content && typeof d.content.text === 'string') piece = d.content.text;
              if(piece){
                answerChars += piece.length;
                await push('delta', {content: piece});
              }
            }
            if(typeof d.reasoning_content === 'string' && d.reasoning_content){
              await push('reasoning', {content: d.reasoning_content});
            }
            if(ch.finish_reason){
              sawFinish = true;
              // 只有确实拿到过正文才提前收尾；否则继续读后续 response
              if(answerChars > 0) finished = true;
            }
          }
        }
        if(chunk.done) finished = true;
      }
      if(answerChars === 0){
        await push('error', {
          message: sawFinish ? '模型返回了空内容，请换个档位或重试' : '模型未返回任何内容，请重试',
          empty: true,
        });
        return;
      }
      await push('done', {chat_id: chatId, response_id: responseId});
    }catch(e){
      await push('error', {message: String(e)});
    }
  };
  window.__qwenRelayDelete = async function(reqId, chatId){
    const push = async function(kind, data){
      try{ await window.pywebview.api.relay_push(reqId, kind, JSON.stringify(data || {})); }catch(e){}
    };
    try{
      const token = readToken();
      if(!token){ await push('error', {message: '未登录通义千问'}); return; }
      const base = (window.location.origin || 'https://chat.qwen.ai').replace(/\/+$/,'');
      const r = await fetch(base + '/api/v2/chats/' + encodeURIComponent(chatId), {
        method: 'DELETE',
        credentials: 'include',
        headers: {
          'Accept': 'application/json, text/plain, */*',
          'Version': '0.3.22', 'source': 'web',
          'Timezone': new Date().toString().replace(/[^\x20-\x7E]/g, ''),
          'X-Request-Id': uuid(),
          'Authorization': 'Bearer ' + token,
        },
      });
      await push('done', {chat_id: chatId, status: r.status});
    }catch(e){ await push('error', {message: String(e)}); }
  };
  return 'ok';
})()
"""


_TOKEN_JS = r"""
(function(){
  try{
    const r = localStorage.getItem('qwen_access_token_state');
    if(r){ const o = JSON.parse(r); const s = (o && o.state) ? o.state : o;
      if(s && typeof s.token === 'string' && s.token) return s.token; }
    const t = localStorage.getItem('token');
    if(t && t.split('.').length === 3) return t;
  }catch(e){}
  return '';
})()
"""


class _Runtime:
    def __init__(self, acc_id: str) -> None:
        self.acc_id = acc_id
        self.window = None
        self.queues: dict[str, queue.Queue] = {}
        self.lock = threading.Lock()
        self.js_lock = threading.Lock()
        self.ready = False
        self.logged_in = False
        self.url = ""
        self.stop = threading.Event()
        self.reload_needed = threading.Event()

    def has_active(self) -> bool:
        with self.lock:
            return bool(self.queues)

    def push(self, req_id: str, kind: str, data: str) -> bool:
        with self.lock:
            q = self.queues.get(req_id)
        if q is not None:
            try:
                q.put((kind, data), timeout=5)
            except Exception:
                pass
        if kind == "error":
            try:
                obj = json.loads(data or "{}")
            except Exception:
                obj = None
            if isinstance(obj, dict) and obj.get("reload"):
                self.reload_needed.set()
        return True

    def js(self, expr):
        if self.window is None:
            raise RuntimeError("中继窗口不可用")
        with self.js_lock:
            return self.window.evaluate_js(expr)

    def start_request(self, req_id: str, req: dict) -> None:
        if not self.ready or self.window is None:
            raise RuntimeError("中继浏览器尚未就绪")
        with self.js_lock:
            if not self.window.evaluate_js("typeof window.__qwenRelayStart==='function'"):
                raise RuntimeError("页面正在自动重连，请稍后重试")
            self.window.evaluate_js(
                "window.__qwenRelayStart("
                + json.dumps(req_id)
                + ","
                + json.dumps(req, ensure_ascii=False)
                + ")")

    def delete_chat(self, req_id: str, chat_id: str) -> None:
        if not self.ready or self.window is None:
            raise RuntimeError("中继浏览器尚未就绪")
        js = (
            "window.__qwenRelayDelete("
            + json.dumps(req_id)
            + ","
            + json.dumps(chat_id)
            + ")"
        )
        with self.js_lock:
            self.window.evaluate_js(js)


_RUNTIME: _Runtime | None = None


class _RelayApi:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime

    def relay_push(self, req_id: str, kind: str, data: str) -> bool:
        return self.runtime.push(req_id, kind, data)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    runtime: _Runtime

    def log_message(self, *a):
        pass

    def _send_json(self, obj: dict, status: int = 200) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except Exception:
            pass

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json(_health_payload(self.runtime))
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.startswith("/shutdown"):
            self._send_json({"ok": True})
            self.runtime.stop.set()
            return
        if self.path.startswith("/delete"):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:
                self._send_json({"error": f"bad json: {exc}"}, 400)
                return
            chat_id = str(req.get("chat_id") or "").strip()
            if not chat_id:
                self._send_json({"error": "chat_id required"}, 400)
                return
            req_id = uuid.uuid4().hex
            q: queue.Queue = queue.Queue(maxsize=100)
            with self.runtime.lock:
                self.runtime.queues[req_id] = q
            try:
                self.runtime.delete_chat(req_id, chat_id)
            except Exception as exc:
                with self.runtime.lock:
                    self.runtime.queues.pop(req_id, None)
                self._send_json({"error": str(exc)}, 503)
                return
            status = 500
            result: dict = {}
            try:
                kind, data = q.get(timeout=30)
                try:
                    result = json.loads(data) if data else {}
                except Exception:
                    result = {}
                status = int(result.get("status") or 200) if kind == "done" else 502
            except queue.Empty:
                pass
            finally:
                with self.runtime.lock:
                    self.runtime.queues.pop(req_id, None)
            self._send_json({
                "ok": status in (200, 201, 204),
                "status": status,
                "chat_id": chat_id,
                "detail": result,
            })
            return
        if not self.path.startswith("/chat"):
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._send_json({"error": f"bad json: {exc}"}, 400)
            return
        req_id = uuid.uuid4().hex
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self.runtime.lock:
            self.runtime.queues[req_id] = q
        try:
            self.runtime.start_request(req_id, req)
        except Exception as exc:
            with self.runtime.lock:
                self.runtime.queues.pop(req_id, None)
            self._send_json({"error": str(exc)}, 503)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        deadline = time.time() + 600
        try:
            while time.time() < deadline:
                try:
                    kind, data = q.get(timeout=1.0)
                except queue.Empty:
                    if self.runtime.stop.is_set():
                        break
                    continue
                try:
                    payload = json.loads(data) if data else {}
                except Exception:
                    payload = {"raw": data}
                event = {"type": kind}
                if isinstance(payload, dict):
                    event.update(payload)
                else:
                    event["data"] = payload
                self.wfile.write(
                    ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
                if kind in ("done", "error"):
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self.runtime.lock:
                self.runtime.queues.pop(req_id, None)


def _run_webview(acc_id: str, quiet: bool = False) -> None:
    global _RUNTIME
    import webview

    data_dir = user_data_dir()
    store = Store(data_dir / "webai2api.db")
    acc = store.get_account(acc_id)
    if not acc or acc.get("provider") != "qwen":
        clear_state(acc_id)
        return

    storage = data_dir / "browser_profiles" / acc_id
    storage.mkdir(parents=True, exist_ok=True)

    runtime = _Runtime(acc_id)
    _RUNTIME = runtime
    api = _RelayApi(runtime)

    window = webview.create_window(
        "通义千问浏览器会话",
        _QWEN_URL,
        width=1000,
        height=760,
        hidden=True,
        text_select=True,
    )
    window.expose(api.relay_push)
    runtime.window = window

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    _Handler.runtime = runtime
    port = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    write_state(acc_id, {
        "pid": os.getpid(),
        "port": port,
        "provider": "qwen",
        "started_at": time.time(),
    })

    def _wait_ready(timeout: float) -> bool:
        """等页面重新加载完、pywebview 桥接与注入脚本都就位。"""
        runtime.ready = False
        deadline = time.time() + timeout
        while time.time() < deadline and not runtime.stop.is_set():
            time.sleep(0.5)
            try:
                url = window.get_current_url() or ""
            except Exception:
                url = ""
            runtime.url = url
            if "chat.qwen.ai" not in url:
                continue
            try:
                if not runtime.js(
                    "typeof window.pywebview!=='undefined' && !!window.pywebview.api"):
                    continue
                runtime.js(_RELAY_JS)  # 幂等：已注入则直接返回
                ok = runtime.js("typeof window.__qwenRelayStart==='function'")
            except Exception:
                continue
            if ok:
                runtime.ready = True
                return True
        return False

    def worker():
        try:
            _wait_ready(60)
            shown = False
            last_check = 0.0
            last_sync = 0.0
            last_token = ""
            while not runtime.stop.is_set():
                time.sleep(1.0)
                # 请求侧发现过期会置位；这里统一整页重载换票
                if runtime.reload_needed.is_set():
                    runtime.reload_needed.clear()
                    if runtime.has_active():
                        continue  # 有请求在途，等下一轮再重载，不打断当前流
                    runtime.ready = False
                    runtime.logged_in = False
                    try:
                        runtime.js(
                            "try{location.replace((location.origin||"
                            "'https://chat.qwen.ai')+'/?t='+Date.now());}catch(e){}")
                    except Exception:
                        pass
                    _wait_ready(60)
                    continue
                if not runtime.ready:
                    if not _wait_ready(20):
                        continue
                if time.time() - last_check < 4.0:
                    continue
                last_check = time.time()
                try:
                    tok = runtime.js(_TOKEN_JS) or ""
                except Exception:
                    tok = ""
                runtime.logged_in = bool(tok)
                if runtime.logged_in:
                    if shown:
                        try:
                            window.hide()
                        except Exception:
                            pass
                        shown = False
                    # token 临期（<90s）且无在途请求时主动重载换票
                    if _jwt_exp(tok) and _jwt_exp(tok) < time.time() + 90 \
                            and not runtime.has_active():
                        runtime.reload_needed.set()
                        continue
                    # 把页面里的新 token 同步回库，面板健康检查/直连兜底才能用
                    if time.time() - last_sync > 30.0 and tok != last_token:
                        last_sync = time.time()
                        try:
                            store.update_account(acc_id, auth_token=tok)
                            last_token = tok
                        except Exception:
                            pass
                elif not shown and not quiet:
                    # quiet（预热）模式：登录态失效时保持隐藏，不弹窗打扰用户；
                    # 真正发请求走的是非 quiet 路径，届时仍会弹窗提示登录。
                    shown = True
        finally:
            try:
                server.shutdown()
            except Exception:
                pass
            clear_state(acc_id)

    webview.start(worker, storage_path=str(storage), private_mode=False)


def run_qwen_relay(acc_id: str, quiet: bool = False) -> None:
    _run_webview(acc_id, quiet=quiet)


# ── 主程序侧：启动 / 复用 ──────────────────────────────────

def stop_relay(acc_id: str, timeout: float = 5.0) -> None:
    """停止指定账号的中继（登录前/退出时用）。"""
    st = read_state(acc_id)
    port = int(st.get("port") or 0)
    pid = int(st.get("pid") or 0)
    if port:
        try:
            import urllib.request
            urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{port}/shutdown", data=b"{}", method="POST"),
                timeout=2)
        except Exception:
            pass
        time.sleep(0.5)
    if pid:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, timeout=timeout,
                               creationflags=0x08000000)
            else:
                os.kill(pid, 15)
        except Exception:
            pass
    clear_state(acc_id)


def stop_all_relays() -> None:
    try:
        for p in _state_dir().glob("*.json"):
            stop_relay(p.stem)
    except Exception:
        pass


_ENSURE_LOCKS: dict[str, threading.Lock] = {}
_ENSURE_GUARD = threading.Lock()


def _account_lock(acc_id: str) -> threading.Lock:
    with _ENSURE_GUARD:
        return _ENSURE_LOCKS.setdefault(acc_id, threading.Lock())


class RelayManager:
    """按账号管理千问中继子进程。"""

    def __init__(self, acc_id: str, logger=None) -> None:
        self.acc_id = acc_id
        self.logger = logger
        self._lock = _account_lock(acc_id)

    @staticmethod
    def _helper_cmd(acc_id: str, quiet: bool = False) -> list[str]:
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--relay-qwen", acc_id]
        else:
            from .paths import resource_path
            cmd = [sys.executable, str(resource_path("run.py")), "--relay-qwen", acc_id]
        if quiet:
            cmd.append("--quiet")
        return cmd

    @staticmethod
    def _health(port: int, timeout: float = 2.0) -> dict:
        import urllib.request
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return {}

    def ensure(self, timeout: float = 45.0, quiet: bool = False) -> int:
        with self._lock:
            st = read_state(self.acc_id)
            port = int(st.get("port") or 0)
            pid = int(st.get("pid") or 0)
            started = False
            if port and pid and self._alive(pid):
                # 已有中继进程：只等它就绪，绝不重复启动（会抢同一个 profile）
                pass
            else:
                if port:
                    clear_state(self.acc_id)
                flags = 0x08000000 if os.name == "nt" else 0
                try:
                    subprocess.Popen(
                        self._helper_cmd(self.acc_id, quiet=quiet),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=flags,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"启动千问中继失败: {exc}") from exc
                started = True

            deadline = time.time() + timeout
            last_err = ""
            while time.time() < deadline:
                time.sleep(0.8)
                st = read_state(self.acc_id)
                port = int(st.get("port") or 0)
                if not port:
                    continue
                health = self._health(port)
                if health.get("ok"):
                    return port
                last_err = str(health.get("error") or "")
                if started and not self._alive(int(st.get("pid") or 0)):
                    break
            raise RuntimeError(f"千问中继启动超时{('：' + last_err) if last_err else ''}")

    @staticmethod
    def _alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            try:
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=0x08000000,
                ).stdout
                return str(pid) in out
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

def _health_payload(runtime: "_Runtime") -> dict:
    """中继 /health 应答体；active=是否有进行中的请求（内存看护的空闲依据）。"""
    return {
        "ok": bool(runtime.ready),
        "logged_in": bool(runtime.logged_in),
        "url": runtime.url,
        "account_id": runtime.acc_id,
        "active": bool(runtime.has_active()),
    }


# ── 内存看护：长驻隐藏浏览器的回收 ──────────────────────
def _win_tree_pids(pid: int) -> list[int]:
    """Windows：Toolhelp32 快照建父子表，取 pid 及其全部后代。"""
    import ctypes
    from ctypes import wintypes

    class PE32(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    snap = ctypes.windll.kernel32.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if snap in (0, -1):
        return []
    children: dict[int, list[int]] = {}
    e = PE32()
    e.dwSize = ctypes.sizeof(PE32)
    ok = ctypes.windll.kernel32.Process32FirstW(snap, ctypes.byref(e))
    while ok:
        try:
            children.setdefault(e.th32ParentProcessID, []).append(e.th32ProcessID)
        except Exception:
            pass
        ok = ctypes.windll.kernel32.Process32NextW(snap, ctypes.byref(e))
    ctypes.windll.kernel32.CloseHandle(snap)
    out, stack = [], [pid]
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(children.get(cur, []))
    return out


def _posix_tree_pids(pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            fields = (d / "stat").read_text().rsplit(")", 1)[-1].split()
            children.setdefault(int(fields[1]), []).append(int(d.name))
        except Exception:
            continue
    out, stack = [], [pid]
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(children.get(cur, []))
    return out


def _win_rss_mb(pid: int) -> float:
    import ctypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint), ("PageFaultCount", ctypes.c_uint),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    h = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)  # QUERY|VM_READ
    if not h:
        return 0.0
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    ctypes.windll.kernel32.CloseHandle(h)
    return (pmc.WorkingSetSize / 1048576.0) if ok else 0.0


def _posix_rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def relay_tree_rss_mb(pid: int) -> float:
    """中继进程树（含 WebView2 子进程）的常驻内存合计，MB；失败返回 0。"""
    if pid <= 0:
        return 0.0
    try:
        tree = (_win_tree_pids if os.name == "nt" else _posix_tree_pids)(pid)
        rss = (_win_rss_mb if os.name == "nt" else _posix_rss_mb)
        return round(sum(rss(p) for p in tree), 1)
    except Exception:
        return 0.0


def _relay_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # SYNCHRONIZE
            if h:
                ctypes.windll.kernel32.CloseHandle(h)
                return True
            return False
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _should_recycle(rss_mb: float, limit_mb: float, health: dict) -> bool:
    """回收判定（纯函数，供测试）：超阈值 且 中继空闲。"""
    return (limit_mb > 0 and rss_mb > limit_mb
            and bool(health) and health.get("ok") and not health.get("active"))


_GUARD_STARTED = threading.Event()


def start_memory_guard(store, logger=None, interval: float = 60.0) -> None:
    """启动中继内存看护（单例守护线程）。

    每轮扫描 relay_state/*.json：进程树内存超 qwen_relay_max_mb（默认
    900MB，面板可调）且中继空闲（/health active=false）时回收——下一次
    请求会自动冷启动（约 10~15 秒），登录态不受影响。测量失败/中继
    忙碌一律跳过，绝不打断进行中的流。
    """
    if _GUARD_STARTED.is_set():
        return
    _GUARD_STARTED.set()

    def _loop():
        while True:
            time.sleep(interval)
            try:
                limit = float(store.get_setting("qwen_relay_max_mb", "900") or 0)
                for st_path in _state_dir().glob("*.json"):
                    try:
                        st = read_state(st_path.stem)
                        pid = int(st.get("pid") or 0)
                        port = int(st.get("port") or 0)
                        if not pid or not _relay_alive(pid):
                            continue
                        rss = relay_tree_rss_mb(pid)
                        if rss <= 0:
                            continue
                        health = RelayManager._health(port) if port else {}
                        if _should_recycle(rss, limit, health):
                            if logger:
                                logger.warning(
                                    "千问中继内存 %.0fMB 超限 %.0fMB 且空闲，回收 %s"
                                    "（下次请求自动冷启动）",
                                    rss, limit, st_path.stem)
                            stop_relay(st_path.stem)
                        elif rss > limit and logger:
                            logger.info("千问中继内存 %.0fMB 超限但有进行中请求，"
                                        "暂不回收 %s", rss, st_path.stem)
                    except Exception:
                        continue
            except Exception:
                pass

    threading.Thread(target=_loop, daemon=True,
                     name="qwen-relay-memguard").start()
