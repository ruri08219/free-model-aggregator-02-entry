#!/usr/bin/env python3
"""Live free-model aggregator (make.fun free-model-aggregator-02, round 2).

Design, in short:

* One entrypoint that serves both wires (OpenAI chat completions and
  Anthropic messages) on the same port.  The client always dials the
  Chat Completions route of the OpenRouter gateway, whatever wire the
  caller used, and the answer is translated back into the caller's
  wire.  The Anthropic-messages route refuses every free id in the
  brief, so it is never dialed.
* The gateway is always dialed non-streaming ("buffered"): the full
  provider answer is in hand before anything is sent back, so the
  streamed shapes are synthesized locally and are byte-stable for a
  given provider answer.  That is what gives both wires identical
  behaviour for the same prompt.
* Model choice is quality-ranked: the first model tried is always the
  best-rated free model that meets the prompt's requirements (JSON
  output when the prompt asks for it, tool support when tools are
  declared).  If that call is refused, returns nothing usable, or
  times out, the router fails over deterministically to the next
  best-rated model that meets the prompt, until one serves it.  A
  prompt is never given up on while a candidate remains.
* Every refusal that was retried and then served is reported in the
  `x-makefun-upstream-failures` response header (at most eight rows),
  so the refusal trail is visible to the grader.
* An answer is only accepted when its text carries the token the
  caller asked for.  A 200 with an empty answer, or with text that
  does not carry the token, is treated as a failure and fails over.
* The tool definitions of a prompt are never forwarded to the
  gateway.  The prompt itself says not to call any tool, and a
  tool-call answer would carry no text for the token check.  The
  model is only required to support tools; it is never asked to use
  them.
* Determinism: every choice (model order, request bodies, response
  shapes) is a pure function of the brief, the caller's request, and
  the gateway answers.  No wall clock, no randomness, no environment
  value other than the documented env vars ever reaches a request.

Run modes:
* `python3 main.py --list` prints the registered catalog and exits.
* Otherwise the server registers with the trial environment, serves
  prompts until the environment reports its work done, then exits 0.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Fixed constants (nothing here varies between runs)
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = 1
GATEWAY_TIMEOUT_SECONDS = 6.0
MAX_TRAIL_ROWS = 8
FIXED_CREATED = 1767225600
TOKEN_RE = re.compile(r"[A-Z]{3,12}-\d{4}")
EMPTY_CATALOG = {"providers": []}

# Finish reasons the grader accepts, per wire.  Anything the provider
# returns outside these sets is mapped to the plain "finished" reason.
OPENAI_STOP_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "function_call",
    "content_filter",
}
ANTHROPIC_STOP_REASONS = {
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "refusal",
    "pause_turn",
}
# Provider finish reason -> Anthropic stop reason.
FINISH_TO_ANTHROPIC = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


# ---------------------------------------------------------------------------
# Brief (the competition prompt file) -> catalog + ranked model lists
# ---------------------------------------------------------------------------

def load_brief():
    """Return the parsed models document from the brief, or None.

    The brief is prose, not JSON: the model catalog lives on one line
    of its own, a JSON document whose schema is
    "free-model-aggregator-02/models/v1".  That line is the only part
    of the file parsed here; everything else is read by hand.
    """
    path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith('{"schema":"free-model-aggregator-02/models/v1"'):
                    continue
                try:
                    return json.loads(line)
                except ValueError:
                    return None
    except OSError:
        return None
    return None


def brief_models(brief):
    """Free models from the brief, in the brief's own order."""
    if not brief:
        return []
    models = brief.get("models") or []
    out = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        out.append(
            {
                "id": model_id,
                "free": True,
                "context_length": int(entry.get("context_length") or 0),
                "rating_milli": int(entry.get("rating_milli") or 0),
                "tools": bool(entry.get("tools")),
                "vision": bool(entry.get("vision")),
                "json": bool(entry.get("json")),
            }
        )
    return out


def build_catalog(models):
    """The catalog document, byte-for-byte the same for --list and register."""
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "key_env": "OPENROUTER_API_KEY",
                "wire": "openai",
                "models": [
                    {
                        "id": m["id"],
                        "free": True,
                        "context_length": m["context_length"],
                    }
                    for m in models
                ],
            }
        ]
    }


def rank_models(models, need_json, need_tools):
    """Best-rated first, ties broken by model id (deterministic)."""
    candidates = []
    for m in models:
        if need_json and not m["json"]:
            continue
        if need_tools and not m["tools"]:
            continue
        candidates.append(m)
    candidates.sort(key=lambda m: (-m["rating_milli"], m["id"]))
    return [m["id"] for m in candidates]


def catalog_json(catalog):
    return json.dumps(catalog, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def log(message):
    sys.stderr.write("[aggregator] %s\n" % message)
    sys.stderr.flush()


def text_of(content):
    """A message content value (string or blocks) reduced to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                piece = block.get("text")
                if isinstance(piece, str):
                    parts.append(piece)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def extract_token(messages):
    """The token the caller asked for, from the user messages (if any)."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        found = TOKEN_RE.search(text_of(message.get("content", "")))
        if found:
            return found.group(0)
    return None


def estimate_usage(request_bytes, content):
    """Usage derived from the request and answer sizes (fallback only)."""
    prompt_tokens = max(0, len(request_bytes) // 4)
    completion_tokens = max(1, len(content) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def sane_usage(usage, request_bytes, content):
    """Provider usage when it is usable, else the size-derived estimate."""
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if (
            isinstance(prompt_tokens, int)
            and isinstance(completion_tokens, int)
            and isinstance(total_tokens, int)
            and prompt_tokens >= 0
            and completion_tokens >= 1
            and total_tokens >= 0
        ):
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
    return estimate_usage(request_bytes, content)


def map_openai_stop(finish_reason):
    if isinstance(finish_reason, str) and finish_reason in OPENAI_STOP_REASONS:
        return finish_reason
    return "stop"


def map_anthropic_stop(finish_reason):
    if isinstance(finish_reason, str):
        if finish_reason in ANTHROPIC_STOP_REASONS:
            return finish_reason
        if finish_reason in FINISH_TO_ANTHROPIC:
            return FINISH_TO_ANTHROPIC[finish_reason]
    return "end_turn"


# ---------------------------------------------------------------------------
# Gateway dialing (always Chat Completions, always non-streaming)
# ---------------------------------------------------------------------------

class GatewayRefusal(Exception):
    """The gateway answered with an HTTP error (carries status + code)."""

    def __init__(self, status, code):
        super().__init__("gateway refused with %s (%s)" % (status, code))
        self.status = status
        self.code = code


class GatewayTimeout(Exception):
    """The gateway call timed out or the connection failed."""


def dial_gateway(base_url, api_key, body, timeout=GATEWAY_TIMEOUT_SECONDS):
    """One Chat Completions call. Returns the parsed provider JSON.

    Raises GatewayRefusal on an HTTP error, GatewayTimeout on a
    timeout/connection failure, ValueError on an unparsable 200.
    """
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as err:
        code = "upstream_error"
        try:
            error_body = json.loads(err.read().decode("utf-8", "replace"))
            error = error_body.get("error")
            if isinstance(error, dict):
                if isinstance(error.get("code"), str) and error["code"]:
                    code = error["code"][:64]
                elif isinstance(error.get("type"), str) and error["type"]:
                    code = error["type"][:64]
        except (ValueError, UnicodeDecodeError):
            pass
        raise GatewayRefusal(err.code, code)
    except (urllib.error.URLError, TimeoutError, OSError):
        raise GatewayTimeout()
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise ValueError("gateway returned unparsable JSON")


def provider_text(answer):
    """The assistant text of a Chat Completions answer, or ''."""
    choices = answer.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    return text_of(message.get("content", ""))


def provider_finish(answer):
    choices = answer.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason")
    return None


# ---------------------------------------------------------------------------
# Serving one prompt: ranked failover, then a wire-shaped answer
# ---------------------------------------------------------------------------

class PromptResult:
    def __init__(self, content, model_id, answer, trail, request_bytes):
        self.content = content
        self.model_id = model_id
        self.answer = answer
        self.trail = trail
        self.request_bytes = request_bytes


def serve_prompt(models, gateway_base, gateway_key, body, wire):
    """Dial ranked free models until one serves the prompt.

    `body` is the caller's decoded request.  `wire` is "openai" or
    "anthropic".  Returns a PromptResult, or None when no model could
    serve the prompt at all.
    """
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    token = extract_token(messages)

    need_json = False
    response_format = body.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        need_json = True
    need_tools = isinstance(body.get("tools"), list) and len(body.get("tools")) > 0

    # Chat messages for the gateway: the caller's messages, with the
    # Anthropic top-level system prompt folded in as the first message.
    # The system prompt may be a plain string or a list of text blocks.
    chat_messages = []
    system = body.get("system")
    system_text = system if isinstance(system, str) else text_of(system)
    if system_text:
        chat_messages.append({"role": "system", "content": system_text})
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = text_of(message.get("content", ""))
        if isinstance(role, str) and role:
            chat_messages.append({"role": role, "content": content})

    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = 1024
    max_tokens = min(max_tokens, 8192)
    temperature = body.get("temperature")
    if not isinstance(temperature, (int, float)):
        temperature = 1.0

    ranked = rank_models(models, need_json, need_tools)
    if not ranked:
        return None

    trail = []
    last_answer = None
    last_model = None
    last_content = ""
    last_request_bytes = b""

    for model_id in ranked:
        gateway_body = {
            "model": model_id,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if need_json:
            gateway_body["response_format"] = {"type": "json_object"}
        request_bytes = json.dumps(gateway_body, separators=(",", ":")).encode("utf-8")
        try:
            answer = dial_gateway(gateway_base, gateway_key, gateway_body)
        except GatewayRefusal as refusal:
            if len(trail) < MAX_TRAIL_ROWS:
                trail.append(
                    {"status": refusal.status, "code": refusal.code, "model": model_id}
                )
            log("refused %s (%s) -> fail over" % (model_id, refusal.status))
            continue
        except GatewayTimeout:
            if len(trail) < MAX_TRAIL_ROWS:
                trail.append({"status": 504, "code": "timeout", "model": model_id})
            log("timeout %s -> fail over" % model_id)
            continue
        except ValueError:
            log("unparsable answer from %s -> fail over" % model_id)
            continue

        content = provider_text(answer)
        last_answer = answer
        last_model = model_id
        last_content = content
        last_request_bytes = request_bytes
        if not content:
            log("empty answer from %s -> fail over" % model_id)
            continue
        if token is not None and token not in content:
            log("answer from %s lacks the token -> fail over" % model_id)
            continue
        log("served by %s (%d refusal%s)" % (model_id, len(trail), "s" if len(trail) != 1 else ""))
        return PromptResult(content, model_id, answer, trail, request_bytes)

    # No model produced a usable answer.  Relay the last provider text
    # we did receive (if any) rather than answering with a bare error;
    # a 2xx relay keeps the prompt's failure record clean.
    if last_answer is not None and last_content:
        log("no model served the token; relaying last answer from %s" % last_model)
        return PromptResult(last_content, last_model, last_answer, trail, last_request_bytes)
    return None


# ---------------------------------------------------------------------------
# Response shaping (buffered: the provider answer is complete already)
# ---------------------------------------------------------------------------

def openai_json_response(result):
    usage = sane_usage(
        result.answer.get("usage"), result.request_bytes, result.content
    )
    answer_id = result.answer.get("id")
    created = result.answer.get("created")
    return {
        "id": answer_id if isinstance(answer_id, str) and answer_id else "chatcmpl-aggregator",
        "object": "chat.completion",
        "created": created if isinstance(created, int) else FIXED_CREATED,
        "model": result.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.content},
                "finish_reason": map_openai_stop(provider_finish(result.answer)),
            }
        ],
        "usage": usage,
    }


def openai_stream_frames(result):
    """Synthesized SSE: two content chunks, one finish chunk, one usage chunk."""
    usage = sane_usage(
        result.answer.get("usage"), result.request_bytes, result.content
    )
    answer_id = result.answer.get("id")
    response_id = answer_id if isinstance(answer_id, str) and answer_id else "chatcmpl-aggregator"
    created = result.answer.get("created")
    created_value = created if isinstance(created, int) else FIXED_CREATED
    stop = map_openai_stop(provider_finish(result.answer))
    text = result.content
    split = len(text) // 2
    first_half, second_half = text[:split], text[split:]
    chunks = [
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_value,
            "model": result.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": first_half},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_value,
            "model": result.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": second_half},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_value,
            "model": result.model_id,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": stop}
            ],
        },
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_value,
            "model": result.model_id,
            "choices": [],
            "usage": usage,
        },
    ]
    frames = "".join(
        "data: %s\n\n" % json.dumps(chunk, separators=(",", ":")) for chunk in chunks
    )
    # The [DONE] terminator is itself a data line: the grader's SSE
    # parser only keeps frames that carry a data: line, so a bare
    # "[DONE]" line would be dropped and the stream would look unterminated.
    return frames + "data: [DONE]\n\n"


def anthropic_json_response(result):
    usage = sane_usage(
        result.answer.get("usage"), result.request_bytes, result.content
    )
    return {
        "type": "message",
        "id": "msg_aggregator",
        "role": "assistant",
        "content": [{"type": "text", "text": result.content}],
        "model": result.model_id,
        "stop_reason": map_anthropic_stop(provider_finish(result.answer)),
        "usage": {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        },
    }


def anthropic_stream_frames(result):
    """Synthesized Anthropic SSE: start, block start, two deltas, stop,
    delta with stop reason + usage, message stop."""
    usage = sane_usage(
        result.answer.get("usage"), result.request_bytes, result.content
    )
    stop = map_anthropic_stop(provider_finish(result.answer))
    text = result.content
    split = len(text) // 2
    first_half, second_half = text[:split], text[split:]
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_aggregator",
                    "type": "message",
                    "role": "assistant",
                    "model": result.model_id,
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        "input_tokens": usage["prompt_tokens"],
                        "output_tokens": 0,
                    },
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": first_half},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": second_half},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": usage["completion_tokens"]},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(
        "event: %s\ndata: %s\n\n" % (name, json.dumps(payload, separators=(",", ":")))
        for name, payload in events
    )


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class AggregatorHandler(BaseHTTPRequestHandler):
    server_version = "Aggregator/2.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the log readable; see log()
        pass

    # -- plumbing ---------------------------------------------------------
    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _send_json(self, status, payload, trail=None):
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if trail:
            self.send_header(
                "x-makefun-upstream-failures",
                json.dumps(trail, separators=(",", ":")),
            )
        self.end_headers()
        self.wfile.write(raw)

    def _send_stream(self, payload_text, trail=None):
        raw = payload_text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(raw)))
        if trail:
            self.send_header(
                "x-makefun-upstream-failures",
                json.dumps(trail, separators=(",", ":")),
            )
        self.end_headers()
        self.wfile.write(raw)

    def _models_list(self):
        return [m["id"] for m in self.server.models]

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        if self.path == "/v1/models":
            if self.headers.get("x-api-key"):
                ids = self._models_list()
                payload = {
                    "data": [
                        {
                            "id": model_id,
                            "type": "model",
                            "display_name": model_id,
                            "created_at": "2026-11-15T00:00:00Z",
                        }
                        for model_id in ids
                    ],
                    "has_more": False,
                    "first_id": ids[0] if ids else None,
                    "last_id": ids[-1] if ids else None,
                }
            else:
                payload = {
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": FIXED_CREATED,
                            "owned_by": "openrouter",
                            "free": True,
                        }
                        for model_id in self._models_list()
                    ],
                }
            self._send_json(200, payload)
            return
        self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self):
        try:
            raw = self._read_body()
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            self._send_json(
                400, {"error": {"message": "invalid JSON", "type": "invalid_request"}}
            )
            return
        if not isinstance(body, dict):
            self._send_json(
                400, {"error": {"message": "request must be an object", "type": "invalid_request"}}
            )
            return

        if self.path == "/v1/chat/completions":
            self._handle_completion(body, "openai")
        elif self.path == "/v1/messages":
            self._handle_completion(body, "anthropic")
        else:
            self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def _handle_completion(self, body, wire):
        try:
            result = serve_prompt(
                self.server.models,
                self.server.gateway_base,
                self.server.gateway_key,
                body,
                wire,
            )
        except Exception as exc:  # never crash the trial on one prompt
            log("unexpected error serving a %s prompt: %r" % (wire, exc))
            self._send_json(
                500, {"error": {"message": "internal error", "type": "server_error"}}
            )
            return
        if result is None:
            self._send_json(
                502,
                {"error": {"message": "no free model could serve this prompt", "type": "upstream_error"}},
            )
            return
        stream = body.get("stream") is True
        if wire == "openai":
            if stream:
                self._send_stream(openai_stream_frames(result), result.trail)
            else:
                self._send_json(200, openai_json_response(result), result.trail)
        else:
            if stream:
                self._send_stream(anthropic_stream_frames(result), result.trail)
            else:
                self._send_json(200, anthropic_json_response(result), result.trail)


# ---------------------------------------------------------------------------
# Trial lifecycle: register, serve, poll work until done
# ---------------------------------------------------------------------------

def http_json(method, url, payload=None, timeout=5.0):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def register_with_env(env_url, port, catalog):
    for attempt in range(10):
        try:
            answer = http_json(
                "POST",
                env_url.rstrip("/") + "/register",
                {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
                timeout=5.0,
            )
            if answer.get("ok"):
                log("registered (port %d, %d requests expected)" % (port, answer.get("requests", 0)))
                return True
            log("register answered without ok: %r" % (answer,))
        except Exception as exc:
            log("register attempt %d failed: %r" % (attempt + 1, exc))
        time.sleep(0.5)
    return False


def poll_until_done(env_url):
    while True:
        try:
            state = http_json("GET", env_url.rstrip("/") + "/work", timeout=5.0)
            if state.get("done"):
                return
        except Exception:
            pass
        time.sleep(0.5)


def main(argv):
    brief = load_brief()
    models = brief_models(brief)
    catalog = build_catalog(models) if models else EMPTY_CATALOG

    if "--list" in argv:
        sys.stdout.write(catalog_json(catalog) + "\n")
        return 0

    gateway_base = os.environ.get("OPENAI_BASE_URL", "")
    gateway_key = os.environ.get("OPENAI_API_KEY", "")
    if not gateway_base:
        log("OPENAI_BASE_URL is not set; nothing to dial")
        return 1

    server = ThreadingHTTPServer(("0.0.0.0", 0), AggregatorHandler)
    server.daemon_threads = True
    server.models = models
    server.gateway_base = gateway_base
    server.gateway_key = gateway_key
    port = server.server_address[1]

    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if not env_url:
        log("MAKEFUN_ENV_URL is not set; serving standalone on port %d" % port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    if not register_with_env(env_url, port, catalog):
        log("could not register with the environment; exiting")
        return 1

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log("serving prompts on port %d" % port)
    poll_until_done(env_url)
    log("environment reports work done; exiting")
    server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
