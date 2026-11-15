"""free-model-aggregator-02: one server, two wires, live free models behind every answer.

An aggregator for OpenRouter's free models. A client points at this server and every
completion it answers comes from a real free model dialled through the grader's gateway
on Chat Completions; the server itself never writes an answer of its own.

Routes
    GET  /v1/models            the free catalog, in the wire the auth header names
    POST /v1/chat/completions  OpenAI Chat Completions, JSON or server-sent events
    POST /v1/messages          Anthropic Messages, JSON or server-sent events

Lifecycle
    Listen on 0.0.0.0 at a free port, register {protocol_version, port, catalog} with
    the env, serve the six prompts, poll GET $MAKEFUN_ENV_URL/work and exit 0 when it
    says done. `python3 main.py --list` prints the catalog and exits 0.

Routing
    The brief at $MAKEFUN_PROMPT_FILE lists the free models with the house's ratings
    and capabilities. For each request the server reads the requirements off the
    request body itself (JSON mode, tools, vision), ranks the free models that meet
    them by rating (best first, id order breaking ties) and dials the best. A refusal
    (429 or 5xx) fails over to the next best model that meets the requirements, and a
    model that refused is not tried again for the rest of the trial: a free model
    that rate-limits stays limited for the minute. Recovered refusals are reported on
    the answer in the x-makefun-upstream-failures header, so the failover column can
    see them. Every choice is a function of the brief, the requests and the gateway's
    answers: no wall clock, no unseeded randomness, so a replay leaves the same state.
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---------------------------------------------------------------- constants

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
# One gateway call may take this long; a prompt's whole dial loop may take this long.
GATEWAY_CALL_TIMEOUT_SEC = 8.0
PROMPT_BUDGET_SEC = 25.0
MAX_ATTEMPTS_PER_PROMPT = 3
MAX_TRAIL_ROWS = 8
FAILURES_HEADER = "x-makefun-upstream-failures"

# Fixed stamps, so no reply ever reads the clock (a replay must match byte for byte).
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"

# The provider's public endpoint, as the catalog names it (never the gateway's address).
OPENROUTER_PUBLIC_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_ROOT = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")
# Read for completeness: this server uses no randomness at all, so the seed is never drawn on.
TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "")

MODELS_SCHEMA = "free-model-aggregator-02/models/v1"
PROMPTS_SCHEMA = "free-model-aggregator-02/prompts/v1"
SCHEMA_PREFIX = "free-model-aggregator-02/"

Json = dict[str, object]


# ---------------------------------------------------------------- small helpers


def log(message: str) -> None:
    sys.stderr.write(f"aggregator: {message}\n")
    sys.stderr.flush()


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def text_of(value: object) -> str:
    """The text of a string or of a block array's text blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            block["text"]
            for block in value
            if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def int_or(value: object, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return fallback


def error_code_of(body_text: str) -> str:
    """The gateway error body's code (or type), truncated, for the failures trail."""
    try:
        value = json.loads(body_text)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not is_record(value):
        return ""
    error = value.get("error")
    if not is_record(error):
        return ""
    code = error.get("code")
    if isinstance(code, str) and code != "":
        return code[:64]
    error_type = error.get("type")
    if isinstance(error_type, str) and error_type != "":
        return error_type[:64]
    return ""


# ---------------------------------------------------------------- the brief


def _object_end(text: str, start: int) -> int:
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents(text: str) -> list[Json]:
    """Every JSON object in `text` carrying a string `schema`, in order."""
    found: list[Json] = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                found.append(value)
        start = text.find("{", start + 1)
    return found


def _model_row(value: object) -> Json | None:
    """One model row from the brief, normalized to the shape the router uses."""
    if not is_record(value):
        return None
    raw = value.get("id") if isinstance(value.get("id"), str) else ""
    if raw == "":
        raw = value.get("model") if isinstance(value.get("model"), str) else ""
    if raw == "":
        return None
    provider = value.get("provider") if isinstance(value.get("provider"), str) else ""
    if provider == "openrouter" and not raw.lower().startswith("openrouter/"):
        raw = f"openrouter/{raw}"
    display = value.get("display_name") if isinstance(value.get("display_name"), str) else ""
    rating = value.get("rating_milli")
    rating = rating if is_number(rating) and math.isfinite(rating) else 0
    context = value.get("context_length")
    context = context if is_number(context) and math.isfinite(context) else None
    return {
        "id": raw,
        "provider": provider if provider != "" else "openrouter",
        "display_name": display if display != "" else raw,
        "rating_milli": rating,
        "context_length": context,
        "tools": value.get("tools") if isinstance(value.get("tools"), bool) else None,
        "vision": value.get("vision") if isinstance(value.get("vision"), bool) else None,
        "json": value.get("json") if isinstance(value.get("json"), bool) else None,
        "free": value.get("free") is not False,
    }


def _prompt_row(value: object) -> Json | None:
    """One prompt row from the brief's manifest (kept for the record; routing reads the body)."""
    if not is_record(value):
        return None
    prompt_id = value.get("id") if isinstance(value.get("id"), str) else ""
    if prompt_id == "":
        return None
    requires = value.get("requires") if is_record(value.get("requires")) else {}
    return {
        "id": prompt_id,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": {
            "tools": requires.get("tools") is True,
            "vision": requires.get("vision") is True,
            "json": requires.get("json") is True,
        },
    }


def read_brief(path: str) -> Json:
    """The brief's two documents, found by schema. Never parses the brief's prose."""
    if path == "":
        return {"models": [], "prompts": [], "error": "MAKEFUN_PROMPT_FILE is not set"}
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        return {"models": [], "prompts": [], "error": f"cannot read {path}: {err}"}
    docs = _documents(text)
    models: list[Json] = []
    prompts: list[Json] = []
    for doc in docs:
        schema = doc.get("schema")
        if not isinstance(schema, str) or SCHEMA_PREFIX not in schema:
            continue
        rows = doc.get("models")
        if rows is None:
            rows = doc.get("ratings")
        if isinstance(rows, list) and not models:
            models = [m for m in (_model_row(v) for v in rows) if m is not None]
        rows = doc.get("prompts")
        if rows is None:
            rows = doc.get("manifest")
        if isinstance(rows, list) and not prompts:
            prompts = [p for p in (_prompt_row(v) for v in rows) if p is not None]
    error = None
    if not models:
        error = f"the brief at {path} holds no {MODELS_SCHEMA} document"
    return {"models": models, "prompts": prompts, "error": error}


def build_catalog(brief: Json) -> Json:
    """The catalog: one provider, exactly the brief's free models, once each."""
    seen: set[str] = set()
    rows: list[Json] = []
    for model in brief["models"]:
        model_id = model["id"]
        if not model["free"] or model_id in seen:
            continue
        seen.add(model_id)
        rows.append({"id": model_id, "free": True, "context_length": model["context_length"]})
    if not rows:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": OPENROUTER_PUBLIC_BASE,
                "key_env": OPENROUTER_KEY_ENV,
                "wire": "openai",
                "models": rows,
            }
        ]
    }


BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)


def catalog_text() -> str:
    """The one catalog document, serialized the same way everywhere it is sent."""
    return json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False)


def free_models() -> list[Json]:
    return [m for m in BRIEF["models"] if m["free"]]


# ---------------------------------------------------------------- routing


def requirements_from_body(body: Json) -> Json:
    """What a model must do to serve this request, read off the request body itself."""
    requires = {"tools": False, "vision": False, "json": False}
    tools = body.get("tools")
    if isinstance(tools, list) and len(tools) > 0:
        requires["tools"] = True
    response_format = body.get("response_format")
    if is_record(response_format) and response_format.get("type") in ("json_object", "json_schema"):
        requires["json"] = True
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not is_record(message):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if is_record(block) and block.get("type") in ("image", "image_url"):
                    requires["vision"] = True
    return requires


def meets(requires: Json, model: Json) -> bool:
    """Whether a model may serve a request with these requirements."""
    if requires["tools"] and model["tools"] is not True:
        return False
    if requires["vision"] and model["vision"] is not True:
        return False
    if requires["json"] and model["json"] is not True:
        return False
    return True


# Models that refused in this trial: a free model that rate-limits stays limited
# for the minute, so they are skipped for later prompts. Deterministic: it only
# grows from gateway answers this trial has already seen.
REFUSED: set[str] = set()
REFUSED_LOCK = threading.Lock()


def ranked_candidates(requires: Json, skip: set[str]) -> list[Json]:
    """Eligible free models, best rating first, id order breaking ties."""
    eligible = [m for m in free_models() if m["id"] not in skip and meets(requires, m)]
    eligible.sort(key=lambda m: (-m["rating_milli"], m["id"]))
    return eligible


def openai_model_list(models: list[Json]) -> Json:
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]}
            for m in models
        ],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    return {
        "data": [
            {
                "type": "model",
                "id": m["id"],
                "display_name": m["display_name"],
                "created_at": FIXED_CREATED_AT,
            }
            for m in models
        ],
        "has_more": False,
    }


# ---------------------------------------------------------------- wire translation


def _anthropic_part(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part."""
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    source = block.get("source")
    if block.get("type") == "image" and is_record(source):
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{source['data']}"}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"), ensure_ascii=False)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def anthropic_to_openai(body: Json) -> Json:
    """An Anthropic Messages request as a Chat Completions request (model set by dial)."""
    messages: list[Json] = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                calls.append(
                    {
                        "id": block.get("id") if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(
                                block.get("input") if is_record(block.get("input")) else {},
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {"role": "assistant", "content": text if text != "" else None}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                        "content": _tool_result_text(block.get("content")),
                    }
                )
                continue
            part = _anthropic_part(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out: Json = {"model": "auto", "messages": messages}
    if is_number(body.get("max_tokens")):
        out["max_tokens"] = body["max_tokens"]
    if is_number(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_number(body.get("top_p")):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and len(body["stop_sequences"]) > 0:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools: list[Json] = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": raw["name"],
                        "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                        "parameters": raw.get("input_schema")
                        if is_record(raw.get("input_schema"))
                        else {"type": "object", "properties": {}},
                    },
                }
            )
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _arguments_of(value: object) -> Json:
    """A tool call's arguments as an object; {} when they are not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def _stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def openai_to_anthropic(answer: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object, naming the dialled model."""
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content: list[Json] = []
    raw_text = message.get("content")
    if isinstance(raw_text, str) and raw_text != "":
        content.append({"type": "text", "text": raw_text})
    elif isinstance(raw_text, list):
        text = text_of(raw_text)
        if text != "":
            content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") if isinstance(call.get("id"), str) and call.get("id") != "" else "toolu_unknown",
                "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                "input": _arguments_of(fn.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = answer.get("usage") if is_record(answer.get("usage")) else {}
    prompt_tokens = int_or(usage.get("prompt_tokens"), 0)
    completion_tokens = int_or(usage.get("completion_tokens"), -1)
    if completion_tokens < 0:
        # The provider omitted usage: the answer provably holds at least one token of text.
        completion_tokens = 1 if any(block.get("type") == "text" and block.get("text") != "" for block in content) else 0
    answer_id = answer.get("id") if isinstance(answer.get("id"), str) and answer.get("id") != "" else "aggregate"
    return {
        "id": f"msg_{answer_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason_of(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
    }


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: an `event:` line when named, the JSON data line, a blank line."""
    prefix = "" if event is None else f"event: {event}\n"
    return f"{prefix}data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"


def _rewrite_openai_frame(frame: str, model: str) -> str:
    """One OpenAI SSE frame with every data payload's `model` the dialled id."""
    lines: list[str] = []
    for line in frame.split("\n"):
        trimmed = line.strip()
        if not trimmed.startswith("data:"):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == "" or payload == "[DONE]":
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append(f"data: {json.dumps({**chunk, 'model': model}, separators=(',', ':'), ensure_ascii=False)}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n\n"


def _iter_frames(chunks):
    """SSE frames from upstream byte chunks, split on the blank line ending each one."""
    buffer = ""
    for chunk in chunks:
        buffer += chunk.decode("utf-8", "replace").replace("\r\n", "\n")
        at = buffer.find("\n\n")
        while at != -1:
            yield buffer[:at]
            buffer = buffer[at + 2 :]
            at = buffer.find("\n\n")
    if buffer.strip() != "":
        yield buffer


def _frame_data_lines(frame: str):
    """The data payloads of one SSE frame, in order."""
    for line in frame.split("\n"):
        trimmed = line.strip()
        if trimmed.startswith("data:"):
            yield trimmed[5:].strip()


def collect_frames(upstream, deadline: float) -> list[str]:
    """Read an upstream SSE stream whole, up to the deadline, as frame strings.

    Frames are collected before anything is sent to the client, so a stream the
    provider left empty can be failed over from instead of served blank.
    """
    frames: list[str] = []
    try:
        for frame in _iter_frames(upstream.chunks()):
            if time.monotonic() > deadline:
                break
            frames.append(frame)
    finally:
        upstream.close()
    return frames


def stream_has_answer(frames: list[str]) -> bool:
    """Whether an OpenAI SSE stream carried any text or tool call."""
    for frame in frames:
        for payload in _frame_data_lines(frame):
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            if isinstance(delta.get("content"), str) and delta["content"].strip() != "":
                return True
            calls = delta.get("tool_calls")
            if isinstance(calls, list) and len(calls) > 0:
                return True
    return False


def answer_text_of_openai(answer: Json) -> tuple[str, bool]:
    """A Chat Completions answer's text and whether it called a tool.

    An answer with neither is empty: the model produced nothing to serve, and
    the prompt should be failed over from rather than answered blank.
    """
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = message.get("content")
    text = content if isinstance(content, str) else text_of(content)
    calls = message.get("tool_calls")
    has_tools = isinstance(calls, list) and len(calls) > 0
    return text, has_tools


def openai_stream_frames(frames: list[str], model: str):
    """Collected Chat Completions SSE frames as they came, the model rewritten.

    Repairs only a stream the provider left unfinished: a missing usage chunk is
    added (the wire needs one) and a missing [DONE] is appended.
    """
    saw_usage = False
    saw_done = False
    for frame in frames:
        for payload in _frame_data_lines(frame):
            if payload == "[DONE]":
                saw_done = True
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if is_record(chunk) and is_record(chunk.get("usage")):
                saw_usage = True
        yield _rewrite_openai_frame(frame, model)
    if not saw_usage:
        yield sse_event(
            None,
            {
                "id": "chatcmpl-aggregate",
                "object": "chat.completion.chunk",
                "created": FIXED_CREATED,
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 1},
            },
        )
    if not saw_done:
        yield "data: [DONE]\n\n"


def anthropic_stream_frames(frames: list[str], model: str):
    """Collected Chat Completions SSE frames as the Anthropic event sequence.

    message_start on the first chunk, content_block_start/delta/stop per text block
    or tool call, then message_delta with the stop reason and usage, then
    message_stop. The input count is only known when the provider's usage chunk
    arrives at the end, so message_start carries 0 (the wire allows an input of 0);
    message_delta carries the true output count.
    """
    started = False
    finished = False
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    pending_stop: str | None = None
    output_tokens: int | None = None

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_stream",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        output = output_tokens if output_tokens is not None else 1
        return [
            *close_events(),
            sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output},
                },
            ),
            sse_event("message_stop", {"type": "message_stop"}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal text_index, next_index, pending_stop, output_tokens
        out: list[str] = []
        for payload in _frame_data_lines(frame):
            if payload == "[DONE]":
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else "end_turn"))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get("usage")):
                output_tokens = int_or(chunk["usage"].get("completion_tokens"), output_tokens if output_tokens is not None else -1)
                if output_tokens < 0:
                    output_tokens = None
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            out.extend(start_events())
            if isinstance(delta.get("content"), str) and delta["content"] != "":
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": text_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    )
                out.append(
                    sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_index,
                            "delta": {"type": "text_delta", "text": delta["content"]},
                        },
                    )
                )
            calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call.get("index") if is_number(call.get("index")) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get("function") if is_record(call.get("function")) else {}
                    out.append(
                        sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": call.get("id") if isinstance(call.get("id"), str) and call.get("id") != "" else f"toolu_{index}",
                                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                                    "input": {},
                                },
                            },
                        )
                    )
                fn = call.get("function") if is_record(call.get("function")) else {}
                if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                    out.append(
                        sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                            },
                        )
                    )
            # Hold the stop reason until [DONE]: the usage chunk (which carries the
            # output count message_delta needs) arrives after the finish chunk.
            if choice.get("finish_reason") is not None:
                pending_stop = _stop_reason_of(choice.get("finish_reason"))
        return out

    for frame in frames:
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else "end_turn"):
            yield event


# ---------------------------------------------------------------- the gateway


class Upstream:
    """One gateway answer: its status and a body read whole or as byte chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        reader = getattr(self._response, "read1", None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode("utf-8", "replace")
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass


def gateway_url() -> str:
    """The gateway's Chat Completions route from $OPENAI_BASE_URL."""
    if GATEWAY_ROOT == "":
        return ""
    return f"{GATEWAY_ROOT}/chat/completions" if GATEWAY_ROOT.endswith("/v1") else f"{GATEWAY_ROOT}/v1/chat/completions"


def dial(model: str, payload: Json, stream: bool) -> Upstream:
    """POST one request to the gateway, naming the model this entry dialled."""
    url = gateway_url()
    if url == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    data = json.dumps({**payload, "model": model}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": f"Bearer {GATEWAY_KEY}"}
    if stream:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=GATEWAY_CALL_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as the provider sent it.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def normalize_openai_body(body: Json) -> Json:
    """A Chat Completions body ready to dial: an output cap, and usage on streams."""
    out = {k: v for k, v in body.items() if k != "model"}
    cap = out.get("max_tokens")
    if not is_number(cap) or cap <= 0:
        out["max_tokens"] = 512
    elif cap > 8192:
        out["max_tokens"] = 8192
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def serve_with_failover(
    body: Json, gateway_body: Json, stream: bool, deadline: float
) -> tuple[str | None, object | None, list[Json]]:
    """Dial the best eligible model, failing over down the rating order.

    A refusal (4xx/5xx) fails over, and so does a 200 that carried no answer: a
    model that burned its output cap reasoning and returned nothing has not
    answered the prompt. Returns (model id, answer, trail rows) where the answer
    is the parsed completion (JSON) or the collected SSE frames (stream), or
    (None, None, trail rows) when no model could serve the prompt. A model that
    refused or answered empty is cooled down for the rest of the trial.
    """
    requires = requirements_from_body(body)
    failures: list[Json] = []
    tried: set[str] = set()
    for _ in range(MAX_ATTEMPTS_PER_PROMPT):
        with REFUSED_LOCK:
            skip = set(REFUSED) | tried
        candidates = ranked_candidates(requires, skip)
        if not candidates:
            break
        model = candidates[0]["id"]
        tried.add(model)
        try:
            upstream = dial(model, gateway_body, stream)
        except Exception as err:
            # The gateway itself could not be reached: record it and try the next model.
            failures.append({"status": 502, "code": "gateway_unreachable", "model": model})
            log(f"dial {model} failed: {err}")
            with REFUSED_LOCK:
                REFUSED.add(model)
            if time.monotonic() > deadline:
                break
            continue
        if not upstream.ok:
            code = error_code_of(upstream.text())
            failures.append({"status": upstream.status, "code": code, "model": model})
            log(f"dial {model} refused HTTP {upstream.status} ({code or 'no code'})")
            with REFUSED_LOCK:
                REFUSED.add(model)
            if time.monotonic() > deadline:
                break
            continue
        if stream:
            frames = collect_frames(upstream, deadline)
            if stream_has_answer(frames):
                return model, frames, failures
            failures.append({"status": 200, "code": "empty_response", "model": model})
            log(f"dial {model} answered empty (no text, no tool call)")
        else:
            answer = upstream.text()
            try:
                parsed = json.loads(answer)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if is_record(parsed):
                text, has_tools = answer_text_of_openai(parsed)
                if text.strip() != "" or has_tools:
                    return model, parsed, failures
            failures.append({"status": 200, "code": "empty_response", "model": model})
            log(f"dial {model} answered empty (no text, no tool call)")
        with REFUSED_LOCK:
            REFUSED.add(model)
        if time.monotonic() > deadline:
            break
    return None, None, failures


def trail_header(failures: list[Json]) -> dict[str, str]:
    """The failures trail header, when there is a trail to report."""
    if not failures:
        return {}
    return {FAILURES_HEADER: json.dumps(failures[:MAX_TRAIL_ROWS], separators=(",", ":"), ensure_ascii=False)}


# ---------------------------------------------------------------- the HTTP server


def anthropic_error(error_type: str, message: str) -> Json:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    return {"error": {"message": message, "type": error_type, "code": code}}


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, nothing else."""

    protocol_version = "HTTP/1.1"
    server_version = "makefun-aggregator-py/1"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    # -- plumbing -------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self.models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self.chat()
                return
            if self.command == "POST" and path == "/v1/messages":
                self.messages()
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self.send_error_body(wire, 404, "not_found", f"no route {self.command} {path}")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = "anthropic" if urlsplit(self.path).path.startswith("/v1/messages") else "openai"
                try:
                    self.send_error_body(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, extra: dict[str, str] | None = None) -> None:
        text = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str, extra: dict[str, str] | None = None) -> None:
        if wire == "anthropic":
            self.send_json(status, anthropic_error(error_type, message), extra)
        else:
            self.send_json(status, openai_error(message, error_type, None), extra)

    def stream_headers(self, extra: dict[str, str] | None = None) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        # No content-length on a stream: the close ends it for the client.
        self.send_header("connection", "close")
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object or is too large."""
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    # -- the routes -----------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = [
            {"id": m["id"], "provider": m["provider"], "display_name": m["display_name"]}
            for m in free_models()
        ]
        if self.headers.get("x-api-key") is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model."""
        body = self.read_json()
        if body is None:
            self.send_error_body("openai", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        stream = body.get("stream") is True
        deadline = time.monotonic() + PROMPT_BUDGET_SEC
        model, payload, failures = serve_with_failover(body, normalize_openai_body(body), stream, deadline)
        if model is None or payload is None:
            self.send_error_body(
                "openai",
                502,
                "upstream_error",
                "no free model could serve this prompt",
                trail_header(failures),
            )
            return
        if stream:
            self.stream_headers(trail_header(failures))
            for frame in openai_stream_frames(payload, model):
                self.write_event(frame)
            return
        parsed = payload if is_record(payload) else {}
        # The one truthful claim: the model field is the id this entry dialled.
        parsed["model"] = model
        usage = parsed.get("usage")
        if not is_record(usage):
            text, _ = answer_text_of_openai(parsed)
            parsed["usage"] = {"prompt_tokens": 0, "completion_tokens": 1 if text != "" else 0}
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices and is_record(choices[0]) and choices[0].get("finish_reason") is None:
            choices[0]["finish_reason"] = "stop"
        self.send_json(200, parsed, trail_header(failures))

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten and answered back."""
        body = self.read_json()
        if body is None:
            self.send_error_body("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        stream = body.get("stream") is True
        deadline = time.monotonic() + PROMPT_BUDGET_SEC
        model, payload, failures = serve_with_failover(body, anthropic_to_openai(body), stream, deadline)
        if model is None or payload is None:
            self.send_error_body(
                "anthropic",
                502,
                "api_error",
                "no free model could serve this prompt",
                trail_header(failures),
            )
            return
        if stream:
            self.stream_headers(trail_header(failures))
            for event in anthropic_stream_frames(payload, model):
                self.write_event(event)
            return
        parsed = payload if is_record(payload) else {}
        self.send_json(200, openai_to_anthropic(parsed, model), trail_header(failures))


# ---------------------------------------------------------------- register and poll


def register(port: int, report: Json) -> None:
    """POST /register with the port and the one catalog; retry while the env starts."""
    body = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": report},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    last_error: str | None = None
    for attempt in range(3):
        request = urllib.request.Request(
            f"{ENV_URL}/register",
            data=body.encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                status = response.status
                reply_text = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as err:
            raise RuntimeError(
                f"POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode('utf-8', 'replace')[:500]}"
            ) from err
        except Exception as err:
            last_error = str(err)
            log(f"register attempt {attempt + 1} failed: {err}")
            time.sleep(1)
            continue
        if not (200 <= status < 300):
            raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}")
        try:
            reply = json.loads(reply_text)
        except (json.JSONDecodeError, ValueError):
            reply = None
        if is_record(reply) and is_number(reply.get("requests")):
            log(f"registered: the trial holds {reply['requests']} prompts")
        return
    raise RuntimeError(f"POST {ENV_URL}/register failed: {last_error}")


def poll() -> None:
    """Poll GET /work; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f"{ENV_URL}/work", timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as a clean exit 0."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF["error"] is not None:
        log(BRIEF["error"])
    if "--list" in sys.argv[1:]:
        # The same document the register body carries, byte for byte.
        sys.stdout.write(catalog_text() + "\n")
        sys.stdout.flush()
        return
    if gateway_url() == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
        log(f"listening on 0.0.0.0:{port} with {len(free_models())} free models")
        # Serve before registering: the env asks GET /v1/models the moment it is told.
        threading.Thread(target=server.serve_forever, name="serve", daemon=True).start()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if ENV_URL == "":
            log("MAKEFUN_ENV_URL is not set: serving without a trial (no register, no poll)")
            server.serve_forever()
            return
        register(port, CATALOG)
        poll()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
