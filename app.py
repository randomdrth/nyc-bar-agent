"""Two-Drink Minimum: a tool-calling agent for happy hours and bar crawls in Manhattan.

FastAPI + LiteLLM + Gemini. Keeps the starter's /chat shape:
{"response": str, "session_id": str, "tool_calls": [{"name", "args", "result"}]}.
"""

import json
import logging
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import litellm
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import bar_data as bd
import places
from prompts import build_system_prompt
from tools import TOOLS, run_tool

# --- Config ---

MODEL = os.environ.get("MODEL", "vertex_ai/gemini-3.5-flash-lite")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")  # set on Cloud Run; locally gcloud's default is used

MAX_TOOL_ROUNDS = 8  # a crawl can take 3 to 4 rounds: search, happy hours, plan
MODEL_TIMEOUT_SECONDS = 60
MAX_HISTORY_MESSAGES = 30  # per session, trimmed at user-message boundaries
MAX_SESSIONS = 500
MAX_MESSAGE_CHARS = 2000

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
log = logging.getLogger("two_drink_minimum")
logging.basicConfig(level=logging.INFO)

bd.load()  # fail at startup, not on the first question, if the data file is missing


# --- The Harness ---

def _complete(messages: list[dict], final: bool = False):
    kwargs = {
        "model": MODEL,
        "vertex_location": VERTEX_LOCATION,
        "messages": messages,
        "tools": TOOLS,
        "timeout": MODEL_TIMEOUT_SECONDS,
    }
    if VERTEX_PROJECT:
        kwargs["vertex_project"] = VERTEX_PROJECT
    if final:
        kwargs["tool_choice"] = "none"  # answer with what was gathered, no more tool calls
    return litellm.completion(**kwargs).choices[0].message


def _parse_args(raw: str | None) -> tuple[dict | None, str | None]:
    """Return (args, error). Empty arguments mean no arguments."""
    if raw is None or not str(raw).strip():
        return {}, None
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return None, "Arguments were not valid JSON. Call the tool again with a JSON object matching its schema."
    if not isinstance(args, dict):
        return None, "Arguments must be a JSON object. Call the tool again with an object matching its schema."
    return args, None


def run_agent(history: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Complete until the model answers without asking for a tool.

    Works on a copy of the history, so a failure leaves the caller's history
    untouched. Returns (final text, tool call records, updated history).
    """
    messages = list(history)
    tool_calls: list[dict] = []
    system = {"role": "system", "content": build_system_prompt()}

    for _ in range(MAX_TOOL_ROUNDS):
        reply = _complete([system] + messages)
        # model_dump() keeps it a plain dict: the raw object carries provider-specific
        # fields that trip Pydantic when LiteLLM re-serializes it next round.
        messages.append(reply.model_dump())

        if not reply.tool_calls:
            return reply.content or "Sorry, I didn't get an answer. Try rephrasing.", tool_calls, messages

        # The harness, not the model, runs each tool and appends the result
        for call in reply.tool_calls:
            name = call.function.name
            args, error = _parse_args(call.function.arguments)
            if error:
                result = json.dumps({"error": error})
                args = {"_unparsed_arguments": str(call.function.arguments)[:500]}
            else:
                result = run_tool(name, args)
            tool_calls.append({"name": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Out of rounds: one last call with tools off, so the user still gets an answer
    reply = _complete([system] + messages, final=True)
    messages.append(reply.model_dump())
    text = reply.content or "Sorry, that took too many steps. Try a narrower question."
    return text, tool_calls, messages


# --- Session Store ---

# session_id -> conversation (no system prompt; it is rebuilt each request). In-memory, single process.
_sessions: "OrderedDict[str, list[dict]]" = OrderedDict()
_session_locks: dict[str, threading.Lock] = {}
_store_lock = threading.Lock()


def _session_lock(session_id: str) -> threading.Lock:
    """Get (or create) the lock for a session, evicting the least recently used beyond MAX_SESSIONS."""
    with _store_lock:
        if session_id in _sessions:
            _sessions.move_to_end(session_id)
        else:
            _sessions[session_id] = []
            _session_locks[session_id] = threading.Lock()
            while len(_sessions) > MAX_SESSIONS:
                oldest, _ = _sessions.popitem(last=False)
                _session_locks.pop(oldest, None)
        return _session_locks[session_id]


def trim_history(history: list[dict], limit: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    """Keep about the last `limit` messages, cutting only where a user message starts.

    Cutting anywhere else could separate a tool call from its result, which the model rejects.
    """
    if len(history) <= limit:
        return history
    cut = len(history) - limit
    while cut < len(history) and history[cut].get("role") != "user":
        cut += 1
    if cut >= len(history):  # one very long turn: keep it whole, from its user message
        user_positions = [i for i, m in enumerate(history) if m.get("role") == "user"]
        cut = user_positions[-1] if user_positions else 0
    return history[cut:]


# --- FastAPI App ---

app = FastAPI(title="Two-Drink Minimum")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    message = (request.message or "").strip()
    if not message:
        return ChatResponse(response="Ask me about happy hours or a bar night in the West Village, "
                                     "East Village, or Upper West Side.", session_id=session_id, tool_calls=[])
    if len(message) > MAX_MESSAGE_CHARS:
        return ChatResponse(response=f"That message is too long. Keep it under {MAX_MESSAGE_CHARS} characters.",
                            session_id=session_id, tool_calls=[])

    with _session_lock(session_id):  # one turn at a time per session
        history = trim_history(_sessions.get(session_id, []))
        history = history + [{"role": "user", "content": message}]
        try:
            response, tool_calls, updated = run_agent(history)
        except Exception as e:
            # Auth, quota, a model outage: the session is left exactly as before this message
            log.exception("Turn failed for session %s", session_id)
            return ChatResponse(
                response=f"Sorry, I couldn't reach the model just now ({type(e).__name__}). "
                         f"Your message wasn't saved, so you can send it again.",
                session_id=session_id, tool_calls=[])
        with _store_lock:
            _sessions[session_id] = updated
            _sessions.move_to_end(session_id)

    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    with _store_lock:
        _sessions.pop(session_id, None)
        _session_locks.pop(session_id, None)
    return {"status": "ok"}


@app.get("/health")
def health():
    bars = bd.load()["bars"]
    return {
        "status": "ok",
        "bars_loaded": len(bars),
        "bars_with_happy_hours": sum(1 for b in bars if b.get("deals")),
        "places_key_set": bool(os.environ.get("PLACES_API_KEY")),
        "places_calls_today": places.usage_today(),
        "model": MODEL,
        "active_sessions": len(_sessions),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))
