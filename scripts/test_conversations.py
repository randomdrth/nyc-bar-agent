"""Run scripted conversations against the real agent (Gemini + live tools) and print every step.

Usage:
    uv run scripts/test_conversations.py

Uses a few Places searches and about 15 Gemini calls. Each conversation runs in
its own session, the same way the browser does.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as agent  # noqa: E402

CONVERSATIONS = [
    ("Happy hours, then a follow-up that relies on memory", [
        "What happy hours are on in the East Village Thursday at 6pm?",
        "Only wine bars, please.",
    ]),
    ("A crawl plan", [
        "Plan 3 stops in the West Village Friday starting at 5pm. I want cocktails.",
    ]),
    ("Out of scope", [
        "Any happy hours in SoHo tonight?",
    ]),
    ("Vibe search", [
        "Find me a dive bar on the Upper West Side that's open late Friday.",
    ]),
    ("A brand new session should remember nothing", [
        "What did I just ask you about?",
    ]),
]


def summarize(result: str) -> str:
    try:
        r = json.loads(result)
    except json.JSONDecodeError:
        return result[:120]
    if "error" in r and "stops" not in r:
        return f"ERROR: {r['error'][:140]}"
    if "happening_now" in r:
        return f"{r['counts']['happening_now']} happening now, {r['counts']['starting_soon']} starting soon"
    if "results" in r:
        return f"source={r['source']}, {r['counts']['shown']} shown of {r['counts']['matched']} matched"
    if "stops" in r:
        s = r["summary"]
        return f"{s['stops']} stops {s['starts']} to {s['ends']}, {s['happy_hour_minutes']} min of happy hour"
    if "status" in r:
        return f"status={r['status']}"
    return result[:120]


def run() -> None:
    for title, turns in CONVERSATIONS:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        session_id = None
        for message in turns:
            print(f"\nUSER: {message}")
            start = time.time()
            reply = agent.chat(agent.ChatRequest(message=message, session_id=session_id))
            session_id = reply.session_id
            for call in reply.tool_calls:
                args = json.dumps(call["args"])
                print(f"  TOOL {call['name']}({args[:110]}) -> {summarize(call['result'])}")
            print(f"AGENT ({time.time() - start:.1f}s):\n{reply.response}")


if __name__ == "__main__":
    run()
