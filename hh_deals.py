"""Turn a bar's deal snippets into structured, checked happy hour deals.

Gemini reads the snippets and copies the deal text exactly as written. Code
then does all the interpreting (times, days) and checks every piece against
the snippets, so a deal only counts as verified if the page really says it.

Used by scripts/build_hh_dataset.py (offline build) and later by the live
check_happy_hour_online tool in tools.py.
"""

import hashlib
import json
import os
import re
import threading
import time
from datetime import date
from pathlib import Path

MODEL = "vertex_ai/gemini-3.5-flash-lite"
VERTEX_LOCATION = "global"
PROMPT_VERSION = "v3"  # bump when SYSTEM_PROMPT changes, so cached answers are not reused

DAY_CODES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

SYSTEM_PROMPT = """You extract happy hour deals for New York City bars from text scraped from each bar's own website.

Return JSON only, with this shape:
{
  "deals": [
    {
      "label": "short name, e.g. Weekday happy hour or Half price Mondays",
      "days_text": "the exact words that say which days, e.g. Monday through Friday, Mon-Thu, Daily, Monday",
      "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
      "start_text": "the exact words for the start time, e.g. 3pm, 4, 4:30, open",
      "end_text": "the exact words for the end time, e.g. 7 PM, close, 2am",
      "specials": [
        {"item": "what is discounted", "category": "drink or food", "price": 6, "discount": "half price, $2 off, or null"}
      ],
      "conditions": "short restriction copied from the text, e.g. bar only, some exclusions, or null",
      "evidence": "one exact, contiguous quote from ONE snippet that contains the days, the times, and the deal",
      "snippet_id": "the id of the snippet the evidence comes from, e.g. S1",
      "multi_location": false,
      "possibly_outdated": false
    }
  ],
  "notes": "one short sentence on what you found, or why there is no deal"
}

Rules:
1. Include recurring, time-bound happy hours and drink deals. A happy hour with stated days and times counts even if no prices or specials are listed. Return it with "specials": []. If a deal also discounts food, keep it and list each item with its category.
2. Skip regular menu prices, food-only specials, one-off events, holiday events, private events, and opening hours that have no deal. Skip brunch entirely, including brunch drink specials and bottomless brunch.
3. Copy days_text, start_text, end_text, conditions, and evidence exactly as they appear in the text. Do not fix spelling, reformat times, or add words. Only whitespace may differ.
4. evidence must come from a single snippet, stay under 400 characters, and include days_text, start_text, end_text, and the main specials. Use one contiguous quote when you can. If the days, times, and deal are not next to each other (for example one time heading above several days, or a date above several events), copy each needed piece exactly and join the pieces with " ... " in the order they appear in the snippet. Never join pieces that belong to different days or different deals.
5. If a deal starts at a time and has no end time (for example "after 7PM" or "from 9pm"), set start_text to that phrase exactly and end_text to "".
6. If one day has two time windows (e.g. 3pm-8pm and 8pm-close), return two deals.
7. If the same window applies to several days, return one deal listing all of those days.
8. Set multi_location to true if the text covers several locations and you cannot tell the deal applies to this bar.
9. Set possibly_outdated to true if the text ties the deal to a date, year, or season that has passed.
10. Never guess. If the days or times are not stated, do not return the deal.
11. If nothing meets these rules, return {"deals": [], "notes": "..."}.

Example input:
Bar: Example Tavern
[S1] source: https://example.com/specials/
Specials
Monday
3pm-8pm
1/2 Price whole bar *some exclusions*
$1 Wings
Mon - Fri 4-7 PM $6 Drafts, $8 House Wine, $5 Sliders
Brunch Sat & Sun 11am-4pm
[S2] source: https://example.com/events/
Daily Specials 5PM-9PM
TUESDAY: $5 Margaritas
WEDNESDAY: Trivia Night
[S3] source: https://example.com/
Happy Hour Sun - Thu 2-6pm

Example output:
{
  "deals": [
    {
      "label": "Half price Mondays",
      "days_text": "Monday",
      "days": ["mon"],
      "start_text": "3pm",
      "end_text": "8pm",
      "specials": [
        {"item": "whole bar", "category": "drink", "price": null, "discount": "half price"},
        {"item": "wings", "category": "food", "price": 1, "discount": null}
      ],
      "conditions": "some exclusions",
      "evidence": "Monday 3pm-8pm 1/2 Price whole bar *some exclusions* $1 Wings",
      "snippet_id": "S1",
      "multi_location": false,
      "possibly_outdated": false
    },
    {
      "label": "Weekday happy hour",
      "days_text": "Mon - Fri",
      "days": ["mon", "tue", "wed", "thu", "fri"],
      "start_text": "4",
      "end_text": "7 PM",
      "specials": [
        {"item": "drafts", "category": "drink", "price": 6, "discount": null},
        {"item": "house wine", "category": "drink", "price": 8, "discount": null},
        {"item": "sliders", "category": "food", "price": 5, "discount": null}
      ],
      "conditions": null,
      "evidence": "Mon - Fri 4-7 PM $6 Drafts, $8 House Wine, $5 Sliders",
      "snippet_id": "S1",
      "multi_location": false,
      "possibly_outdated": false
    },
    {
      "label": "Tuesday margaritas",
      "days_text": "TUESDAY",
      "days": ["tue"],
      "start_text": "5PM",
      "end_text": "9PM",
      "specials": [
        {"item": "margaritas", "category": "drink", "price": 5, "discount": null}
      ],
      "conditions": null,
      "evidence": "Daily Specials 5PM-9PM ... TUESDAY: $5 Margaritas",
      "snippet_id": "S2",
      "multi_location": false,
      "possibly_outdated": false
    },
    {
      "label": "Happy hour",
      "days_text": "Sun - Thu",
      "days": ["sun", "mon", "tue", "wed", "thu"],
      "start_text": "2",
      "end_text": "6pm",
      "specials": [],
      "conditions": null,
      "evidence": "Happy Hour Sun - Thu 2-6pm",
      "snippet_id": "S3",
      "multi_location": false,
      "possibly_outdated": false
    }
  ],
  "notes": "Half price Mondays, a weekday happy hour, Tuesday margaritas, and a Sunday to Thursday happy hour with no specials listed. Brunch and trivia skipped."
}"""


# --- Prompt + Gemini call ---

def build_user_prompt(name: str, address: str, snippets: list[dict]) -> str:
    lines = [f"Today's date: {date.today().isoformat()}", f"Bar: {name}", f"Address: {address}", ""]
    for i, s in enumerate(snippets, 1):
        lines.append(f"[S{i}] source: {s['url']}")
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)


def _parse_json_object(content: str) -> dict:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def call_gemini(user_prompt: str, cache_dir: Path | None = None, retries: int = 2) -> dict:
    """Return {"ok", "data", "error"}. Successful answers are cached on disk."""
    key = hashlib.sha1((PROMPT_VERSION + MODEL + SYSTEM_PROMPT + user_prompt).encode()).hexdigest()
    path = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if path and path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass

    import logging

    import litellm  # imported here so the rest of this module stays light

    logging.getLogger("LiteLLM").setLevel(logging.ERROR)  # hide the temperature deprecation notice

    kwargs = {}
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        kwargs["vertex_project"] = os.environ["GOOGLE_CLOUD_PROJECT"]

    last_error = None
    for attempt in range(retries + 1):
        try:
            reply = litellm.completion(
                model=MODEL,
                vertex_location=VERTEX_LOCATION,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                timeout=60,
                **kwargs,
            )
            data = _parse_json_object(reply.choices[0].message.content or "")
            result = {"ok": True, "data": data, "error": None}
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(result))
                os.replace(tmp, path)
            return result
        except (ValueError, json.JSONDecodeError) as e:
            last_error = f"bad JSON from model: {e}"
        except Exception as e:  # API, auth, quota: report it, never crash the run
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    return {"ok": False, "data": None, "error": last_error}


# --- Text normalization for quote checks ---

def norm(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)
    s = s.replace("\u2026", "...")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\$\s+", "$", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    return s.strip()


def squash(s: str | None) -> str:
    """norm() with all whitespace removed, for sites that split words across lines."""
    return re.sub(r"\s+", "", norm(s))


_FULL_DAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\s*\|\s*|\*{2,}|(?<=[ap]m)\s+|\s+(?=\$\d)")


def _pieces_in_order(parts: list[str], text: str, max_gap: int, deal_days: set[str] | None):
    """Find parts in order. Returns None, "ok", or "crosses_days".

    "crosses_days" means that after the quote reached its day heading, it skipped
    over text naming a different day, so the pieces may come from another day's
    section of the page.
    """
    pos, crosses, seen_day = None, False, False
    for part in parts:
        found = text.find(part, pos or 0)
        if found == -1 or (pos is not None and found - pos > max_gap):
            return None
        if pos is not None and seen_day and deal_days is not None:
            skipped = {m.group(1)[:3] for m in _FULL_DAY.finditer(text[pos:found])}
            if skipped - deal_days:
                crosses = True
        if _FULL_DAY.search(part):
            seen_day = True
        pos = found + len(part)
    return "crosses_days" if crosses else "ok"


def quote_match(quote: str, text: str, deal_days: set[str] | None = None, max_gap: int = 1000) -> str | None:
    """How the quote matches the text: "exact", "pieced", "crosses_days", or None.

    1. Pieces joined with '...' must appear in order, each within max_gap
       characters of the previous one.
    2. A single quote that fails is retried ignoring all whitespace (some sites
       split words across lines).
    3. If that fails, the quote is split at natural boundaries (sentence ends,
       '|', '*****', after am/pm, before prices) and the pieces are searched in
       order, because the model sometimes quotes text that is not contiguous.
    """
    q = norm(quote)
    if not q:
        return None
    parts = [p.strip() for p in q.split("...") if p.strip()]
    if not parts or any(len(p) < 4 for p in parts):
        return None
    t = norm(text)

    result = _pieces_in_order(parts, t, max_gap, deal_days)
    if result:
        return "exact" if len(parts) == 1 else ("pieced" if result == "ok" else "crosses_days")

    if len(parts) == 1 and squash(parts[0]) in squash(text):
        return "exact"

    sub = [s.strip() for p in parts for s in _BOUNDARY.split(p) if s and len(s.strip()) >= 3]
    if len(sub) > len(parts):
        result = _pieces_in_order(sub, t, max_gap, deal_days)
        if result:
            return "pieced" if result == "ok" else "crosses_days"
    return None


def quote_in_text(quote: str, text: str, max_gap: int = 1000) -> bool:
    return quote_match(quote, text, None, max_gap) is not None


def contains(needle: str | None, haystack: str | None) -> bool:
    """Substring check that tolerates whitespace differences."""
    n = norm(needle)
    return bool(n) and (n in norm(haystack) or squash(needle) in squash(haystack))


# --- Time parsing ---

_TIME_TOKEN = re.compile(
    r"(?<![\d:])(\d{1,2})(?::(\d{2}))?(?:\s*(a\.m\.|p\.m\.|am|pm|a|p)(?![a-z]))?",
    re.IGNORECASE,
)
_OPEN_WORDS = re.compile(r"\b(open|opening|all\s*day|all\s*night)\b", re.IGNORECASE)
_UNTIL = re.compile(r"\b(until|till|til)\b", re.IGNORECASE)
_OPEN_ENDED = re.compile(r"\b(after|from|starting|starts)\b", re.IGNORECASE)
_CLOSE_WORDS = re.compile(r"\b(close|closing|cl|late|end of night|all\s*day|all\s*night)\b", re.IGNORECASE)


def _parse_one(text: str, role: str):
    """Return ("special", "open"|"close"), ("time", minutes_candidates), or None."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if "noon" in t:
        return ("time", [12 * 60])
    if "midnight" in t:
        return ("time", [0])
    m = _TIME_TOKEN.search(t)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        mer = (m.group(3) or "").replace(".", "")
        if minute > 59 or hour > 23:
            return None
        if hour > 12:  # already 24 hour
            return ("time", [hour * 60 + minute])
        if mer.startswith("a"):
            return ("time", [(hour % 12) * 60 + minute])
        if mer.startswith("p"):
            return ("time", [(hour % 12 + 12) * 60 + minute])
        if hour == 0:
            return ("time", [minute])
        # No am/pm written: both readings are possible, decide with the other end
        return ("time", sorted({(hour % 12) * 60 + minute, (hour % 12 + 12) * 60 + minute}))
    if role == "start" and _OPEN_WORDS.search(t):
        return ("special", "open")
    if role == "end" and _CLOSE_WORDS.search(t):
        return ("special", "close")
    return None


def _fmt(minutes: int) -> str:
    minutes %= 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def parse_time_range(start_text: str, end_text: str) -> dict:
    """Return {"start", "end", "ambiguous", "error"}. start/end are "HH:MM", "open", or "close".

    With no am/pm written, prefer readings where the window is under 12 hours and
    starts at 11am or later, since bar deals rarely start in the morning.
    """
    if not (end_text or "").strip() and _OPEN_ENDED.search(start_text or ""):
        end_text = "close"  # "after 7PM" with no end time runs until close
    s = _parse_one(start_text, "start")
    e = _parse_one(end_text, "end")
    if s is None:
        return {"start": None, "end": None, "ambiguous": False, "error": f"could not read start time '{start_text}'"}
    if e is None:
        return {"start": None, "end": None, "ambiguous": False, "error": f"could not read end time '{end_text}'"}

    if s[0] == "special" and e[0] == "special":
        return {"start": "open", "end": "close", "ambiguous": False, "error": None}

    if s[0] == "special":  # "open" to a time
        cands = e[1]
        if len(cands) == 1:
            return {"start": "open", "end": _fmt(cands[0]), "ambiguous": False, "error": None}
        pm = max(cands)
        return {"start": "open", "end": _fmt(pm), "ambiguous": (pm // 60) - 12 <= 4, "error": None}

    if e[0] == "special":  # a time to "close"
        cands = s[1]
        if len(cands) == 1:
            return {"start": _fmt(cands[0]), "end": "close", "ambiguous": False, "error": None}
        preferred = [c for c in cands if c >= 11 * 60]
        pick = preferred[0] if preferred else max(cands)
        return {"start": _fmt(pick), "end": "close", "ambiguous": len(preferred) > 1, "error": None}

    combos = []
    for st in s[1]:
        for en in e[1]:
            duration = (en - st) % 1440
            if 0 < duration <= 12 * 60:
                combos.append((st, en, duration))
    if not combos:
        return {"start": None, "end": None, "ambiguous": False,
                "error": f"no sensible window for '{start_text}' to '{end_text}'"}

    preferred = [c for c in combos if c[0] >= 11 * 60]
    pool = preferred or combos
    pool.sort(key=lambda c: (c[2], -c[0]))
    st, en, _ = pool[0]
    return {"start": _fmt(st), "end": _fmt(en), "ambiguous": len(pool) > 1, "error": None}


# --- Day parsing ---

_DAY_TOKEN = re.compile(
    r"\b(monday|mon|tuesday|tues|tue|wednesday|weds|wed|thursday|thurs|thur|thu"
    r"|friday|fri|saturday|sat|sunday|sun)s?\b",
    re.IGNORECASE,
)
_RANGE_SEP = re.compile(r"^\s*(-|to|through|thru|til|till|until)\s*$", re.IGNORECASE)
_ALL_DAYS = re.compile(r"\b(daily|every\s*day|everyday|7 days|seven days|all week)\b", re.IGNORECASE)
_EXCEPT = re.compile(r"\b(except|excluding|but not|other than)\b", re.IGNORECASE)


def _day_code(token: str) -> str:
    return token.lower()[:3]


def _expand_range(a: str, b: str) -> list[str]:
    i, j = DAY_CODES.index(a), DAY_CODES.index(b)
    if i <= j:
        return DAY_CODES[i:j + 1]
    return DAY_CODES[i:] + DAY_CODES[:j + 1]  # wraps, e.g. Fri-Sun or Sun-Wed


def parse_days(days_text: str | None) -> set[str] | None:
    """Parse "Mon-Fri", "Monday through Thursday", "Tues & Thurs", "Daily", "weekdays".

    Returns None if nothing recognizable is found, so the check is skipped
    rather than failed on wording we do not understand.
    """
    if not days_text:
        return None
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", days_text)

    split = _EXCEPT.split(text, maxsplit=1)
    if len(split) == 3:
        base = parse_days(split[0]) or set(DAY_CODES)
        removed = parse_days(split[2]) or set()
        return (base - removed) or None

    found: set[str] = set()
    if _ALL_DAYS.search(text):
        found |= set(DAY_CODES)
    if re.search(r"\bweekdays?\b", text, re.IGNORECASE):
        found |= set(DAY_CODES[:5])
    if re.search(r"\bweekends?\b", text, re.IGNORECASE):
        found |= {"sat", "sun"}

    tokens = list(_DAY_TOKEN.finditer(text))
    i = 0
    while i < len(tokens):
        current = _day_code(tokens[i].group(1))
        if i + 1 < len(tokens):
            between = text[tokens[i].end():tokens[i + 1].start()]
            if _RANGE_SEP.match(between):
                found |= set(_expand_range(current, _day_code(tokens[i + 1].group(1))))
                i += 2
                continue
        found.add(current)
        i += 1

    return found or None


# --- Validation ---

_HH_WORDS = re.compile(r"happy[\s\-]*h(ou)?r|\bhh\b", re.IGNORECASE)
# A special counts as a drink deal if its item names a drink, even if the model tagged it food
# (e.g. "burger, fries, and beer combo").
_DRINK_WORDS = re.compile(
    r"\b(beers?|wines?|cocktails?|shots?|drafts?|draughts?|pints?|pitchers?|buckets?|ciders?|ales?|lagers?|ipas?"
    r"|margaritas?|martinis?|spritz(es)?|sangrias?|mimosas?|prosecco|bubbles|champagne|cava|rosé|rose"
    r"|wells?|spirits?|whiske?ys?|bourbon|tequila|mezcal|vodka|gin|rum|seltzers?|cans|bottles?|glass(es)?|drinks?)\b",
    re.IGNORECASE,
)


def _to_price(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d+(\.\d+)?", str(value))
    return float(m.group()) if m else None


def _clean_specials(raw) -> list[dict]:
    specials = []
    for sp in raw if isinstance(raw, list) else []:
        if not isinstance(sp, dict) or not str(sp.get("item") or "").strip():
            continue
        category = str(sp.get("category") or "").lower().strip()
        specials.append({
            "item": str(sp["item"]).strip(),
            "category": category if category in ("drink", "food") else "drink",
            "price": _to_price(sp.get("price")),
            "discount": (str(sp["discount"]).strip() or None) if sp.get("discount") else None,
        })
    return specials


def _special_in_text(sp: dict, text: str) -> bool:
    if contains(sp["item"], text):
        return True
    price = sp.get("price")
    if isinstance(price, (int, float)):
        flat = squash(text)
        candidates = {f"${price:g}", f"${price:.2f}"}
        return any(c in flat for c in candidates)
    return False


def validate_deal(raw: dict, snippets: list[dict]) -> dict | None:
    """Check one deal from Gemini against the snippets.

    Returns a deal dict with "status" (verified / needs_review / rejected /
    food_only) and "failed_checks", or None if the deal is malformed.
    """
    if not isinstance(raw, dict):
        return None
    evidence = str(raw.get("evidence") or "").strip()
    days_text = str(raw.get("days_text") or "").strip()
    start_text = str(raw.get("start_text") or "").strip()
    end_text = str(raw.get("end_text") or "").strip()
    label = str(raw.get("label") or "").strip() or "Happy hour"
    conditions = str(raw.get("conditions") or "").strip() or None
    specials = _clean_specials(raw.get("specials"))

    gemini_days = {d.lower()[:3] for d in raw.get("days") or [] if str(d).lower()[:3] in DAY_CODES}
    parsed_days = parse_days(days_text)
    days = parsed_days or gemini_days

    # Which snippet does the quote come from? Prefer the claimed id, then the best match anywhere.
    by_id = {f"S{i}": s for i, s in enumerate(snippets, 1)}
    claimed = by_id.get(str(raw.get("snippet_id") or "").strip().upper())
    ordered = ([claimed] if claimed else []) + [s for s in snippets if s is not claimed]
    rank = {"exact": 0, "pieced": 1, "crosses_days": 2}
    source, match = None, None
    for s in ordered:
        m = quote_match(evidence, s["text"], days or None)
        if m and (match is None or rank[m] < rank[match]):
            source, match = s, m
            if m == "exact":
                break

    failed: list[str] = []
    if source is None:
        failed.append("quote_not_found")
    elif match == "crosses_days":
        failed.append("quote_crosses_days")

    # "after 7PM" has no end time, "until 7 pm" has no start time
    open_ended = not end_text and bool(_OPEN_ENDED.search(start_text))
    open_start = (not start_text or norm(start_text) == "open") and bool(_UNTIL.search(evidence))
    if open_start:
        start_text = "open"
    if not open_start and (not start_text or not contains(start_text, evidence)):
        failed.append("times_not_in_quote")
    elif not open_ended and (not end_text or not contains(end_text, evidence)):
        failed.append("times_not_in_quote")

    times = parse_time_range(start_text, end_text)
    if times["error"]:
        failed.append("time_unreadable")
    elif times["ambiguous"]:
        failed.append("time_ambiguous")

    if not days_text or not any(contains(days_text, s["text"]) for s in snippets):
        failed.append("days_text_not_found")
    if parsed_days is not None and gemini_days and parsed_days != gemini_days:
        failed.append("days_mismatch")
    if not days:
        failed.append("no_days")

    if raw.get("multi_location") is True:
        failed.append("multi_location")
    if raw.get("possibly_outdated") is True:
        failed.append("possibly_outdated")

    # Most specials should be traceable to the source text, by item name or price
    if specials and source is not None:
        supported = sum(1 for sp in specials if _special_in_text(sp, source["text"]))
        if supported * 2 < len(specials):
            failed.append("specials_not_in_text")

    has_drink = any(sp["category"] == "drink" or _DRINK_WORDS.search(sp["item"]) for sp in specials)
    has_food = any(sp["category"] == "food" for sp in specials)
    says_hh = bool(_HH_WORDS.search(f"{label} {evidence}"))
    if not specials and not says_hh:
        failed.append("no_specials_listed")

    if source is None:
        status = "rejected"
    elif specials and not has_drink:
        status = "food_only"
    elif failed:
        status = "needs_review"
    else:
        status = "verified"

    return {
        "label": label,
        "days": [d for d in DAY_CODES if d in days],
        "start": times["start"],
        "end": times["end"],
        "specials": specials,
        "conditions": conditions,
        "has_food": has_food,
        "days_text": days_text,
        "start_text": start_text,
        "end_text": end_text,
        "evidence": evidence,
        "source_url": source["url"] if source else None,
        "status": status,
        "failed_checks": failed,
    }


def bar_status(deals: list[dict]) -> str:
    statuses = {d["status"] for d in deals}
    if "verified" in statuses:
        return "verified"
    if "needs_review" in statuses:
        return "needs_review"
    if "rejected" in statuses:
        return "rejected"
    return "no_deal"


def extract_deals(name: str, address: str, snippets: list[dict], cache_dir: Path | None = None) -> dict:
    """Run Gemini on one bar's snippets and validate every deal it returns."""
    if not snippets:
        return {"llm_ok": True, "llm_error": None, "notes": "no snippets", "deals": [],
                "food_only_dropped": 0, "malformed": 0, "bar_status": "no_deal"}

    answer = call_gemini(build_user_prompt(name, address, snippets), cache_dir=cache_dir)
    if not answer["ok"]:
        return {"llm_ok": False, "llm_error": answer["error"], "notes": None, "deals": [],
                "food_only_dropped": 0, "malformed": 0, "bar_status": "llm_error"}

    data = answer["data"]
    raw_deals = data.get("deals") if isinstance(data.get("deals"), list) else []
    deals, food_only, malformed = [], 0, 0
    for raw in raw_deals:
        deal = validate_deal(raw, snippets)
        if deal is None:
            malformed += 1
        elif deal["status"] == "food_only":
            food_only += 1
        else:
            deals.append(deal)

    return {
        "llm_ok": True,
        "llm_error": None,
        "notes": str(data.get("notes") or "")[:300],
        "deals": deals,
        "food_only_dropped": food_only,
        "malformed": malformed,
        "bar_status": bar_status(deals),
    }
