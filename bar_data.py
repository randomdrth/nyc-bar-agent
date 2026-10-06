"""Shared data and time logic for the Next Round tools.

Time model: every moment in the week is a "week minute", minutes since Monday
00:00 (0 to 10,079). Opening hours and happy hour windows become intervals of
week minutes. An interval may run past the end of the week (Saturday night into
Sunday morning), so containment checks also try the time shifted by one week.
"""

import difflib
import json
import math
import re
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

DATA_FILE = Path(__file__).parent / "data" / "happy_hours.json"
NYC = ZoneInfo("America/New_York")

WEEK = 7 * 1440
DAY_CODES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

WALK_METERS_PER_MIN = 80  # about 3 mph
STREET_FACTOR = 1.3  # streets are longer than a straight line

PRICE_LEVELS = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}

NEIGHBORHOOD_ALIASES = {
    "west village": "west_village", "west_village": "west_village", "wv": "west_village",
    "w village": "west_village", "the west village": "west_village",
    "east village": "east_village", "east_village": "east_village", "ev": "east_village",
    "e village": "east_village", "the east village": "east_village",
    "upper west side": "upper_west_side", "upper_west_side": "upper_west_side", "uws": "upper_west_side",
    "upper west": "upper_west_side", "the upper west side": "upper_west_side",
}


class ToolError(Exception):
    """An error message written for the model: what went wrong and what to do instead."""


# --- Loading ---

@lru_cache(maxsize=1)
def load() -> dict:
    data = json.loads(DATA_FILE.read_text())
    bars = data["bars"]
    return {
        "bars": bars,
        "by_id": {b["place_id"]: b for b in bars},
        "neighborhoods": data["neighborhoods"],  # key -> display name
    }


# Bars returned live by find_bars that are not in our dataset, so the planner can route through them.
_live_places: dict[str, dict] = {}


def register_live_place(place: dict) -> None:
    """Remember a bar from a live search so the other tools can use it. Keeps deals found earlier."""
    pid = place.get("place_id")
    if not pid or pid in load()["by_id"]:
        return
    existing = _live_places.get(pid, {})
    _live_places[pid] = {**place, "deals": existing.get("deals", []), "pending_deals": []}


def add_live_deals(place_id: str, deals: list[dict]) -> bool:
    """Attach deals verified live by check_happy_hour_online. Curated deals are never replaced.

    Returns True if the deals were attached.
    """
    bar = get_bar(place_id)
    if bar is None or not deals:
        return False
    if bar.get("deals") and not any(d.get("live") for d in bar["deals"]):
        return False  # curated, already verified
    bar["deals"] = [{**d, "live": True} for d in deals]
    return True


def all_bars() -> list[dict]:
    return load()["bars"] + list(_live_places.values())


def get_bar(place_id: str) -> dict | None:
    return load()["by_id"].get(place_id) or _live_places.get(place_id)


def neighborhood_name(key: str) -> str:
    return load()["neighborhoods"].get(key, key)


def resolve_neighborhood(text: str | None) -> str:
    key = NEIGHBORHOOD_ALIASES.get(re.sub(r"\s+", " ", (text or "").strip().lower()))
    if not key:
        names = ", ".join(load()["neighborhoods"].values())
        raise ToolError(f"Neighborhood '{text}' is not covered. Supported neighborhoods: {names}.")
    return key


def price_level(bar: dict) -> int | None:
    return PRICE_LEVELS.get(bar.get("price_level"))


def price_symbol(bar: dict) -> str | None:
    level = price_level(bar)
    return "$" * level if level else None


# --- Parsing days and times ---

def now_nyc() -> datetime:
    return datetime.now(NYC)


def parse_day(text: str | None, now: datetime | None = None) -> int:
    """Return a day index, 0 = Monday. Accepts names, abbreviations, today/tonight/tomorrow, or YYYY-MM-DD."""
    now = now or now_nyc()
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    t = re.sub(r"^(on |this |next |coming |the )+", "", t)
    t = re.sub(r" (night|nite|evening|afternoon|morning|eve)$", "", t)  # "Friday night" is Friday
    if t in ("weekend", "the weekend", "this weekend"):
        raise ToolError("'weekend' covers two days. Pick 'Saturday' or 'Sunday', or call once for each.")
    if t in ("", "today", "tonight", "now"):
        return now.weekday()
    if t == "tomorrow":
        return (now.weekday() + 1) % 7
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        try:
            return datetime.strptime(t, "%Y-%m-%d").weekday()
        except ValueError:
            pass
    m = re.fullmatch(r"(mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)[a-z]*?s?", t)
    if m:
        return DAY_CODES.index(m.group(1)[:3])
    raise ToolError(f"Could not read the day '{text}'. Use a weekday name like 'Thursday', "
                    f"'today', 'tomorrow', or a date like 2026-10-08.")


def parse_time(text: str | None, now: datetime | None = None) -> tuple[int, str | None]:
    """Return (minutes after midnight, note). The note explains any assumption made."""
    now = now or now_nyc()
    raw = (text or "").strip()
    t = re.sub(r"\s+", " ", raw.lower().replace(".", ""))
    t = re.sub(r"^(at|around|about|by|from|~) ", "", t).strip()
    if t in ("", "now", "right now"):
        return now.hour * 60 + now.minute, None
    if t in TIME_PHRASES:
        minutes = TIME_PHRASES[t]
        return minutes, f"'{raw}' read as {fmt_clock(minutes)}"
    if t == "noon":
        return 12 * 60, None
    if t == "midnight":
        return 0, None
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a|p)?", t)
    if not m:
        raise ToolError(f"Could not read the time '{text}'. Use a time like '6pm', '6:30 pm', '18:00', or 'now'.")
    hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if hour > 23 or minute > 59 or (mer and hour > 12):
        raise ToolError(f"'{text}' is not a valid time. Use a time like '6pm' or '18:00'.")
    if mer:
        hour = hour % 12 + (12 if mer.startswith("p") else 0)
        return hour * 60 + minute, None
    if 1 <= hour <= 11:  # "6" at a bar means 6pm
        return (hour + 12) * 60 + minute, f"'{text}' read as {hour}:{minute:02d} PM"
    return hour * 60 + minute, None


# Everyday time words, matching the definitions in the system prompt
TIME_PHRASES = {
    "after work": 17 * 60 + 30, "afterwork": 17 * 60 + 30,
    "early evening": 17 * 60, "happy hour": 17 * 60,
    "evening": 18 * 60, "this evening": 18 * 60,
    "afternoon": 15 * 60, "this afternoon": 15 * 60,
    "night": 20 * 60, "tonight": 20 * 60,
    "late": 22 * 60, "late night": 22 * 60, "latenight": 22 * 60,
}


def week_minute(day: int, minutes: int) -> int:
    return day * 1440 + minutes


def fmt_clock(wm: int) -> str:
    minutes = wm % 1440
    hour, minute = divmod(minutes, 60)
    suffix = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}:{minute:02d} {suffix}"


def fmt_time(wm: int, base_day: int | None = None) -> str:
    """'7:00 PM', or 'Fri 1:00 AM' when the time falls on a different day than base_day."""
    day = (wm // 1440) % 7
    if base_day is None or day == base_day % 7:
        return fmt_clock(wm)
    return f"{DAY_NAMES[day][:3]} {fmt_clock(wm)}"


# --- Opening hours ---

def _google_to_wm(point: dict) -> int:
    # Google numbers days with Sunday = 0; ours start at Monday = 0
    return ((point["day"] - 1) % 7) * 1440 + point.get("hour", 0) * 60 + point.get("minute", 0)


@lru_cache(maxsize=4096)
def _intervals_cached(periods_json: str) -> tuple[tuple[int, int], ...] | None:
    periods = json.loads(periods_json)
    if not periods:
        return None
    out = []
    for p in periods:
        if "open" not in p:
            continue
        if "close" not in p:  # Google's way of saying open 24 hours
            return ((0, WEEK),)
        start, end = _google_to_wm(p["open"]), _google_to_wm(p["close"])
        if end <= start:
            end += WEEK  # crosses the end of the week (or a full 24h day ending at the same time)
        out.append((start, end))
    return tuple(sorted(out)) or None


def open_intervals(bar: dict) -> tuple[tuple[int, int], ...] | None:
    """Opening intervals in week minutes, or None if hours are unknown."""
    return _intervals_cached(json.dumps(bar.get("hours_periods") or [], sort_keys=True))


def containing_interval(intervals, t: int) -> tuple[int, int] | None:
    """The opening interval that contains time t, shifted so start <= t < end. None if closed."""
    for start, end in intervals:
        for shift in (0, WEEK, -WEEK):
            if start <= t + shift < end:
                return start - shift, end - shift
    return None


def next_opening(intervals, t: int) -> int | None:
    """Week minute (>= t) of the next time the bar opens."""
    best = None
    for start, _ in intervals:
        for shift in (-WEEK, 0, WEEK, 2 * WEEK):
            s = start + shift
            if s >= t and (best is None or s < best):
                best = s
    return best


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> tuple[int, int] | None:
    start, end = max(a_start, b_start), min(a_end, b_end)
    return (start, end) if start < end else None


def _clip_to_hours(window: tuple[int, int], intervals) -> list[tuple[int, int]]:
    pieces = []
    for start, end in intervals:
        for shift in (-WEEK, 0, WEEK):
            piece = _overlap(window[0], window[1], start + shift, end + shift)
            if piece:
                pieces.append(piece)
    return sorted(pieces)


# --- Happy hour windows ---

def _hhmm(value: str) -> int:
    h, m = value.split(":")
    return int(h) * 60 + int(m)


def deal_windows(bar: dict, deal: dict, day: int) -> list[tuple[int, int]]:
    """Real windows (week minutes) when this deal runs on this day, clipped to opening hours.

    day may be -1 or 7 to look at the day before Monday or after Sunday; results
    are in the same coordinates (e.g. day -1 gives negative week minutes).
    """
    if DAY_CODES[day % 7] not in deal.get("days", []):
        return []
    base = day * 1440
    hours = open_intervals(bar)

    # Start of the deal
    if deal["start"] == "open":
        if not hours:
            return []  # "from opening" with unknown hours cannot be placed in time
        openings = sorted({s + shift for s, _ in hours for shift in (-WEEK, 0, WEEK)
                           if base <= s + shift < base + 1440})
        if not openings:
            return []  # closed that day
        start = openings[0]
    else:
        start = base + _hhmm(deal["start"])

    # End of the deal
    if deal["end"] == "close":
        if not hours:
            return []
        interval = containing_interval(hours, start)
        if interval is None:  # deal starts before the bar opens: use the next opening that day
            nxt = next_opening(hours, start)
            if nxt is None or nxt >= base + 1440:
                return []
            interval = containing_interval(hours, nxt)
        end = interval[1]
    elif deal["end"] == "open":
        return []  # not a sensible end
    else:
        end = base + _hhmm(deal["end"])
        if end <= start:
            end += 1440  # crosses midnight

    if not hours:
        return [(start, end)]  # hours unknown: trust the deal as written
    return _clip_to_hours((start, end), hours)


def contains(window: tuple[int, int], t: int) -> int | None:
    """If t falls inside the window (allowing for week wraparound), return the shift used."""
    for shift in (0, WEEK, -WEEK):
        if window[0] <= t + shift < window[1]:
            return shift
    return None


# --- Bar lookup ---

def norm_name(name: str) -> str:
    s = (name or "").replace("\u2019", "'").replace("`", "'")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("'", "")  # "Cooper's" and "coopers" should match
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\bthe\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_bar(query: str, neighborhood: str | None = None) -> dict:
    """Find one bar by place ID or name. Raises ToolError listing candidates if unclear."""
    q = (query or "").strip()
    if not q:
        raise ToolError("Empty bar name. Pass the bar's name or place_id.")
    direct = get_bar(q)
    if direct:
        return direct

    pool = [b for b in all_bars() if not neighborhood or b.get("neighborhood") == neighborhood]
    nq = norm_name(q)
    exact = [b for b in pool if norm_name(b["name"]) == nq]
    if len(exact) == 1:
        return exact[0]

    partial = exact or [b for b in pool if nq and (nq in norm_name(b["name"]) or norm_name(b["name"]) in nq)]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        names = {norm_name(b["name"]): b for b in pool}
        close = difflib.get_close_matches(nq, list(names), n=5, cutoff=0.75)
        if len(close) == 1:
            return names[close[0]]
        partial = [names[c] for c in close]

    if not partial:
        suggestions = difflib.get_close_matches(nq, [norm_name(b["name"]) for b in pool], n=3, cutoff=0.5)
        hint = f" Closest names: {', '.join(suggestions)}." if suggestions else ""
        raise ToolError(f"No bar named '{query}' in our data.{hint} "
                        f"Use find_bars to search, then pass the exact name or place_id.")
    options = "; ".join(f"{b['name']} ({neighborhood_name(b.get('neighborhood', ''))}, place_id {b['place_id']})"
                        for b in partial[:5])
    raise ToolError(f"'{query}' matches several bars: {options}. Pass the exact name or place_id.")


# --- Walking ---

def walk_meters(a: dict, b: dict) -> float:
    """Estimated walking distance in meters: straight line times a street factor."""
    lat1, lng1, lat2, lng2 = map(math.radians, (a["lat"], a["lng"], b["lat"], b["lng"]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h)) * STREET_FACTOR


def walk_minutes(a: dict, b: dict) -> int:
    """Estimated walking minutes between two bars."""
    meters = walk_meters(a, b)
    return 0 if meters < 1 else max(1, math.ceil(meters / WALK_METERS_PER_MIN))


# --- Formatting for the model ---

def days_label(codes: list[str]) -> str:
    """['mon','tue','wed','thu','fri'] -> 'Mon-Fri'; all seven -> 'Daily'; gaps -> 'Mon, Wed, Fri'."""
    idx = sorted(DAY_CODES.index(c) for c in codes if c in DAY_CODES)
    if len(idx) == 7:
        return "Daily"
    runs, start = [], None
    for i, d in enumerate(idx):
        if start is None:
            start = d
        if i + 1 == len(idx) or idx[i + 1] != d + 1:
            runs.append((start, d))
            start = None
    parts = []
    for a, b in runs:
        name_a, name_b = DAY_NAMES[a][:3], DAY_NAMES[b][:3]
        parts.append(name_a if a == b else (f"{name_a}, {name_b}" if b == a + 1 else f"{name_a}-{name_b}"))
    return ", ".join(parts)


def deal_hours_label(deal: dict) -> str:
    """'4:00 PM to 7:00 PM', 'opening to 7:00 PM', '9:00 PM to close'."""
    def one(value: str) -> str:
        if value == "open":
            return "opening"
        if value == "close":
            return "close"
        return fmt_clock(_hhmm(value))
    return f"{one(deal['start'])} to {one(deal['end'])}"


# --- Formatting specials for the model ---

def specials_summary(deal: dict, limit: int = 5) -> list[str]:
    out = []
    for sp in deal.get("specials", [])[:limit]:
        parts = [sp["item"]]
        if isinstance(sp.get("price"), (int, float)):
            parts.append(f"${sp['price']:g}")
        if sp.get("discount"):
            parts.append(str(sp["discount"]))
        label = " ".join(parts)
        out.append(f"{label} (food)" if sp.get("category") == "food" else label)
    extra = len(deal.get("specials", [])) - limit
    if extra > 0:
        out.append(f"+{extra} more")
    return out
