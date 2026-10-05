"""The tools Two-Drink Minimum can run, and the JSON schemas that describe them to the model.

Every tool returns a JSON string. Errors come back as {"error": "..."} with a
message written for the model: what went wrong and what to try instead.
"""

import itertools
import json
import re

import bar_data as bd
from bar_data import ToolError

STARTING_SOON_MINUTES = 90
MAX_RESULTS = 10
MAX_SOON_RESULTS = 5
MAX_WAIT_FOR_OPENING = 30
MIN_STAY = 20
LONG_WALK_MINUTES = 20
JUST_MISSED_MINUTES = 30

STYLE_TYPES = {
    "cocktail": {"cocktail_bar", "lounge_bar"},
    "wine": {"wine_bar"},
    "beer": {"pub", "irish_pub", "sports_bar", "beer_garden", "brewery", "brewpub", "gastropub", "beer_hall"},
}
STYLE_WORDS = {
    "cocktail": re.compile(r"cocktail|martini|margarita|spritz|negroni|old fashioned|mezcal|tequila|\bwell", re.I),
    "wine": re.compile(r"wine|prosecco|bubbles|champagne|cava|\bros[eé]\b", re.I),
    "beer": re.compile(r"beer|draft|draught|pint|\bipa\b|lager|\bales?\b|pitcher|bucket|\bcans?\b", re.I),
}


def _matches_style(bar: dict, style: str) -> bool:
    if style == "any":
        return True
    if set(bar.get("types", [])) & STYLE_TYPES[style]:
        return True
    return any(STYLE_WORDS[style].search(sp["item"]) for d in bar.get("deals", []) for sp in d.get("specials", []))


def _resolve_when(day: str | None, time: str | None) -> tuple[int, int, list[str]]:
    """Return (day index, week minute, notes about assumptions)."""
    now = bd.now_nyc()
    day_idx = bd.parse_day(day, now)
    minutes, note = bd.parse_time(time, now)
    notes = [note] if note else []
    if not day:
        notes.append(f"no day given, used today ({bd.DAY_NAMES[day_idx]}, New York time)")
    if not time:
        notes.append(f"no time given, used now ({bd.fmt_clock(minutes)} New York time)")
    return day_idx, bd.week_minute(day_idx, minutes), notes


def _bar_basics(bar: dict, with_neighborhood: bool = True) -> dict:
    basics = {"name": bar["name"], "place_id": bar["place_id"]}
    if with_neighborhood:
        basics["neighborhood"] = bd.neighborhood_name(bar.get("neighborhood", ""))
    basics.update({
        "rating": bar.get("rating"),
        "price": bd.price_symbol(bar),
        "lat": round(bar["lat"], 5),
        "lng": round(bar["lng"], 5),
    })
    return basics


def _deal_info(deal: dict) -> dict:
    info = {"deal": deal["label"], "specials": bd.specials_summary(deal) or ["specials not listed"]}
    if deal.get("has_food"):
        info["includes_food"] = True
    if deal.get("conditions"):
        info["conditions"] = deal["conditions"]
    if deal.get("source_url"):
        info["source"] = deal["source_url"]
    return info


# --- Tool: get_happy_hours ---

def get_happy_hours(neighborhood: str, day: str | None = None, time: str | None = None,
                    style: str = "any", with_food: bool = False, max_price: int | None = None) -> str:
    try:
        hood = bd.resolve_neighborhood(neighborhood)
        style = (style or "any").strip().lower()
        if style not in ("any", *STYLE_TYPES):
            raise ToolError(f"Unknown style '{style}'. Use one of: any, cocktail, wine, beer. "
                            f"Dive bars are not a filter; use find_bars and judge from the results.")
        if max_price is not None:
            try:
                max_price = int(max_price)
            except (TypeError, ValueError):
                raise ToolError("max_price must be a number from 1 ($) to 4 ($$$$).")
            if not 1 <= max_price <= 4:
                raise ToolError("max_price must be from 1 ($) to 4 ($$$$).")
        day_idx, t, notes = _resolve_when(day, time)
    except ToolError as e:
        return json.dumps({"error": str(e)})

    now_list, soon_list = [], []
    candidates = [b for b in bd.load()["bars"]
                  if b.get("neighborhood") == hood and b.get("deals") and _matches_style(b, style)]
    price_unknown = 0
    if max_price is not None:
        kept = []
        for b in candidates:
            level = bd.price_level(b)
            if level is None:
                price_unknown += 1
                kept.append(b)
            elif level <= max_price:
                kept.append(b)
        candidates = kept

    for bar in candidates:
        hours = bd.open_intervals(bar)
        active, upcoming = [], []
        for deal in bar["deals"]:
            if with_food and not deal.get("has_food"):
                continue
            # Yesterday's windows matter too: a 10pm-2am deal is still running at 1am
            for d in (day_idx - 1, day_idx, day_idx + 1):
                for window in bd.deal_windows(bar, deal, d):
                    shift = bd.contains(window, t)
                    if shift is not None:
                        end = window[1] - shift
                        active.append({**_deal_info(deal), "ends_at": bd.fmt_time(end, day_idx),
                                       "minutes_left": end - t})
                    else:
                        for s in (0, bd.WEEK, -bd.WEEK):
                            start = window[0] + s
                            if t < start <= t + STARTING_SOON_MINUTES:
                                upcoming.append({**_deal_info(deal), "starts_at": bd.fmt_time(start, day_idx),
                                                 "starts_in_minutes": start - t,
                                                 "ends_at": bd.fmt_time(window[1] + s, day_idx)})
        if not active and not upcoming:
            continue
        basics = _bar_basics(bar, with_neighborhood=False)  # already in the query echo
        if hours is None:
            basics["hours"] = "unknown, check before going"
        else:
            open_now = bd.containing_interval(hours, t)
            basics["bar_closes_at"] = bd.fmt_time(open_now[1], day_idx) if open_now else None
        if active:
            active.sort(key=lambda a: -a["minutes_left"])
            now_list.append({**basics, "happy_hours": _dedupe(active)})
        if upcoming:
            upcoming.sort(key=lambda u: u["starts_in_minutes"])
            soon_list.append({**basics, "happy_hours": _dedupe(upcoming)})

    now_list.sort(key=lambda b: (-(b["rating"] or 0), -b["happy_hours"][0]["minutes_left"]))
    soon_list.sort(key=lambda b: b["happy_hours"][0]["starts_in_minutes"])

    result = {
        "query": {
            "neighborhood": bd.neighborhood_name(hood),
            "when": f"{bd.DAY_NAMES[day_idx]} {bd.fmt_clock(t)}",
            "style": style,
            "with_food": with_food,
            "max_price": ("$" * max_price) if max_price else None,
        },
        "happening_now": now_list[:MAX_RESULTS],
        "starting_soon": soon_list[:MAX_SOON_RESULTS],
        "counts": {"happening_now": len(now_list), "starting_soon": len(soon_list),
                   "shown": {"happening_now": min(len(now_list), MAX_RESULTS),
                             "starting_soon": min(len(soon_list), MAX_SOON_RESULTS)}},
        "note": "Only verified happy hours are listed, checked against each bar's own website. "
                "Bars not listed may still have deals we could not verify.",
    }
    if notes:
        result["assumptions"] = notes
    if price_unknown:
        result["price_note"] = f"{price_unknown} bars have no price level on Google and were kept."
    if not now_list and not soon_list:
        result["suggestion"] = _nothing_found_hint(candidates, day_idx, t, with_food)
    return json.dumps(result)


def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for item in items:
        key = (item["deal"], item.get("ends_at"), item.get("starts_at"))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _nothing_found_hint(candidates: list[dict], day_idx: int, t: int, with_food: bool) -> str:
    """When nothing is running, say when the last one ended and the next one starts."""
    if not candidates:
        return "No bars match these filters at all. Try style 'any', drop max_price, or another neighborhood."
    last_end, next_start = None, None
    for bar in candidates:
        for deal in bar["deals"]:
            if with_food and not deal.get("has_food"):
                continue
            for d in range(day_idx - 1, day_idx + 8):
                for start, end in bd.deal_windows(bar, deal, d):
                    if day_idx * 1440 <= end <= t and (last_end is None or end > last_end[0]):
                        last_end = (end, bar["name"])
                    if start > t and (next_start is None or start < next_start[0]):
                        next_start = (start, bar["name"])
    parts = []
    if last_end:
        parts.append(f"The last one today ended at {bd.fmt_time(last_end[0], day_idx)} ({last_end[1]}).")
    if next_start:
        parts.append(f"The next one starts {bd.DAY_NAMES[(next_start[0] // 1440) % 7]} at "
                     f"{bd.fmt_clock(next_start[0])} ({next_start[1]}).")
    parts.append("Call again with that day and time, or use find_bars for bars open now without a deal.")
    return " ".join(parts)


# --- Tool: plan_bar_crawl ---

def _windows_by_bar(bars: list[dict], base_day: int) -> dict[str, list[tuple[int, int, dict]]]:
    """Every happy hour window near the crawl day, computed once per bar, shifted copies included."""
    out = {}
    for bar in bars:
        windows = []
        for deal in bar.get("deals", []):
            for d in range(base_day - 1, base_day + 2):
                for w_start, w_end in bd.deal_windows(bar, deal, d):
                    for s in (0, bd.WEEK, -bd.WEEK):
                        windows.append((w_start + s, w_end + s, deal))
        out[bar["place_id"]] = windows
    return out


def _simulate(order: list[dict], t0: int, minutes_per_stop: int, base_day: int,
              windows_by_bar: dict[str, list[tuple[int, int, dict]]]) -> dict:
    t, prev = t0, None
    stops, skipped = [], []
    total_walk = total_wait = hh_minutes = 0
    for bar in order:
        walk = bd.walk_minutes(prev, bar) if prev else 0
        arrive = t + walk
        notes = []
        hours = bd.open_intervals(bar)
        close = None
        wait = 0
        if hours is None:
            notes.append("hours unknown, check before going")
        else:
            interval = bd.containing_interval(hours, arrive)
            if interval is None:
                nxt = bd.next_opening(hours, arrive)
                if nxt is not None and nxt - arrive <= MAX_WAIT_FOR_OPENING:
                    wait = nxt - arrive
                    arrive = nxt
                    interval = bd.containing_interval(hours, arrive)
                    notes.append(f"opens at {bd.fmt_time(arrive, base_day)}, about {wait} min wait")
                else:
                    reason = "closed at that time"
                    if nxt is not None:
                        reason += f"; next opens {bd.DAY_NAMES[(nxt // 1440) % 7]} {bd.fmt_clock(nxt)}"
                    skipped.append({"name": bar["name"], "would_arrive": bd.fmt_time(arrive, base_day),
                                    "reason": reason})
                    continue
            close = interval[1]
        leave = arrive + minutes_per_stop
        if close is not None and leave > close:
            if close - arrive < MIN_STAY:
                skipped.append({"name": bar["name"], "would_arrive": bd.fmt_time(arrive, base_day),
                                "reason": f"closes at {bd.fmt_time(close, base_day)}, too soon for a visit"})
                continue
            leave = close
            notes.append(f"stay cut short, closes at {bd.fmt_time(close, base_day)}")

        # Happy hour during the stay
        best, best_overlap, missed_by = None, 0, None
        for ws, we, deal in windows_by_bar[bar["place_id"]]:
            overlap = min(we, leave) - max(ws, arrive)
            if overlap > best_overlap:
                best, best_overlap = (deal, ws, we), overlap
            gap = arrive - we
            if 0 <= gap <= JUST_MISSED_MINUTES and (missed_by is None or gap < missed_by[0]):
                missed_by = (gap, deal)
        happy_hour = None
        if best:
            deal, ws, we = best
            happy_hour = {**_deal_info(deal), "covers_minutes": best_overlap,
                          "runs": f"{bd.fmt_time(ws, base_day)} to {bd.fmt_time(we, base_day)}"}
            if ws > arrive:
                notes.append(f"happy hour starts {bd.fmt_time(ws, base_day)}, during your stay")
            hh_minutes += best_overlap
        elif missed_by:
            notes.append(f"arrives {missed_by[0]} min after its happy hour ({missed_by[1]['label']}) ends")
        if walk > LONG_WALK_MINUTES:
            notes.append(f"{walk} min walk, consider a cab or the subway")

        stops.append({
            **_bar_basics(bar),
            "walk_minutes_from_previous": walk if prev else None,
            "arrive": bd.fmt_time(arrive, base_day),
            "leave": bd.fmt_time(leave, base_day),
            "bar_closes_at": bd.fmt_time(close, base_day) if close is not None else None,
            "happy_hour": happy_hour,
            "notes": notes,
            "_arrive": arrive, "_leave": leave,
        })
        total_walk += walk
        total_wait += wait
        t, prev = leave, bar
    return {"stops": stops, "skipped": skipped, "walk": total_walk, "wait": total_wait, "hh": hh_minutes}


def _score(plan: dict) -> tuple:
    # More bars visited, then more happy hour, then less walking, then less waiting
    return (-len(plan["stops"]), -plan["hh"], plan["walk"], plan["wait"])


def plan_bar_crawl(bars: list, day: str | None = None, start_time: str | None = None,
                   minutes_per_stop: int = 45, keep_order: bool = False) -> str:
    try:
        if isinstance(bars, str):
            bars = [b.strip() for b in bars.split(",") if b.strip()]
        if not isinstance(bars, list) or not 2 <= len(bars) <= 6:
            raise ToolError("Pass 2 to 6 bars (names or place_ids). To find candidates, call "
                            "get_happy_hours or find_bars first.")
        try:
            minutes_per_stop = int(minutes_per_stop)
        except (TypeError, ValueError):
            raise ToolError("minutes_per_stop must be a number of minutes, e.g. 45.")
        if not 20 <= minutes_per_stop <= 120:
            raise ToolError("minutes_per_stop must be between 20 and 120.")
        resolved, errors = [], []
        for name in bars:
            try:
                resolved.append(bd.find_bar(str(name)))
            except ToolError as e:
                errors.append(str(e))
        if errors:
            raise ToolError(" ".join(errors))
        ids = [b["place_id"] for b in resolved]
        if len(set(ids)) != len(ids):
            dupes = sorted({b["name"] for b in resolved if ids.count(b["place_id"]) > 1})
            raise ToolError(f"The same bar appears twice: {', '.join(dupes)}. List each bar once.")
        day_idx, t0, notes = _resolve_when(day, start_time)
    except ToolError as e:
        return json.dumps({"error": str(e)})

    orders = [resolved] if keep_order else [list(p) for p in itertools.permutations(resolved)]
    windows = _windows_by_bar(resolved, day_idx)
    plans = [_simulate(order, t0, minutes_per_stop, day_idx, windows) for order in orders]
    best = min(plans, key=_score)

    # Explain the choice against the shortest walking route among plans visiting as many bars
    why = "Order kept as given." if keep_order else _explain(best, plans)

    warnings = [f"{s['name']}: {n}" for s in best["stops"] for n in s["notes"]]
    for s in best["skipped"]:
        warnings.append(f"{s['name']} left out: {s['reason']}. Drop it, pick another bar, or change the time.")

    meters = sum(bd.walk_meters(a, b) for a, b in zip(best["stops"], best["stops"][1:]))
    stops_out = []
    for i, s in enumerate(best["stops"], 1):
        stop = {k: v for k, v in s.items() if not k.startswith("_") and k != "notes"}
        stops_out.append({"stop": i, **stop})

    result = {
        "query": {"day": bd.DAY_NAMES[day_idx], "start": bd.fmt_clock(t0),
                  "minutes_per_stop": minutes_per_stop, "bars_requested": len(resolved)},
        "summary": {
            "stops": len(best["stops"]),
            "starts": bd.fmt_time(best["stops"][0]["_arrive"], day_idx) if best["stops"] else None,
            "ends": bd.fmt_time(best["stops"][-1]["_leave"], day_idx) if best["stops"] else None,
            "total_walk_minutes": best["walk"],
            "total_walk_miles": round(meters / 1609.34, 1),
            "happy_hour_minutes": best["hh"],
        },
        "why_this_order": why,
        "stops": stops_out,
        "warnings": warnings,
    }
    if notes:
        result["assumptions"] = notes
    if not best["stops"]:
        result["error"] = ("None of these bars can be visited at that time. Try a different day or start "
                           "time, or check which bars are open with find_bars.")
    return json.dumps(result)


def _explain(best: dict, plans: list[dict]) -> str:
    same_count = [p for p in plans if len(p["stops"]) == len(best["stops"])]
    shortest = min(same_count, key=lambda p: (p["walk"], -p["hh"]))
    if best["hh"] > shortest["hh"]:
        extra_walk = best["walk"] - shortest["walk"]
        gained = best["hh"] - shortest["hh"]
        cost = f" for {extra_walk} more min of walking" if extra_walk > 0 else " with no extra walking"
        return f"Ordered to catch {gained} more min of happy hour than the shortest route{cost}."
    if best["hh"] > 0:
        return "Shortest walking route, and it also catches the most happy hour time."
    return "Shortest walking route. None of these bars has a verified happy hour during this time."


# --- Schemas: what the model sees ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_happy_hours",
            "description": (
                "List bars with a verified happy hour running at a given day and time in one neighborhood, "
                "plus deals starting within the next 90 minutes. Each result says when the deal ends, "
                "the specials, whether food is included, and when the bar closes. Covers the West Village, "
                "East Village, and Upper West Side only. Use this for any question about happy hours or "
                "drink deals, and to pick candidate stops before calling plan_bar_crawl."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "neighborhood": {"type": "string",
                                     "description": "'West Village', 'East Village', or 'Upper West Side' "
                                                    "(aliases WV, EV, UWS work)."},
                    "day": {"type": "string",
                            "description": "Weekday like 'Thursday', or 'today', 'tomorrow', or YYYY-MM-DD. "
                                           "Omit for today in New York."},
                    "time": {"type": "string",
                             "description": "Time like '6pm', '6:30 pm', or '18:00'. Omit for now in New York."},
                    "style": {"type": "string", "enum": ["any", "cocktail", "wine", "beer"],
                              "description": "Kind of bar or drinks. Default 'any'."},
                    "with_food": {"type": "boolean",
                                  "description": "Only deals that include food. Default false."},
                    "max_price": {"type": "integer",
                                  "description": "Google price level cap, 1 ($) to 4 ($$$$). Omit for no cap."},
                },
                "required": ["neighborhood"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_bar_crawl",
            "description": (
                "Turn 2 to 6 chosen bars into a timed walking route for a given day and start time. "
                "Tries every order and picks the one that catches the most happy hour time, then the least "
                "walking, while making sure each bar is open on arrival. Returns arrive and leave times, "
                "walking minutes between stops, the happy hour at each stop, and warnings (closed bars, "
                "missed happy hours, long walks). Pass bar names or place_ids from get_happy_hours or find_bars."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bars": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6,
                             "description": "2 to 6 bar names or place_ids."},
                    "day": {"type": "string",
                            "description": "Weekday like 'Friday', 'today', 'tomorrow', or YYYY-MM-DD. "
                                           "Omit for today."},
                    "start_time": {"type": "string",
                                   "description": "When the night starts, e.g. '5pm'. Omit for now."},
                    "minutes_per_stop": {"type": "integer",
                                         "description": "Time at each bar, 20 to 120. Default 45."},
                    "keep_order": {"type": "boolean",
                                   "description": "True to keep the user's order and only time it. "
                                                  "Default false (find the best order)."},
                },
                "required": ["bars"],
            },
        },
    },
]

TOOL_MAP = {
    "get_happy_hours": get_happy_hours,
    "plan_bar_crawl": plan_bar_crawl,
}


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_MAP)}."})
    if not isinstance(args, dict):
        return json.dumps({"error": f"Arguments for {name} must be a JSON object."})
    try:
        return TOOL_MAP[name](**args)
    except TypeError as e:
        return json.dumps({"error": f"Bad arguments for {name}: {e}. Check the parameter names in its schema."})
    except Exception as e:  # never crash the agent loop
        return json.dumps({"error": f"{name} failed unexpectedly ({type(e).__name__}: {str(e)[:200]}). "
                                    f"Try different arguments, or answer without this tool and say so."})
