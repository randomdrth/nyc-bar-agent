"""The tools Two-Drink Minimum can run, and the JSON schemas that describe them to the model.

Every tool returns a JSON string. Errors come back as {"error": "..."} with a
message written for the model: what went wrong and what to try instead.
"""

import itertools
import json
import re
import time as _time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import bar_data as bd
import places
from bar_data import ToolError
from hh_deals import extract_deals
from hh_extract import crawl_site, find_hh_snippets, is_not_own_site

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


def _check_price(max_price) -> int | None:
    if max_price is None:
        return None
    try:
        max_price = int(max_price)
    except (TypeError, ValueError):
        raise ToolError("max_price must be a number from 1 ($) to 4 ($$$$).")
    if not 1 <= max_price <= 4:
        raise ToolError("max_price must be from 1 ($) to 4 ($$$$).")
    return max_price


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
        max_price = _check_price(max_price)
        day_idx, t, notes = _resolve_when(day, time)
    except ToolError as e:
        return json.dumps({"error": str(e)})

    now_list, soon_list = [], []
    candidates = [b for b in bd.all_bars()  # saved data plus bars verified live this session
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


# --- Tool: find_bars ---

MAX_FIND_RESULTS = 8
SERVES_FIELD = {"cocktail": "serves_cocktails", "wine": "serves_wine", "beer": "serves_beer"}
FEATURES = {"outdoor_seating": "outdoor seating", "live_music": "live music", "good_for_groups": "good for groups"}


def _style_ok(bar: dict, style: str, live: bool) -> bool:
    if style == "any":
        return True
    serves = bar.get(SERVES_FIELD[style])
    if serves is True or set(bar.get("types", [])) & STYLE_TYPES[style]:
        return True
    if serves is False:
        return False
    # Google gave no answer: live results were already searched for this style; saved data uses deal specials
    return True if live else _matches_style(bar, style)


def _features(bar: dict) -> list[str]:
    out = [label for field, label in (("serves_cocktails", "cocktails"), ("serves_wine", "wine"),
                                      ("serves_beer", "beer")) if bar.get(field)]
    out += [label for field, label in FEATURES.items() if bar.get(field)]
    return out


def _happy_hour_status(place_id: str, day_idx: int) -> str:
    bar = bd.get_bar(place_id)
    saved = bd.load()["by_id"].get(place_id)
    if bar and bar.get("deals"):
        live = any(d.get("live") for d in bar["deals"])
        today = [d for d in bar["deals"] if bd.DAY_CODES[day_idx] in d.get("days", [])]
        tag = " (verified live this session)" if live else ""
        if today:
            return "; ".join(f"{d['label']}: {bd.deal_hours_label(d)}" for d in today[:3]) + tag
        all_days = sorted({c for d in bar["deals"] for c in d.get("days", [])}, key=bd.DAY_CODES.index)
        return f"verified happy hour on other days ({bd.days_label(all_days)}){tag}"
    if saved is not None:
        return "checked its website, no verified happy hour"
    if bar and bar.get("website"):
        return "not in our happy hour data; check_happy_hour_online can check its website"
    return "not in our happy hour data"


def find_bars(neighborhood: str, style: str = "any", query: str | None = None, day: str | None = None,
              time: str | None = None, max_price: int | None = None, outdoor_seating: bool = False,
              live_music: bool = False, good_for_groups: bool = False) -> str:
    try:
        hood = bd.resolve_neighborhood(neighborhood)
        style = (style or "any").strip().lower()
        if style not in ("any", *STYLE_TYPES):
            raise ToolError(f"Unknown style '{style}'. Use any, cocktail, wine, or beer. For dive bars, "
                            f"rooftops, speakeasies and similar, put the word in 'query'.")
        max_price = _check_price(max_price)
        query = (query or "").strip()
        if len(query) > 60:
            raise ToolError("Keep 'query' short, a few words like 'dive', 'rooftop', or 'jazz'.")
        open_filter = bool(day or time)
        day_idx, t, notes = _resolve_when(day, time)
    except ToolError as e:
        return json.dumps({"error": str(e)})

    wanted = [f for f, on in (("outdoor_seating", outdoor_seating), ("live_music", live_music),
                              ("good_for_groups", good_for_groups)) if on]
    hood_name = bd.neighborhood_name(hood)
    style_word = "" if style == "any" else style
    text_query = " ".join(x for x in (query, style_word, f"bar in {hood_name}, Manhattan, New York") if x)

    source, fallback_reason = "google_places_live", None
    try:
        bars = places.search(hood, text_query)
        for b in bars:
            bd.register_live_place(b)
    except places.PlacesError as e:
        source, fallback_reason = "local_fallback", str(e)
        bars = [b for b in bd.load()["bars"] if b.get("neighborhood") == hood]
    live = source == "google_places_live"

    kept, dropped = [], Counter()
    for b in bars:
        if not _style_ok(b, style, live):
            dropped[f"not a {style} spot"] += 1
            continue
        if live and any(b.get(f) is not True for f in wanted):
            dropped["missing a requested feature"] += 1
            continue
        level = bd.price_level(b)
        if max_price is not None and level is not None and level > max_price:
            dropped["over the price limit"] += 1
            continue
        hours = bd.open_intervals(b)
        if hours is None:
            if open_filter:
                dropped["hours unknown"] += 1
                continue
            status = "hours unknown"
        else:
            interval = bd.containing_interval(hours, t)
            when = "then" if open_filter else "now"
            if interval:
                status = f"open {when}, until {bd.fmt_time(interval[1], day_idx)}"
            else:
                if open_filter:
                    dropped["closed then"] += 1
                    continue
                nxt = bd.next_opening(hours, t)
                status = f"closed now, opens {bd.fmt_time(nxt, day_idx)}" if nxt is not None else "closed"
        kept.append((b, status))

    if not live:  # live results keep Google's relevance order ("dive" should rank dive bars first)
        kept.sort(key=lambda x: (-(x[0].get("rating") or 0), -(x[0].get("rating_count") or 0)))
    results = []
    for b, status in kept[:MAX_FIND_RESULTS]:
        item = {**_bar_basics(b, with_neighborhood=False), "reviews": b.get("rating_count"),
                "address": b.get("address"), "status": status,
                "happy_hour": _happy_hour_status(b["place_id"], day_idx)}
        if b.get("summary"):
            item["description"] = b["summary"]
        features = _features(b)
        if features:
            item["features"] = features
        if b.get("website"):
            item["website"] = b["website"]
        results.append(item)

    result = {
        "source": source,
        "query": {"neighborhood": hood_name, "style": style, "search_words": query or None,
                  "open_at": f"{bd.DAY_NAMES[day_idx]} {bd.fmt_clock(t)}" if open_filter else None,
                  "max_price": ("$" * max_price) if max_price else None,
                  "features": [FEATURES[f] for f in wanted] or None},
        "results": results,
        "counts": {"found": len(bars), "matched": len(kept), "shown": len(results)},
    }
    if dropped:
        result["filtered_out"] = dict(dropped)
    if open_filter and notes:
        result["assumptions"] = notes
    if not live:
        ignored = [x for x, on in (("the search words", bool(query)), ("feature filters", bool(wanted))) if on]
        result["fallback_reason"] = fallback_reason
        result["note"] = ("Live Google search unavailable, so these results come from our saved data."
                          + (f" Could not apply {' and '.join(ignored)} without live data." if ignored else ""))
    if not results:
        result["suggestion"] = ("Nothing matched. Remove a filter (style, price, features), try another "
                                "time, or search a different neighborhood.")
    return json.dumps(result)


# --- Tool: check_happy_hour_online ---

CHECK_BUDGET_SECONDS = 40
CHECK_CACHE_SECONDS = 24 * 3600
_check_cache: dict[str, tuple[float, dict]] = {}
_check_pool = ThreadPoolExecutor(max_workers=2)


def _check_pipeline(bar: dict) -> dict:
    crawl = crawl_site(bar["website"], cache_dir=None, max_pages=3)
    pages = [p["final_url"] for p in crawl["pages"]]
    if crawl["status"] == "fetch_failed":
        return {"status": "site_unreachable", "detail": crawl.get("error"),
                "message": f"Could not load {bar['name']}'s website ({crawl.get('error')}). "
                           f"It may block automated visits or be down. Say you could not check it."}
    if crawl["status"] != "ok":
        return {"status": "site_unreadable", "pages_checked": pages,
                "message": f"{bar['name']}'s website shows almost no readable text (it is likely built with "
                           f"JavaScript or images). Say deals could not be read from it."}
    snippets = find_hh_snippets(crawl["pages"])
    if not snippets:
        return {"status": "no_deal_text", "pages_checked": pages,
                "message": f"No happy hour or deal text on the {len(pages)} pages checked. That does not "
                           f"prove there is none; it may be on Instagram or a menu image."}
    found = extract_deals(bar["name"], bar.get("address", ""), snippets, cache_dir=None)
    if not found["llm_ok"]:
        return {"status": "analysis_failed", "pages_checked": pages,
                "message": "Found deal text but could not analyze it right now. Try again later."}
    verified = [d for d in found["deals"] if d["status"] == "verified"]
    unconfirmed = sum(1 for d in found["deals"] if d["status"] == "needs_review")
    return {"status": "verified" if verified else "no_deal", "pages_checked": pages,
            "verified": verified, "unconfirmed": unconfirmed}


def check_happy_hour_online(bar: str) -> str:
    try:
        record = bd.find_bar(bar)
    except ToolError as e:
        return json.dumps({"error": f"{e} If the bar came from find_bars, pass its place_id."})
    pid, name = record["place_id"], record["name"]

    hit = _check_cache.get(pid)
    if hit and _time.time() - hit[0] < CHECK_CACHE_SECONDS:
        return json.dumps({**hit[1], "cached": True})

    website = record.get("website")
    if not website:
        return json.dumps({"error": f"Google lists no website for {name}, so there is nothing to check. "
                                    f"Tell the user to check the bar's Instagram or call ahead."})
    if is_not_own_site(website):
        return json.dumps({"error": f"{name}'s listed website is {website}, a social or booking page "
                                    f"that cannot be read. Tell the user to check it directly."})

    saved = bd.load()["by_id"].get(pid)
    on_file = ("verified happy hour on file" if saved and saved.get("deals") and
               not any(d.get("live") for d in saved["deals"])
               else "checked before, none verified" if saved else "not in saved data")

    future = _check_pool.submit(_check_pipeline, record)
    try:
        outcome = future.result(timeout=CHECK_BUDGET_SECONDS)
    except FuturesTimeout:
        return json.dumps({"bar": name, "status": "timed_out",
                           "message": f"{name}'s website took over {CHECK_BUDGET_SECONDS} seconds. "
                                      f"Say it could not be checked right now."})
    except Exception as e:
        return json.dumps({"bar": name, "status": "error",
                           "message": f"Checking failed ({type(e).__name__}). Say it could not be checked."})

    result = {"bar": name, "place_id": pid, "website": website, "saved_data": on_file, **outcome}
    if outcome["status"] == "verified":
        deals = outcome.pop("verified")
        result.pop("verified", None)
        result["deals"] = [{"deal": d["label"], "days": bd.days_label(d["days"]),
                            "hours": bd.deal_hours_label(d), **{k: v for k, v in _deal_info(d).items()
                                                                if k != "deal"},
                            "evidence": d["evidence"]} for d in deals]
        attached = bd.add_live_deals(pid, deals)
        result["message"] = (
            f"Found {len(deals)} verified deal(s) on the website. "
            + ("They are now used by get_happy_hours and plan_bar_crawl for this session."
               if attached else "Our saved verified deals for this bar are kept for planning.")
        )
    elif outcome["status"] == "no_deal":
        result.pop("verified", None)
        result["message"] = "The website mentions deals or hours, but no happy hour with clear days and times."
    if outcome.get("unconfirmed"):
        result["unconfirmed_deals"] = (f"{outcome['unconfirmed']} more deal(s) found but not confirmed by "
                                       f"our checks; do not present them as fact.")
    result.pop("unconfirmed", None)

    if outcome["status"] in ("verified", "no_deal", "no_deal_text", "site_unreadable"):
        _check_cache[pid] = (_time.time(), result)
    return json.dumps(result)


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

TOOLS += [
    {
        "type": "function",
        "function": {
            "name": "find_bars",
            "description": (
                "Search bars in one neighborhood with live Google data: rating, price, whether it is open at a "
                "given time, a short description, features (cocktails, wine, beer, outdoor seating, live music, "
                "good for groups), and our happy hour status for that day. Use this for any request about kinds of "
                "bars, vibes, or what is open, and to find candidates that have no happy hour. Covers the West "
                "Village, East Village, and Upper West Side. The 'source' field says whether results are live "
                "or from saved data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "neighborhood": {"type": "string",
                                     "description": "'West Village', 'East Village', or 'Upper West Side' "
                                                    "(aliases WV, EV, UWS work)."},
                    "style": {"type": "string", "enum": ["any", "cocktail", "wine", "beer"],
                              "description": "Kind of drinks. Default 'any'."},
                    "query": {"type": "string",
                              "description": "A few extra search words, e.g. 'dive', 'rooftop', 'speakeasy', "
                                             "'jazz', 'quiet'. Optional."},
                    "day": {"type": "string",
                            "description": "Only bars open on this day, e.g. 'Friday'. Use with time. Optional."},
                    "time": {"type": "string",
                             "description": "Only bars open at this time, e.g. '11pm'. Optional."},
                    "max_price": {"type": "integer", "description": "Price cap, 1 ($) to 4 ($$$$). Optional."},
                    "outdoor_seating": {"type": "boolean", "description": "Only bars with outdoor seating."},
                    "live_music": {"type": "boolean", "description": "Only bars with live music."},
                    "good_for_groups": {"type": "boolean", "description": "Only bars good for groups."},
                },
                "required": ["neighborhood"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_happy_hour_online",
            "description": (
                "Read one bar's own website right now and extract its happy hours, checking every deal against "
                "the page text. Slow (up to 40 seconds), so use it only when the user asks about a specific bar "
                "that has no happy hour data (find_bars says 'not in our happy hour data') or asks to recheck "
                "one. Verified deals found here become available to get_happy_hours and plan_bar_crawl."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bar": {"type": "string",
                            "description": "The bar's place_id from find_bars or get_happy_hours, or its exact name."},
                },
                "required": ["bar"],
            },
        },
    },
]

TOOL_MAP = {
    "get_happy_hours": get_happy_hours,
    "plan_bar_crawl": plan_bar_crawl,
    "find_bars": find_bars,
    "check_happy_hour_online": check_happy_hour_online,
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
