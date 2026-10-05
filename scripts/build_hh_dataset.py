"""Build the happy hour dataset in stages.

Stage 1, discover: find bars in each neighborhood with Places API Text Search,
keep only those inside the neighborhood polygon, save to data/stage_discover.json.

Stage 2, crawl: fetch each bar's own website and keep the text that looks like
deals (happy hours, half price nights, timed specials). Saves to
data/stage_crawl.json. Raw pages are cached in data/cache.nosync/.

Stage 3, extract: Gemini reads each bar's deal text and returns structured
deals, which code checks against the text. Saves to data/stage_extract.json.

Stage 4, export: merge everything plus data/manual_overrides.json into
data/happy_hours.json (what the app reads) and data/happy_hours_review.csv.

Usage:
    uv run scripts/build_hh_dataset.py discover
    uv run scripts/build_hh_dataset.py discover --neighborhood west_village
    uv run scripts/build_hh_dataset.py preview
    uv run scripts/build_hh_dataset.py crawl --top 5      # pilot: 5 per neighborhood
    uv run scripts/build_hh_dataset.py crawl              # every discovered bar
    uv run scripts/build_hh_dataset.py extract --name "Jake"   # one bar, printed in full
    uv run scripts/build_hh_dataset.py extract --limit 10      # pilot, printed in full
    uv run scripts/build_hh_dataset.py extract                 # every bar with deal text
    uv run scripts/build_hh_dataset.py extract --only-status no_deal   # redo only these
    uv run scripts/build_hh_dataset.py export
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so we can import hh_extract from the project root

from hh_deals import DAY_CODES, extract_deals  # noqa: E402
from hh_extract import crawl_site, find_hh_snippets  # noqa: E402

DATA = ROOT / "data"
CACHE_DIR = DATA / "cache.nosync"  # .nosync keeps iCloud from offloading it
LLM_CACHE_DIR = CACHE_DIR / "llm"
NEIGHBORHOODS_FILE = DATA / "neighborhoods.json"
DISCOVER_FILE = DATA / "stage_discover.json"
CRAWL_FILE = DATA / "stage_crawl.json"
EXTRACT_FILE = DATA / "stage_extract.json"
OVERRIDES_FILE = DATA / "manual_overrides.json"
OUTPUT_JSON = DATA / "happy_hours.json"
OUTPUT_CSV = DATA / "happy_hours_review.csv"
PREVIEW_FILE = DATA / "preview.geojson"

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.types",
    "places.primaryType",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.regularOpeningHours",
    "places.websiteUri",
    "places.googleMapsUri",
    "places.businessStatus",
    "nextPageToken",  # required in the mask to get pagination
])

QUERIES = ["bar", "cocktail bar", "wine bar", "pub", "dive bar", "beer bar", "happy hour"]
MAX_PAGES_PER_QUERY = 3          # 20 results per page, 60 max per query
MAX_REQUESTS_PER_RUN = 120       # hard stop on top of the GCP daily quota
PAUSE_SECONDS = 0.5

TOP_N_DEFAULT = 1000             # effectively no cap: crawl every discovered bar
CRAWL_WORKERS_DEFAULT = 6        # different bars in parallel, one site at a time each
EXTRACT_WORKERS_DEFAULT = 4

BAR_TYPES = {
    "bar", "pub", "wine_bar", "cocktail_bar", "night_club", "brewery", "brewpub",
    "beer_garden", "sports_bar", "lounge_bar", "irish_pub", "bar_and_grill",
    "gastropub", "hookah_bar",
}

request_count = 0


# --- Geometry ---

def bounding_rectangle(polygon: list[list[float]]) -> dict:
    lats = [p[0] for p in polygon]
    lngs = [p[1] for p in polygon]
    return {
        "low": {"latitude": min(lats), "longitude": min(lngs)},
        "high": {"latitude": max(lats), "longitude": max(lngs)},
    }


def point_in_polygon(lat: float, lng: float, polygon: list[list[float]]) -> bool:
    """Ray casting. Polygon points are [lat, lng]."""
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        lat_i, lng_i = polygon[i]
        lat_j, lng_j = polygon[j]
        if (lat_i > lat) != (lat_j > lat):
            lng_cross = lng_i + (lat - lat_i) * (lng_j - lng_i) / (lat_j - lat_i)
            if lng < lng_cross:
                inside = not inside
        j = i
    return inside


# --- Places API ---

def search_text(query: str, rectangle: dict, api_key: str, page_token: str | None = None) -> dict:
    global request_count
    if request_count >= MAX_REQUESTS_PER_RUN:
        sys.exit(f"Stopped: hit MAX_REQUESTS_PER_RUN ({MAX_REQUESTS_PER_RUN}).")

    body = {"textQuery": query, "pageSize": 20, "locationRestriction": {"rectangle": rectangle}}
    if page_token:
        body["pageToken"] = page_token

    resp = requests.post(
        PLACES_URL,
        json=body,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": FIELD_MASK,
        },
        timeout=15,
    )
    request_count += 1
    time.sleep(PAUSE_SECONDS)

    if resp.status_code != 200:
        sys.exit(f"Places API error {resp.status_code} on query '{query}': {resp.text[:400]}")
    return resp.json()


def slim(place: dict, hood_key: str) -> dict:
    hours = place.get("regularOpeningHours", {})
    return {
        "place_id": place["id"],
        "name": place.get("displayName", {}).get("text", ""),
        "address": place.get("formattedAddress", ""),
        "lat": place["location"]["latitude"],
        "lng": place["location"]["longitude"],
        "neighborhood": hood_key,
        "primary_type": place.get("primaryType", ""),
        "types": place.get("types", []),
        "rating": place.get("rating"),
        "rating_count": place.get("userRatingCount"),
        "price_level": place.get("priceLevel"),
        "website": place.get("websiteUri"),
        "maps_url": place.get("googleMapsUri"),
        "hours_text": hours.get("weekdayDescriptions", []),
        "hours_periods": hours.get("periods", []),
    }


# --- Stage 1: discover ---

def discover_neighborhood(hood_key: str, hood: dict, api_key: str) -> list[dict]:
    rectangle = bounding_rectangle(hood["polygon"])
    seen: dict[str, dict] = {}
    raw_total = 0

    for query in QUERIES:
        page_token = None
        for _ in range(MAX_PAGES_PER_QUERY):
            data = search_text(query, rectangle, api_key, page_token)
            places = data.get("places", [])
            raw_total += len(places)
            for p in places:
                seen.setdefault(p["id"], p)
            page_token = data.get("nextPageToken")
            if not page_token:
                break

    kept, outside, not_bar, closed = [], 0, 0, 0
    for p in seen.values():
        loc = p.get("location", {})
        if not point_in_polygon(loc.get("latitude", 0), loc.get("longitude", 0), hood["polygon"]):
            outside += 1
            continue
        if not set(p.get("types", [])) & BAR_TYPES:
            not_bar += 1
            continue
        if p.get("businessStatus", "OPERATIONAL") != "OPERATIONAL":
            closed += 1
            continue
        kept.append(slim(p, hood_key))

    print(
        f"{hood['name']}: {raw_total} raw results, {len(seen)} unique, "
        f"kept {len(kept)} (dropped {outside} outside polygon, {not_bar} not a bar, {closed} closed)"
    )
    return kept


def run_discover(only: str | None) -> None:
    neighborhoods = json.loads(NEIGHBORHOODS_FILE.read_text())
    if only and only not in neighborhoods:
        sys.exit(f"Unknown neighborhood '{only}'. Options: {', '.join(neighborhoods)}")

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("PLACES_API_KEY")
    if not api_key:
        sys.exit("PLACES_API_KEY is missing. Add it to .env in the project root.")
    targets = {only: neighborhoods[only]} if only else neighborhoods

    # Keep results for neighborhoods we are not rerunning
    existing = []
    if DISCOVER_FILE.exists():
        existing = json.loads(DISCOVER_FILE.read_text()).get("places", [])
    places = [p for p in existing if p["neighborhood"] not in targets]

    for key, hood in targets.items():
        places += discover_neighborhood(key, hood, api_key)

    DISCOVER_FILE.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "places": places,
    }, indent=2))
    print(f"\nSaved {len(places)} bars to {DISCOVER_FILE.relative_to(ROOT)}")
    print(f"Places API requests this run: {request_count}")


# --- Stage 2: crawl ---

def select_top(places: list[dict], top_n: int) -> list[dict]:
    """Top N bars per neighborhood by review count."""
    by_hood: dict[str, list[dict]] = {}
    for p in places:
        by_hood.setdefault(p["neighborhood"], []).append(p)
    chosen = []
    for items in by_hood.values():
        items.sort(key=lambda p: p.get("rating_count") or 0, reverse=True)
        chosen += items[:top_n]
    return chosen


def crawl_one(place: dict) -> dict:
    result = crawl_site(place.get("website"), cache_dir=CACHE_DIR)
    snippets = find_hh_snippets(result["pages"]) if result["status"] == "ok" else []
    return {
        "place_id": place["place_id"],
        "name": place["name"],
        "neighborhood": place["neighborhood"],
        "website": place.get("website"),
        "crawl_status": result["status"],
        "error": result.get("error"),
        "pages_fetched": [p["final_url"] for p in result["pages"]],
        "deal_text_found": bool(snippets),
        "says_happy_hour": any(s["strong_hh"] for s in snippets),
        "snippets": snippets,
    }


def crawl_label(r: dict) -> str:
    if r["deal_text_found"]:
        return "DEAL TEXT (says happy hour)" if r["says_happy_hour"] else "DEAL TEXT"
    return "no deal text" if r["crawl_status"] == "ok" else r["crawl_status"]


def run_crawl(top_n: int, workers: int) -> None:
    if not DISCOVER_FILE.exists():
        sys.exit("Run the discover stage first.")
    places = json.loads(DISCOVER_FILE.read_text())["places"]
    chosen = select_top(places, top_n)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cap_note = "all bars" if top_n >= 1000 else f"top {top_n} per neighborhood"
    print(f"Crawling {len(chosen)} bars ({cap_note}), {workers} at a time...\n")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(crawl_one, p): p for p in chosen}
        for i, future in enumerate(as_completed(futures), 1):
            place = futures[future]
            try:
                r = future.result()
            except Exception as e:  # one bad site should never stop the run
                r = {
                    "place_id": place["place_id"], "name": place["name"],
                    "neighborhood": place["neighborhood"], "website": place.get("website"),
                    "crawl_status": "error", "error": f"{type(e).__name__}: {e}",
                    "pages_fetched": [], "deal_text_found": False,
                    "says_happy_hour": False, "snippets": [],
                }
            results.append(r)
            print(f"[{i}/{len(chosen)}] {r['name']}: {crawl_label(r)}")

    results.sort(key=lambda r: (r["neighborhood"], r["name"]))
    CRAWL_FILE.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "top_n": top_n,
        "bars": results,
    }, indent=2))

    print("\nSummary")
    statuses = Counter(r["crawl_status"] for r in results)
    for status, count in statuses.most_common():
        print(f"  {status}: {count}")
    for hood in sorted({r["neighborhood"] for r in results}):
        hood_rows = [r for r in results if r["neighborhood"] == hood]
        with_deals = sum(r["deal_text_found"] for r in hood_rows)
        with_hh = sum(r["says_happy_hour"] for r in hood_rows)
        print(f"  {hood}: {with_deals}/{len(hood_rows)} have deal text "
              f"({with_hh} say 'happy hour')")
    print(f"\nSaved to {CRAWL_FILE.relative_to(ROOT)}")


# --- Stage 3: extract ---

def pilot_order(bars: list[dict]) -> list[dict]:
    """Stable pseudo-random order, so a --limit pilot mixes neighborhoods."""
    return sorted(bars, key=lambda b: hashlib.sha1(b["place_id"].encode()).hexdigest())


def print_bar_result(name: str, result: dict) -> None:
    print(f"\n=== {name}: {result['bar_status']} ===")
    if result["llm_error"]:
        print(f"  model error: {result['llm_error']}")
    if result["notes"]:
        print(f"  model notes: {result['notes']}")
    if result["food_only_dropped"]:
        print(f"  food-only deals dropped: {result['food_only_dropped']}")
    for deal in result["deals"]:
        days = ",".join(deal["days"]) or "?"
        print(f"  [{deal['status']}] {deal['label']}: {days} {deal['start']} to {deal['end']}")
        for sp in deal["specials"]:
            price = f"${sp['price']:g}" if sp["price"] is not None else ""
            extra = " ".join(x for x in (price, sp["discount"] or "") if x)
            print(f"      {sp['category']}: {sp['item']} {extra}".rstrip())
        if deal["conditions"]:
            print(f"      conditions: {deal['conditions']}")
        print(f"      evidence: \"{deal['evidence']}\"")
        if deal["failed_checks"]:
            print(f"      failed checks: {', '.join(deal['failed_checks'])}")


def run_extract(name_filter: str | None, limit: int | None, workers: int,
                only_status: str | None = None) -> None:
    if not CRAWL_FILE.exists() or not DISCOVER_FILE.exists():
        sys.exit("Run the discover and crawl stages first.")
    places = {p["place_id"]: p for p in json.loads(DISCOVER_FILE.read_text())["places"]}
    crawled = json.loads(CRAWL_FILE.read_text())["bars"]

    targets = [b for b in crawled if b.get("deal_text_found")]
    if only_status:
        if not EXTRACT_FILE.exists():
            sys.exit("No earlier results yet, so --only-status has nothing to filter on.")
        previous = json.loads(EXTRACT_FILE.read_text()).get("results", {})
        targets = [b for b in targets if previous.get(b["place_id"], {}).get("bar_status") == only_status]
        if not targets:
            sys.exit(f"No bars currently have status '{only_status}'.")
    if name_filter:
        targets = [b for b in targets if name_filter.lower() in b["name"].lower()]
        if not targets:
            sys.exit(f"No bar with deal text matches '{name_filter}'.")
    if limit:
        targets = pilot_order(targets)[:limit]
    verbose = bool(name_filter) or (limit is not None and limit <= 20)

    # Merge into earlier results, so a pilot never wipes a full run
    existing = {}
    if EXTRACT_FILE.exists():
        existing = json.loads(EXTRACT_FILE.read_text()).get("results", {})

    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting deals for {len(targets)} bars, {workers} at a time...")

    def work(bar: dict) -> tuple[dict, dict]:
        address = places.get(bar["place_id"], {}).get("address", "")
        return bar, extract_deals(bar["name"], address, bar["snippets"], cache_dir=LLM_CACHE_DIR)

    new_results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, b) for b in targets]
        for i, future in enumerate(as_completed(futures), 1):
            bar, result = future.result()
            result.update({"name": bar["name"], "neighborhood": bar["neighborhood"]})
            new_results[bar["place_id"]] = result
            if verbose:
                print_bar_result(bar["name"], result)
            else:
                counts = Counter(d["status"] for d in result["deals"])
                detail = ", ".join(f"{n} {s}" for s, n in counts.items()) or "no deals"
                print(f"[{i}/{len(targets)}] {bar['name']}: {result['bar_status']} ({detail})")

    existing.update(new_results)
    EXTRACT_FILE.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": existing,
    }, indent=2))

    print("\nSummary (this run)")
    for status, count in Counter(r["bar_status"] for r in new_results.values()).most_common():
        print(f"  bars {status}: {count}")
    deal_counts = Counter(d["status"] for r in new_results.values() for d in r["deals"])
    print(f"  deals: {dict(deal_counts)}")
    print(f"  food-only deals dropped: {sum(r['food_only_dropped'] for r in new_results.values())}")
    failed = Counter(c for r in new_results.values() for d in r["deals"] for c in d["failed_checks"])
    if failed:
        print(f"  failed checks: {dict(failed.most_common())}")
    print(f"\nSaved to {EXTRACT_FILE.relative_to(ROOT)} ({len(existing)} bars total)")


# --- Stage 4: export ---

TIME_VALUE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$|^open$|^close$")


def check_manual_deal(deal, where: str, errors: list[str]) -> dict | None:
    """Validate a hand-written deal from the overrides file."""
    if not isinstance(deal, dict):
        errors.append(f"{where}: each deal must be an object {{...}}")
        return None
    days = deal.get("days")
    if not isinstance(days, list) or not days or any(d not in DAY_CODES for d in days):
        errors.append(f"{where}: 'days' must be a list like [\"mon\", \"tue\"] using {DAY_CODES}")
    for field in ("start", "end"):
        if not isinstance(deal.get(field), str) or not TIME_VALUE.match(deal[field]):
            errors.append(f"{where}: '{field}' must be \"HH:MM\" in 24 hour time, \"open\", or \"close\"")
    specials = deal.get("specials", [])
    if not isinstance(specials, list):
        errors.append(f"{where}: 'specials' must be a list")
        specials = []
    clean_specials = []
    for sp in specials:
        if not isinstance(sp, dict) or not sp.get("item"):
            errors.append(f"{where}: each special needs at least an 'item'")
            continue
        category = sp.get("category", "drink")
        if category not in ("drink", "food"):
            errors.append(f"{where}: special category must be \"drink\" or \"food\"")
        clean_specials.append({"item": sp["item"], "category": category,
                               "price": sp.get("price"), "discount": sp.get("discount")})
    return {
        "label": deal.get("label") or "Happy hour",
        "days": [d for d in DAY_CODES if d in (days or [])],
        "start": deal.get("start"),
        "end": deal.get("end"),
        "specials": clean_specials,
        "conditions": deal.get("conditions"),
        "has_food": any(sp["category"] == "food" for sp in clean_specials),
        "evidence": deal.get("evidence") or "Added by hand",
        "source_url": deal.get("source_url"),
        "status": "verified",
        "failed_checks": [],
        "manual": True,
    }


def load_overrides(known_ids: set[str], neighborhoods: dict) -> dict:
    """Read data/manual_overrides.json. Exit with every problem listed if anything is wrong."""
    if not OVERRIDES_FILE.exists():
        return {}
    try:
        raw = json.loads(OVERRIDES_FILE.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"{OVERRIDES_FILE.name} is not valid JSON: line {e.lineno}, column {e.colno}: {e.msg}. "
                 f"Common causes: a missing comma, a trailing comma, or single quotes instead of double.")
    if not isinstance(raw, dict):
        sys.exit(f"{OVERRIDES_FILE.name} must be one object: {{\"place_id\": {{\"action\": ...}}, ...}}")

    errors: list[str] = []
    overrides = {}
    for place_id, entry in raw.items():
        if place_id.startswith("_"):  # keys like "_comment" are ignored
            continue
        where = f"overrides[{place_id}]"
        if not isinstance(entry, dict) or entry.get("action") not in ("approve", "drop", "replace", "add"):
            errors.append(f"{where}: needs \"action\": one of approve, drop, replace, add")
            continue
        action = entry["action"]
        if action in ("approve", "drop", "replace") and place_id not in known_ids:
            errors.append(f"{where}: unknown place_id. Copy it from the place_id column of the CSV, "
                          f"or use \"add\" for a bar that is not in the data")
            continue
        if action == "add":
            if place_id in known_ids:
                errors.append(f"{where}: this bar already exists, use \"replace\" instead of \"add\"")
                continue
            for field in ("name", "lat", "lng"):
                if field not in entry:
                    errors.append(f"{where}: \"add\" needs \"{field}\"")
            if entry.get("neighborhood") not in neighborhoods:
                errors.append(f"{where}: \"neighborhood\" must be one of {list(neighborhoods)}")
        deals = []
        if action in ("replace", "add"):
            if not isinstance(entry.get("deals"), list) or not entry["deals"]:
                errors.append(f"{where}: \"{action}\" needs a non-empty \"deals\" list")
            else:
                for i, deal in enumerate(entry["deals"]):
                    checked = check_manual_deal(deal, f"{where}.deals[{i}]", errors)
                    if checked:
                        deals.append(checked)
        overrides[place_id] = {**entry, "deals": deals}

    if errors:
        print(f"Problems in {OVERRIDES_FILE.name}, nothing was exported:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    return overrides


def public_deal(deal: dict) -> dict:
    keep = ("label", "days", "start", "end", "specials", "conditions", "has_food",
            "evidence", "source_url", "failed_checks")
    out = {k: deal.get(k) for k in keep}
    if deal.get("manual"):
        out["manual"] = True
    if deal.get("approved"):
        out["approved"] = True
    return out


def run_export() -> None:
    for f in (DISCOVER_FILE, CRAWL_FILE, EXTRACT_FILE):
        if not f.exists():
            sys.exit(f"Missing {f.relative_to(ROOT)}. Run the earlier stages first.")
    neighborhoods = json.loads(NEIGHBORHOODS_FILE.read_text())
    places = json.loads(DISCOVER_FILE.read_text())["places"]
    crawl_doc = json.loads(CRAWL_FILE.read_text())
    crawl = {b["place_id"]: b for b in crawl_doc["bars"]}
    extract = json.loads(EXTRACT_FILE.read_text())["results"]
    overrides = load_overrides({p["place_id"] for p in places}, neighborhoods)
    last_checked = crawl_doc.get("generated_at", "")[:10]

    bars, csv_rows, warnings = [], [], []
    for place in places:
        pid = place["place_id"]
        c = crawl.get(pid)
        x = extract.get(pid)

        if x is not None:
            status = x["bar_status"]
            deals = [d for d in x["deals"] if d["status"] == "verified"]
            pending = [d for d in x["deals"] if d["status"] == "needs_review"]
            all_deals = x["deals"]
        else:
            if c is None:
                status = "not_crawled"
            elif c["crawl_status"] == "ok":
                status = "no_deal_text" if not c.get("deal_text_found") else "not_extracted"
            else:
                status = c["crawl_status"]
            deals, pending, all_deals = [], [], []

        manual = False
        ov = overrides.get(pid)
        if ov:
            if ov["action"] == "approve":
                # Only deals with readable days and times can be approved; the rest need "replace"
                usable = [d for d in pending if d["days"] and d["start"] and d["end"]]
                unusable = [d for d in pending if d not in usable]
                for d in unusable:
                    warnings.append(f"{place['name']}: deal '{d['label']}' has unreadable days or times "
                                    f"({', '.join(d['failed_checks'])}). Use \"replace\" to write it by hand.")
                deals = deals + [{**d, "approved": True} for d in usable]
                pending = unusable
                status = "verified" if deals else status
            elif ov["action"] == "drop":
                deals, pending, status = [], [], "dropped"
            elif ov["action"] == "replace":
                deals, pending, status, manual = ov["deals"], [], "verified", True

        bars.append({
            **place,
            "hh_status": status,
            "hh_has_food": any(d.get("has_food") for d in deals),
            "deals": [public_deal(d) for d in deals],
            "pending_deals": [public_deal(d) for d in pending],
            "manual": manual,
            "last_checked": last_checked,
        })

        review_deals = ov["deals"] if (ov and ov["action"] == "replace") else all_deals
        for i, d in enumerate(review_deals):
            csv_rows.append({
                "place_id": pid, "bar": place["name"], "neighborhood": place["neighborhood"],
                "bar_status": status, "deal_index": i,
                "deal_status": "verified (manual)" if d.get("manual") else d["status"],
                "failed_checks": "; ".join(d.get("failed_checks") or []),
                "label": d["label"], "days": ",".join(d["days"]),
                "start": d["start"], "end": d["end"],
                "specials": "; ".join(
                    f"{sp['category']}: {sp['item']}"
                    + (f" ${sp['price']:g}" if isinstance(sp.get("price"), (int, float)) else "")
                    + (f" {sp['discount']}" if sp.get("discount") else "")
                    for sp in d["specials"]),
                "conditions": d.get("conditions") or "",
                "evidence": d.get("evidence") or "",
                "source_url": d.get("source_url") or "",
                "website": place.get("website") or "",
            })
        if x is not None and x["bar_status"] == "llm_error":
            csv_rows.append({"place_id": pid, "bar": place["name"], "neighborhood": place["neighborhood"],
                             "bar_status": status, "deal_status": "llm_error",
                             "failed_checks": x.get("llm_error") or "", "website": place.get("website") or ""})

    # Bars added by hand
    for pid, ov in overrides.items():
        if ov["action"] != "add":
            continue
        bars.append({
            "place_id": pid, "name": ov["name"], "address": ov.get("address", ""),
            "lat": ov["lat"], "lng": ov["lng"], "neighborhood": ov["neighborhood"],
            "primary_type": "bar", "types": ["bar"], "rating": None, "rating_count": None,
            "price_level": None, "website": ov.get("website"), "maps_url": None,
            "hours_text": ov.get("hours_text", []), "hours_periods": [],
            "hh_status": "verified", "hh_has_food": any(d["has_food"] for d in ov["deals"]),
            "deals": [public_deal(d) for d in ov["deals"]], "pending_deals": [],
            "manual": True, "last_checked": last_checked,
        })

    OUTPUT_JSON.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Only 'deals' are verified. 'pending_deals' failed a check and are not used by the agent.",
        "neighborhoods": {k: v["name"] for k, v in neighborhoods.items()},
        "bars": bars,
    }, indent=2))

    order = {"needs_review": 0, "verified": 1, "verified (manual)": 2, "rejected": 3, "llm_error": 4}
    csv_rows.sort(key=lambda r: (r["neighborhood"], order.get(r["deal_status"], 9), r["bar"], r.get("deal_index", 0)))
    columns = ["place_id", "bar", "neighborhood", "bar_status", "deal_index", "deal_status",
               "failed_checks", "label", "days", "start", "end", "specials", "conditions",
               "evidence", "source_url", "website"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel reads accents
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(csv_rows)

    print("Export summary")
    for hood_key, hood in neighborhoods.items():
        rows = [b for b in bars if b["neighborhood"] == hood_key]
        verified = sum(1 for b in rows if b["deals"])
        only_pending = sum(1 for b in rows if not b["deals"] and b["pending_deals"])
        with_food = sum(1 for b in rows if b["hh_has_food"])
        print(f"  {hood['name']}: {verified} bars with verified deals ({with_food} include food), "
              f"{only_pending} more bars with deals only waiting on review, {len(rows)} bars total")
    if overrides:
        print(f"  overrides applied: {dict(Counter(o['action'] for o in overrides.values()))}")
    else:
        print(f"  no {OVERRIDES_FILE.name} found (optional)")
    for w in warnings:
        print(f"  warning: {w}")
    print(f"\nSaved {OUTPUT_JSON.relative_to(ROOT)} and {OUTPUT_CSV.relative_to(ROOT)}")


# --- Preview: polygons + bars as GeoJSON for geojson.io ---

def run_preview() -> None:
    neighborhoods = json.loads(NEIGHBORHOODS_FILE.read_text())
    features = []
    for key, hood in neighborhoods.items():
        ring = [[lng, lat] for lat, lng in hood["polygon"]]
        ring.append(ring[0])
        features.append({
            "type": "Feature",
            "properties": {"name": hood["name"]},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    if DISCOVER_FILE.exists():
        for p in json.loads(DISCOVER_FILE.read_text())["places"]:
            features.append({
                "type": "Feature",
                "properties": {"name": p["name"], "neighborhood": p["neighborhood"]},
                "geometry": {"type": "Point", "coordinates": [p["lng"], p["lat"]]},
            })

    PREVIEW_FILE.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))
    print(f"Wrote {PREVIEW_FILE.relative_to(ROOT)}. Drag it onto https://geojson.io to view.")


# --- CLI ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the happy hour dataset.")
    sub = parser.add_subparsers(dest="stage", required=True)

    d = sub.add_parser("discover", help="Find bars with Places API")
    d.add_argument("--neighborhood", help="Only run one neighborhood, e.g. west_village")

    sub.add_parser("preview", help="Write polygons and bars to GeoJSON for checking")

    c = sub.add_parser("crawl", help="Fetch bar websites and find deal text")
    c.add_argument("--top", type=int, default=TOP_N_DEFAULT,
                   help="Bars per neighborhood, by review count (default: all)")
    c.add_argument("--workers", type=int, default=CRAWL_WORKERS_DEFAULT)

    x = sub.add_parser("extract", help="Gemini reads deal text, code checks it")
    x.add_argument("--name", help="Only bars whose name contains this text, printed in full")
    x.add_argument("--limit", type=int, help="Only this many bars (a mixed sample), printed in full if 20 or fewer")
    x.add_argument("--workers", type=int, default=EXTRACT_WORKERS_DEFAULT)
    x.add_argument("--only-status", choices=["no_deal", "needs_review", "rejected", "llm_error", "verified"],
                   help="Only re-extract bars whose current result has this status")

    sub.add_parser("export", help="Write happy_hours.json and the review CSV")

    args = parser.parse_args()
    if args.stage == "discover":
        run_discover(args.neighborhood)
    elif args.stage == "preview":
        run_preview()
    elif args.stage == "crawl":
        run_crawl(args.top, args.workers)
    elif args.stage == "extract":
        run_extract(args.name, args.limit, args.workers, args.only_status)
    elif args.stage == "export":
        run_export()


if __name__ == "__main__":
    main()
