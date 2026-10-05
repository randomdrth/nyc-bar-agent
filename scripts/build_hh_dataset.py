"""Build the happy hour dataset in stages.

Stage 1, discover: find bars in each neighborhood with Places API Text Search,
keep only those inside the neighborhood polygon, save to data/stage_discover.json.

Stage 2, crawl: fetch each bar's own website and keep the text that looks like
deals (happy hours, half price nights, timed specials). Saves to
data/stage_crawl.json. Raw pages are cached in data/cache/, so reruns are fast.

Usage:
    uv run scripts/build_hh_dataset.py discover
    uv run scripts/build_hh_dataset.py discover --neighborhood west_village
    uv run scripts/build_hh_dataset.py preview
    uv run scripts/build_hh_dataset.py crawl --top 5      # pilot: 5 per neighborhood
    uv run scripts/build_hh_dataset.py crawl              # every discovered bar
"""

import argparse
import json
import os
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

from hh_extract import crawl_site, find_hh_snippets  # noqa: E402

DATA = ROOT / "data"
CACHE_DIR = DATA / "cache.nosync"
NEIGHBORHOODS_FILE = DATA / "neighborhoods.json"
DISCOVER_FILE = DATA / "stage_discover.json"
CRAWL_FILE = DATA / "stage_crawl.json"
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

    args = parser.parse_args()
    if args.stage == "discover":
        run_discover(args.neighborhood)
    elif args.stage == "preview":
        run_preview()
    elif args.stage == "crawl":
        run_crawl(args.top, args.workers)


if __name__ == "__main__":
    main()
