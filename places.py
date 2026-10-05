"""Live Google Places search for find_bars, with a cache and a daily request limit.

Every failure raises PlacesError with a short reason, so the tool can fall back
to the local dataset and tell the model why.
"""

import json
import os
import threading
import time
from pathlib import Path

import requests

import bar_data as bd

try:  # load .env when running locally; on Cloud Run the key comes from the environment
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
NEIGHBORHOODS_FILE = Path(__file__).parent / "data" / "neighborhoods.json"

# Enterprise + Atmosphere fields: description, serves cocktails/wine/beer, outdoor seating, live music, groups
FIELD_MASK = ",".join([
    "places.id", "places.displayName", "places.formattedAddress", "places.location",
    "places.types", "places.primaryType", "places.rating", "places.userRatingCount",
    "places.priceLevel", "places.regularOpeningHours", "places.websiteUri", "places.googleMapsUri",
    "places.businessStatus", "places.editorialSummary", "places.servesCocktails", "places.servesWine",
    "places.servesBeer", "places.outdoorSeating", "places.liveMusic", "places.goodForGroups",
])

DAILY_LIMIT = 90  # below the 100/day GCP quota, so we fall back before Google refuses
CACHE_SECONDS = 6 * 3600
TIMEOUT_SECONDS = 8

BAR_TYPES = {
    "bar", "pub", "wine_bar", "cocktail_bar", "night_club", "brewery", "brewpub",
    "beer_garden", "sports_bar", "lounge_bar", "irish_pub", "bar_and_grill",
    "gastropub", "hookah_bar",
}

_lock = threading.Lock()
_cache: dict[str, tuple[float, list[dict]]] = {}
_usage = {"date": None, "count": 0}


class PlacesError(Exception):
    """Live search failed; the message is the reason shown to the model."""


def _polygons() -> dict:
    return json.loads(NEIGHBORHOODS_FILE.read_text())


def _point_in_polygon(lat: float, lng: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        lat_i, lng_i = polygon[i]
        lat_j, lng_j = polygon[j]
        if (lat_i > lat) != (lat_j > lat):
            if lng < lng_i + (lat - lat_i) * (lng_j - lng_i) / (lat_j - lat_i):
                inside = not inside
        j = i
    return inside


def _rectangle(polygon: list[list[float]]) -> dict:
    lats, lngs = [p[0] for p in polygon], [p[1] for p in polygon]
    return {"low": {"latitude": min(lats), "longitude": min(lngs)},
            "high": {"latitude": max(lats), "longitude": max(lngs)}}


def _take_request_slot() -> None:
    today = bd.now_nyc().date().isoformat()
    with _lock:
        if _usage["date"] != today:
            _usage.update(date=today, count=0)
        if _usage["count"] >= DAILY_LIMIT:
            raise PlacesError(f"daily limit of {DAILY_LIMIT} live searches reached")
        _usage["count"] += 1


def usage_today() -> int:
    return _usage["count"] if _usage["date"] == bd.now_nyc().date().isoformat() else 0


def _slim(place: dict, hood: str) -> dict:
    hours = place.get("regularOpeningHours", {})
    return {
        "place_id": place["id"],
        "name": place.get("displayName", {}).get("text", ""),
        "address": place.get("formattedAddress", ""),
        "lat": place["location"]["latitude"],
        "lng": place["location"]["longitude"],
        "neighborhood": hood,
        "primary_type": place.get("primaryType", ""),
        "types": place.get("types", []),
        "rating": place.get("rating"),
        "rating_count": place.get("userRatingCount"),
        "price_level": place.get("priceLevel"),
        "website": place.get("websiteUri"),
        "maps_url": place.get("googleMapsUri"),
        "hours_text": hours.get("weekdayDescriptions", []),
        "hours_periods": hours.get("periods", []),
        "summary": (place.get("editorialSummary") or {}).get("text"),
        "serves_cocktails": place.get("servesCocktails"),
        "serves_wine": place.get("servesWine"),
        "serves_beer": place.get("servesBeer"),
        "outdoor_seating": place.get("outdoorSeating"),
        "live_music": place.get("liveMusic"),
        "good_for_groups": place.get("goodForGroups"),
    }


def search(hood: str, text_query: str) -> list[dict]:
    """Live Text Search inside one neighborhood. Returns bars in the same shape as the dataset.

    Raises PlacesError on any failure. Results are cached for CACHE_SECONDS.
    """
    cache_key = f"{hood}|{text_query.lower()}"
    with _lock:
        hit = _cache.get(cache_key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]

    api_key = os.environ.get("PLACES_API_KEY")
    if not api_key:
        raise PlacesError("PLACES_API_KEY is not set on the server")
    polygon = _polygons()[hood]["polygon"]
    _take_request_slot()

    try:
        resp = requests.post(
            PLACES_URL,
            json={"textQuery": text_query, "pageSize": 20,
                  "locationRestriction": {"rectangle": _rectangle(polygon)}},
            headers={"Content-Type": "application/json", "X-Goog-Api-Key": api_key,
                     "X-Goog-FieldMask": FIELD_MASK},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise PlacesError(f"could not reach Google Places ({type(e).__name__})")
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("status", "")
        except ValueError:
            detail = ""
        raise PlacesError(f"Google Places returned HTTP {resp.status_code} {detail}".strip())
    try:
        raw = resp.json().get("places", [])
    except ValueError:
        raise PlacesError("Google Places returned an unreadable response")

    bars = []
    for p in raw:
        loc = p.get("location") or {}
        if "id" not in p or "latitude" not in loc:
            continue
        if p.get("businessStatus", "OPERATIONAL") != "OPERATIONAL":
            continue
        if not set(p.get("types", [])) & BAR_TYPES:
            continue
        if not _point_in_polygon(loc["latitude"], loc["longitude"], polygon):
            continue
        bars.append(_slim(p, hood))

    with _lock:
        _cache[cache_key] = (time.time(), bars)
    return bars
