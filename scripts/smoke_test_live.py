"""Test the two external tools against the real Google Places API and a real bar website.

Usage:
    uv run scripts/smoke_test_live.py

Uses 2 Places searches (out of your 100/day quota) and 1 Gemini call.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import places  # noqa: E402
from tools import check_happy_hour_online, find_bars  # noqa: E402


def show_find(title: str, **kwargs) -> dict:
    print(f"\n=== find_bars: {title} ===")
    start = time.time()
    r = json.loads(find_bars(**kwargs))
    if "error" in r:
        print(f"  error: {r['error']}")
        return r
    print(f"  source: {r['source']}  ({time.time() - start:.1f}s)")
    if r["source"] != "google_places_live":
        print(f"  !! NOT LIVE. Reason: {r.get('fallback_reason')}")
        print("     Check PLACES_API_KEY in .env, that the key allows Places API (New), and the daily quota.")
    print(f"  counts: {r['counts']}  filtered out: {r.get('filtered_out')}")
    for b in r["results"]:
        print(f"  {b['name'][:30]:30} {b.get('rating')!s:4} {b.get('price') or '':4} {b['status'][:30]:30} "
              f"| {b['happy_hour'][:55]}")
        if b.get("description") or b.get("features"):
            print(f"      {b.get('description') or ''} {b.get('features') or ''}")
    return r


if __name__ == "__main__":
    first = show_find("East Village cocktail bars", neighborhood="East Village", style="cocktail")
    second = show_find("UWS dive bars open Friday 11pm", neighborhood="Upper West Side", query="dive",
                       day="friday", time="11pm")

    # Pick a bar we have no happy hour data for, and check its website live
    pick = None
    for r in (first, second):
        for b in r.get("results", []):
            if b["happy_hour"].startswith("not in our happy hour data") and b.get("website"):
                pick = b
                break
        if pick:
            break

    print("\n=== check_happy_hour_online ===")
    if not pick:
        print("  No result lacked happy hour data, so there was nothing new to check. That's fine.")
    else:
        print(f"  checking {pick['name']} ({pick['website']}) ...")
        start = time.time()
        r = json.loads(check_happy_hour_online(pick["place_id"]))
        print(f"  took {time.time() - start:.1f}s")
        print("  " + json.dumps(r, indent=2).replace("\n", "\n  ")[:2500])

    print(f"\nLive Places searches used by this app today: {places.usage_today()} of {places.DAILY_LIMIT}")
