"""Run the tools on the real data with fixed questions and print readable results.

Usage:
    uv run scripts/smoke_test_tools.py

Check the output against what you know about these bars. Fixed days and times
are used, so the results are the same whenever you run it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import get_happy_hours, plan_bar_crawl, run_tool  # noqa: E402


def show_happy_hours(title: str, **kwargs) -> None:
    print(f"\n=== {title} ===")
    r = json.loads(get_happy_hours(**kwargs))
    if "error" in r:
        print(f"  error: {r['error']}")
        return
    c = r["counts"]
    print(f"  {r['query']['when']}: {c['happening_now']} happening now, {c['starting_soon']} starting soon")
    for b in r["happening_now"][:5]:
        h = b["happy_hours"][0]
        print(f"  NOW   {b['name'][:30]:30} {h['deal'][:26]:26} ends {h['ends_at']:>10}  {', '.join(h['specials'][:2])}")
    for b in r["starting_soon"][:3]:
        h = b["happy_hours"][0]
        print(f"  SOON  {b['name'][:30]:30} {h['deal'][:26]:26} starts {h['starts_at']}")
    if r.get("suggestion"):
        print(f"  suggestion: {r['suggestion']}")


def show_crawl(title: str, **kwargs) -> None:
    print(f"\n=== {title} ===")
    r = json.loads(plan_bar_crawl(**kwargs))
    if "stops" not in r:
        print(f"  error: {r['error']}")
        return
    s = r["summary"]
    print(f"  {s['starts']} to {s['ends']}, {s['total_walk_minutes']} min walking, "
          f"{s['happy_hour_minutes']} min of happy hour")
    print(f"  why: {r['why_this_order']}")
    for stop in r["stops"]:
        hh = stop["happy_hour"]
        deal = f"{hh['deal']} ({hh['runs']})" if hh else "no happy hour"
        print(f"  {stop['stop']}. {stop['name'][:32]:32} arrive {stop['arrive']:>12}  {deal}")
    for w in r["warnings"]:
        print(f"  ! {w}")


if __name__ == "__main__":
    show_happy_hours("East Village, Thursday 6pm", neighborhood="East Village", day="thursday", time="6pm")
    show_happy_hours("West Village, Friday 5pm, cocktail bars", neighborhood="West Village", day="friday",
                     time="5pm", style="cocktail")
    show_happy_hours("Upper West Side, Monday 1am (Sunday night deals)", neighborhood="UWS", day="monday", time="1am")
    show_happy_hours("East Village, Saturday 4pm, wine with food, $$ max", neighborhood="EV", day="saturday",
                     time="4pm", style="wine", with_food=True, max_price=2)
    show_happy_hours("West Village, Tuesday 9am (should suggest a time)", neighborhood="WV", day="tuesday", time="9am")

    bars = ["Jake's Dilemma", "Vin Sur Vingt Wine Bar - Upper West Side", "The Consulate - UWS"]
    show_crawl("UWS crawl, Thursday 5pm, best order", bars=bars, day="thursday", start_time="5pm")
    show_crawl("Same bars, kept in the given order", bars=bars, day="thursday", start_time="5pm", keep_order=True)
    show_crawl("West Village, Monday 5pm, one bar closed Mondays",
               bars=["Automatic Slim's", "Down the Hatch", "Greenwich Treehouse"], day="monday", start_time="5pm")

    print("\n=== Error handling (as the model would see it) ===")
    print(" ", run_tool("get_happy_hours", {"neighborhood": "SoHo"}))
    print(" ", run_tool("plan_bar_crawl", {"bars": ["Vin Sur Vingt", "Jake's Dilemma"]})[:200])
    print(" ", run_tool("get_weather", {"location": "NYC"}))
