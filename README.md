# Next Round

Happy hours and bar crawls for the West Village, East Village, and Upper West Side.

**Live app:** https://nyc-bar-agent-git-808237570445.europe-west1.run.app

## What it is

I like bar hopping in Manhattan, but finding happy hours is a pain. They're buried on bar websites, menu PDFs, and Instagram, and a lot of what's listed online is out of date.

Next Round is a chat agent for planning a night out. Ask it what's on at a certain day and time, or look for a bar by style or vibe. It can also plan the walking route. The planner tries every order of your bars and picks the one that catches the most happy hour time with the least walking.

Every happy hour in the data was checked against the bar's own website. If the agent doesn't have a verified deal for a bar, it says so instead of guessing.

## Tools

| Tool | What it does | Data source |
|---|---|---|
| `find_bars` | Searches bars by neighborhood, style, rating, price, features, and whether they're open at a given time | Google Places API, live |
| `get_happy_hours` | Lists the happy hours running at a day and time, with end times and specials | My verified dataset |
| `plan_bar_crawl` | Orders 2 to 6 bars into the route that catches the most happy hour with the least walking | Computed from hours and deals |
| `check_happy_hour_online` | Reads one bar's website live and pulls out its happy hours, checking each one against the page | The bar's website + Gemini, live |

`find_bars` and `check_happy_hour_online` call outside services every time they run. `find_bars` results include a `source` field that says `google_places_live`.

## How to use it

Type a question or tap one of the suggestions. Each answer lists the tools it used, and "details" shows the exact arguments and results. Crawl plans come with "Open in Google Maps" and "Copy plan" buttons. Bar names in an answer link to their spot on the map.

Try these:

1. Plan 3 stops in the West Village Friday starting at 5pm, cocktails please
2. Find me a dive bar on the Upper West Side open late Friday
3. Wine bars rated 4 stars and up with happy hour food on the Upper West Side, Saturday 4pm

It remembers the conversation, so follow-ups work. After the first one, try "start at 7 instead" or "make it wine bars."

## How the happy hour data was built

I couldn't find an API with happy hour data, so I built my own with `scripts/build_hh_dataset.py`:

1. Searched for bars with the Google Places API and kept the 327 inside the three neighborhood boundaries.
2. Crawled each bar's website (the homepage plus up to three menu or specials pages) and kept the text around deal language.
3. Had Gemini pull out the days, times, and specials, quoting the page word for word.
4. Checked every quote against the page in code. Anything that wasn't on the page got thrown out, and deals with unreadable times or mismatched days were held back for review.

79 bars came out with verified happy hours. I collected the data in October 2026, so some deals have probably changed since.

## Project structure

```
app.py                 FastAPI server, agent loop, sessions
prompts.py             system prompt
tools.py               the four tools and their schemas
bar_data.py            hours, happy hour windows, bar lookup, walking times
places.py              Google Places client with caching and a daily limit
hh_extract.py          website crawler and deal-text finder
hh_deals.py            Gemini extraction and quote checking
index.html             frontend (chat, map, crawl buttons)
data/
  happy_hours.json     the dataset the app reads
  neighborhoods.json   neighborhood boundaries
scripts/
  build_hh_dataset.py  data pipeline: discover, crawl, extract, export
  smoke_test_*.py      tool tests on real data
  test_conversations.py  scripted conversations against the live agent
```
