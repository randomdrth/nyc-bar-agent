"""Fetch a bar's website and pull out the text that describes its deals.

Used by scripts/build_hh_dataset.py (offline build) and later by the live
check_happy_hour_online tool in tools.py.

Many bars never write "happy hour". They write "1/2 price whole bar, 3-8pm".
So we look for deal language, keep only text windows that also contain a time
range, price, or weekday, and rank those windows so navigation menus and other
noise never crowd out the real deals.
"""

import hashlib
import io
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

USER_AGENT = "nyc-bar-agent/0.1 (Columbia University class project)"
TIMEOUT = 12
MAX_PAGES = 4
MAX_PAGE_CHARS = 60_000
DELAY_BETWEEN_PAGES = 1.0

# Websites that are not the bar's own site. Nothing useful to crawl.
NOT_OWN_SITE = (
    "instagram.com", "facebook.com", "linktr.ee", "tiktok.com", "x.com", "twitter.com",
    "yelp.com", "resy.com", "opentable.com", "sevenrooms.com", "toasttab.com",
    "doordash.com", "ubereats.com", "grubhub.com",
)

# How promising a link is, based on its text and URL.
LINK_WEIGHTS = {
    "happy": 10, "special": 5, "deal": 5,
    "drink": 3, "cocktail": 3, "menu": 3,
    "wine": 1, "beer": 1, "bar": 1,
}

# --- Deal detection patterns ---

# The literal phrase. Strongest signal.
STRONG_HH = re.compile(r"happy[\s\-]*h(ou)?r|\bhh\b", re.IGNORECASE)

# Words that start a window worth looking at.
DEAL_TRIGGER = re.compile(
    r"happy[\s\-]*h(ou)?r|\bhh\b"
    r"|1/2\s*(price|off)|half[\s\-]*(price|off)|\d{1,2}\s*%\s*off"
    r"|\b(2|two)[\s\-]*(for|4)[\s\-]*(1|one)\b|\bbogo\b"
    r"|\bspecials?\b|\bdeals?\b",
    re.IGNORECASE,
)

# 3pm-8pm, 3-7 PM, 4:30 to 7pm, 8pm-close, 8pm- CL
TIME_RANGE = re.compile(
    r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\s*(-|–|to)\s*(\d{1,2}(:\d{2})?\s*(am|pm)?|close|cl)\b",
    re.IGNORECASE,
)

DAY = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)s?\b",
    re.IGNORECASE,
)

PRICE = re.compile(r"\$\s*\d")

_robots_cache: dict[str, RobotFileParser] = {}


# --- Helpers ---

def _domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def is_not_own_site(url: str) -> bool:
    d = _domain(url)
    return any(d == s or d.endswith("." + s) for s in NOT_OWN_SITE)


def allowed_by_robots(url: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _robots_cache:
        rp = RobotFileParser()
        try:
            r = requests.get(base + "/robots.txt", timeout=8, headers={"User-Agent": USER_AGENT})
            if r.status_code in (401, 403):
                rp.disallow_all = True
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except requests.RequestException:
            rp.parse([])
        _robots_cache[base] = rp
    return _robots_cache[base].can_fetch(USER_AGENT, url)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    lines = (re.sub(r"\s+", " ", line).strip() for line in soup.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)


def extract_links(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        url, _ = urldefrag(urljoin(base_url, a["href"]))
        if url.startswith(("http://", "https://")):
            links.append({"url": url, "text": a.get_text(" ", strip=True)[:80]})
    return links


def pdf_to_text(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:10])
    except Exception:
        return ""


# --- Fetching ---

def _fetch_live(url: str) -> dict:
    result = {"url": url, "final_url": url, "ok": False, "kind": None,
              "text": "", "links": [], "error": None, "transient": False}

    if not allowed_by_robots(url):
        result["error"] = "blocked by robots.txt"
        return result

    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as e:
        result["error"] = f"request failed: {type(e).__name__}"
        result["transient"] = True
        return result

    result["final_url"] = r.url
    if r.status_code >= 400:
        result["error"] = f"HTTP {r.status_code}"
        return result

    ctype = r.headers.get("Content-Type", "").lower()
    if "pdf" in ctype or r.url.lower().split("?")[0].endswith(".pdf"):
        result["kind"] = "pdf"
        result["text"] = pdf_to_text(r.content)
    elif "html" in ctype or not ctype:
        encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else "utf-8"
        html = r.content.decode(encoding, errors="replace")
        result["kind"] = "html"
        result["text"] = html_to_text(html)
        result["links"] = extract_links(html, r.url)
    else:
        result["error"] = f"unsupported content type: {ctype}"
        return result

    result["text"] = result["text"][:MAX_PAGE_CHARS]
    result["ok"] = True
    return result


def fetch(url: str, cache_dir: Path | None = None) -> dict:
    """Fetch one page. Cached on disk when cache_dir is set (network errors are not cached)."""
    path = None
    if cache_dir:
        path = Path(cache_dir) / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass  # half-written or corrupt cache file, fetch again

    result = _fetch_live(url)
    if path and not result["transient"]:
        # Write to a temp file, then rename. The rename is atomic, so another
        # thread never reads a half-written file (two bars can share a site).
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(result))
        os.replace(tmp, path)
    return result


# --- Crawling one bar ---

def score_link(link: dict, home_domain: str) -> int:
    url = link["url"]
    label = (link["text"] + " " + url).lower()
    is_pdf = url.lower().split("?")[0].endswith(".pdf")
    if _domain(url) != home_domain and not is_pdf:
        return 0
    score = sum(weight for word, weight in LINK_WEIGHTS.items() if word in label)
    return score + (2 if is_pdf and score else 0)


def crawl_site(url: str | None, cache_dir: Path | None = None, max_pages: int = MAX_PAGES) -> dict:
    if not url:
        return {"status": "no_website", "pages": []}
    if is_not_own_site(url):
        return {"status": "not_own_site", "pages": []}

    home = fetch(url, cache_dir)
    if not home["ok"] and home["error"] == "HTTP 404":
        # Stale location page from Google. Try the site's homepage instead.
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}/"
        if root.rstrip("/") != url.rstrip("/"):
            url = root
            home = fetch(url, cache_dir)
    if not home["ok"]:
        return {"status": "fetch_failed", "error": home["error"], "pages": []}

    pages = [home]
    home_domain = _domain(home["final_url"])
    seen = {url.rstrip("/"), home["final_url"].rstrip("/")}
    unique_links = {link["url"]: link for link in home["links"]}.values()
    ranked = sorted(unique_links, key=lambda l: score_link(l, home_domain), reverse=True)

    for link in ranked:
        if len(pages) >= max_pages or score_link(link, home_domain) == 0:
            break
        if link["url"].rstrip("/") in seen:
            continue
        seen.add(link["url"].rstrip("/"))
        time.sleep(DELAY_BETWEEN_PAGES)
        page = fetch(link["url"], cache_dir)
        if page["ok"]:
            pages.append(page)

    total_chars = sum(len(p["text"]) for p in pages)
    status = "ok" if total_chars >= 300 else "empty_or_js"
    return {"status": status, "pages": pages}


# --- Finding deal text ---

def score_window(text: str) -> int:
    """How much a window looks like a real deal. 0 means drop it."""
    strong = bool(STRONG_HH.search(text))
    times = len(TIME_RANGE.findall(text))
    prices = len(PRICE.findall(text))
    days = len(DAY.findall(text))
    if not strong and not (times and (prices or days)):
        return 0
    return 5 * strong + 2 * min(times, 4) + min(prices, 4) + min(days, 4)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _uncovered(start: int, end: int, covered: list[tuple[int, int]]) -> int:
    """Characters in [start, end) not already inside a covered interval."""
    overlap = sum(max(0, min(end, e) - max(start, s)) for s, e in _merge_intervals(covered))
    return (end - start) - overlap


def find_hh_snippets(pages: list[dict], before: int = 150, after: int = 700,
                     max_chars: int = 6000, min_new_chars: int = 120) -> list[dict]:
    """Return the most deal-like text from a bar's pages, best first.

    Each trigger match (deal word or time range) gets its own window. Windows
    are scored and picked best first. A window is kept only if it adds at least
    min_new_chars of text not already picked, and the budget counts only new
    text. So repeated nav links ("Specials" in the menu bar) cannot crowd out
    the real deals, and nearby deals are not lost to overlap.
    """
    candidates = []
    for page_idx, page in enumerate(pages):
        text = page["text"]
        # Deal words start a window, and so do time ranges, because many deals
        # are written as "Mon-Fri 3-7 PM $6 apps" with no deal word at all.
        triggers = list(DEAL_TRIGGER.finditer(text)) + list(TIME_RANGE.finditer(text))
        for m in triggers:
            start, end = max(0, m.start() - before), min(len(text), m.end() + after)
            score = score_window(text[start:end])
            if score:
                candidates.append({"page_idx": page_idx, "start": start, "end": end, "score": score})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    covered: dict[int, list[tuple[int, int]]] = {}
    best_score: dict[tuple[int, int, int], int] = {}
    total = 0
    for c in candidates:
        page_cov = covered.setdefault(c["page_idx"], [])
        new_chars = _uncovered(c["start"], c["end"], page_cov)
        needed = min(min_new_chars, c["end"] - c["start"])  # short pages still count
        if new_chars < needed or new_chars == 0 or total + new_chars > max_chars:
            continue
        page_cov.append((c["start"], c["end"]))
        best_score[(c["page_idx"], c["start"], c["end"])] = c["score"]
        total += new_chars

    # Join picked windows that touch, and rank each block by its best window.
    out = []
    for page_idx, intervals in covered.items():
        page = pages[page_idx]
        for start, end in _merge_intervals(intervals):
            score = max(s for (p, a, b), s in best_score.items()
                        if p == page_idx and a >= start and b <= end)
            chunk = page["text"][start:end]
            out.append({"url": page["final_url"], "text": chunk, "score": score,
                        "strong_hh": bool(STRONG_HH.search(chunk))})
    out.sort(key=lambda c: c["score"], reverse=True)
    return out
