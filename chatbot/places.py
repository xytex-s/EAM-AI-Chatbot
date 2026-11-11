"""
places.py - local recommendations from OpenStreetMap (Ahmed)

    lookup(question: str) -> list[dict] | None
        None -> not a question about places nearby, so /ask carries on
        list -> places, closest first. Each one:
                {"name": str, "type": str, "distance_m": int, "map_url": str}
                The list can be empty if OpenStreetMap is down and nothing is cached.

    summary(question: str, places: list) -> str
        The sentence /ask uses as "answer", e.g.
        "Here are places to eat within about a 10-minute walk of City Campus."

Data comes from OpenStreetMap through the free Overpass API (no key, no cost).
Only the campus location and the place types are sent to Overpass. The
student's question never leaves our server.

Results are cached in data/places_cache.json for 7 days. If Overpass is slow,
down, or the VM has no internet, an older cached result is used instead.
Before a demo, fill the cache (needs internet):

    python places.py --warm

Quick manual test:

    python places.py "Where can I get lunch near Collegiate Crescent?"
"""

import json
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Overpass asks every app to identify itself
USER_AGENT = "SHU-EAM-Group7-Chatbot-PoC/1.0 (student project, Sheffield Hallam University)"
TIMEOUT_S = 10                  # give up on Overpass after 10 seconds
RADIUS_M = 800                  # about a 10-minute walk
MAX_RESULTS = 8
CACHE_MAX_AGE_S = 7 * 24 * 3600
CACHE_FILE = Path(__file__).parent / "data" / "places_cache.json"

# [CHECK] Campus centre points (latitude, longitude). Looked up 11 Nov 2026:
#   City Campus: Howard Street, which runs through City Campus (Wikipedia, "Howard Street")
#   Collegiate Crescent: SHU Main Building on Collegiate Crescent (britishlistedbuildings.co.uk/101270958)
# Ahmed: check both on openstreetmap.org (right-click > "Show address").
CAMPUSES = {
    "City Campus": (53.3783, -1.4657),
    "Collegiate Crescent": (53.3724, -1.4915),
}
DEFAULT_CAMPUS = "City Campus"

# Which OpenStreetMap tags each kind of place means
KIND_TAGS = {
    "cafe": [("amenity", "cafe")],
    "food": [("amenity", "cafe"), ("amenity", "restaurant"), ("amenity", "fast_food")],
    "shop": [("shop", "supermarket"), ("shop", "convenience")],
}

# Words that ask for each kind
KIND_WORDS = {
    "cafe": ["coffee", "cafe", "cafes", "coffee shop", "coffee shops"],
    "food": ["food", "eat", "eating", "lunch", "breakfast", "brunch", "dinner",
             "restaurant", "restaurants", "takeaway", "takeaways", "snack", "snacks",
             "sandwich", "sandwiches", "meal", "meals", "places to eat"],
    "shop": ["shop", "shops", "supermarket", "supermarkets", "grocery", "groceries",
             "convenience store", "corner shop"],
}

# A places question also needs a word about location, so "Is food included
# in my accommodation?" goes to the normal search instead.
LOCATION_WORDS = ["near", "nearby", "nearest", "closest", "close", "around", "walk",
                  "walking", "local", "campus", "where", "recommend", "best", "good", "cheap"]

# These are support questions, not places questions
EXCLUDE = ["eating disorder", "eating disorders", "food bank", "foodbank",
           "food poverty", "hardship", "allergy", "allergies"]


def _normalise(text: str) -> str:
    """'Cafés near City Campus?' -> ' cafes near city campus ' (accents removed, padded)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " " + " ".join(text.split()) + " "


def _has(text: str, words: list[str]) -> bool:
    return any(f" {w} " in text for w in words)


def kinds_asked(question: str) -> set[str]:
    """Which kinds of place a question asks for: a set of 'cafe', 'food', 'shop'."""
    text = _normalise(question)
    # "coffee shop" means a cafe, not a supermarket
    text = text.replace(" coffee shops ", " coffee ").replace(" coffee shop ", " coffee ")
    return {kind for kind, words in KIND_WORDS.items() if _has(text, words)}


def is_places_question(question: str) -> bool:
    if not isinstance(question, str):
        return False
    text = _normalise(question)
    return bool(kinds_asked(question)) and _has(text, LOCATION_WORDS) and not _has(text, EXCLUDE)


def campus_for(question: str) -> str:
    """City Campus unless Collegiate Crescent is mentioned."""
    return "Collegiate Crescent" if "collegiate" in _normalise(question) else DEFAULT_CAMPUS


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    """Straight-line distance in metres between two points on Earth."""
    r = 6371000  # Earth's radius in metres
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)))


def build_query(lat: float, lon: float, kinds: set[str]) -> str:
    """Overpass QL: every cafe/restaurant/shop (as asked) within RADIUS_M of the campus."""
    tags = sorted({tag for kind in kinds for tag in KIND_TAGS[kind]})
    parts = "".join(f'nwr["{k}"="{v}"](around:{RADIUS_M},{lat},{lon});' for k, v in tags)
    # "out center tags" gives a centre point for buildings (ways) as well as nodes
    return f"[out:json][timeout:{TIMEOUT_S}];({parts});out center tags;"


def _post_overpass(query: str) -> dict:
    """Send the query to Overpass. Raises requests.RequestException on failure."""
    res = requests.post(OVERPASS_URL, data={"data": query},
                        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    res.raise_for_status()
    return res.json()


def parse_elements(data: dict, lat: float, lon: float, kinds: set[str]) -> list[dict]:
    """Turn Overpass JSON into our place dicts, closest first. Unnamed places are skipped."""
    wanted = {tag for kind in kinds for tag in KIND_TAGS[kind]}
    places, seen = [], set()
    for item in data.get("elements", []):
        tags = item.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        place_type = next((v for k, v in sorted(wanted) if tags.get(k) == v), None)
        if place_type is None:
            continue
        # Nodes have lat/lon; ways and relations have a "center"
        point = item if "lat" in item else item.get("center")
        if not point:
            continue
        key = (item["type"], item["id"])
        if key in seen:
            continue
        seen.add(key)
        places.append({
            "name": name,
            "type": place_type,
            "distance_m": haversine_m(lat, lon, point["lat"], point["lon"]),
            "map_url": f"https://www.openstreetmap.org/{item['type']}/{item['id']}",
        })
    places.sort(key=lambda p: p["distance_m"])
    return places[:MAX_RESULTS]


def fetch_places(campus: str, kinds: set[str]) -> list[dict]:
    lat, lon = CAMPUSES[campus]
    return parse_elements(_post_overpass(build_query(lat, lon, kinds)), lat, lon, kinds)


# ---------- Cache ----------
def _cache_key(campus: str, kinds: set[str]) -> str:
    return campus + "|" + "+".join(sorted(kinds))


def _read_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_cache(key: str, places: list[dict]) -> None:
    cache = _read_cache()
    cache[key] = {"fetched_at": int(time.time()), "places": places}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then swap it in, so two Gunicorn workers can't
    # leave a half-written file
    fd, tmp = tempfile.mkstemp(dir=CACHE_FILE.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CACHE_FILE)


# ---------- The two functions /ask uses ----------
def lookup(question: str) -> list[dict] | None:
    if not is_places_question(question):
        return None

    campus = campus_for(question)
    kinds = kinds_asked(question)
    key = _cache_key(campus, kinds)
    cached = _read_cache().get(key)

    if cached and time.time() - cached["fetched_at"] < CACHE_MAX_AGE_S:
        return cached["places"][:MAX_RESULTS]

    try:
        places = fetch_places(campus, kinds)
    except (requests.RequestException, ValueError, KeyError) as err:
        # Overpass slow or down, or no internet: use an older cache if we have one.
        # (The question itself is never logged.)
        print(f"places.py: Overpass failed ({type(err).__name__}); "
              f"{'using old cache' if cached else 'no cache'}", file=sys.stderr)
        return cached["places"][:MAX_RESULTS] if cached else []

    try:
        _write_cache(key, places)
    except OSError as err:
        print(f"places.py: couldn't write cache ({err})", file=sys.stderr)
    return places


def summary(question: str, places: list[dict]) -> str:
    kinds = kinds_asked(question)
    if kinds == {"cafe"}:
        what = "cafés"
    elif kinds == {"shop"}:
        what = "shops"
    elif "shop" in kinds:
        what = "places to eat and shops"
    else:
        what = "places to eat"
    campus = campus_for(question)
    if not places:
        return f"Sorry, I couldn't load {what} near {campus} right now. Please try again in a minute."
    return f"Here are {what} within about a 10-minute walk of {campus}."


# ---------- Command line ----------
def warm_cache() -> None:
    """Fill the cache for every campus and kind the page's buttons ask for."""
    for campus in CAMPUSES:
        for kinds in ({"food"}, {"cafe"}, {"shop"}):
            try:
                places = fetch_places(campus, kinds)
                _write_cache(_cache_key(campus, kinds), places)
                print(f"OK   {campus:20} {'+'.join(sorted(kinds)):6} {len(places)} places")
            except (requests.RequestException, ValueError, KeyError) as err:
                print(f"FAIL {campus:20} {'+'.join(sorted(kinds)):6} {err}")
            time.sleep(2)   # be polite to the free Overpass server


if __name__ == "__main__":
    if sys.argv[1:] == ["--warm"]:
        warm_cache()
    else:
        q = " ".join(sys.argv[1:]) or "Where can I get lunch near City Campus?"
        result = lookup(q)
        print(json.dumps({"answer": summary(q, result or []), "places": result}, indent=2, ensure_ascii=False))
