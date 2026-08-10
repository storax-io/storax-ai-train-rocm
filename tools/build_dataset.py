#!/usr/bin/env python3
"""Build data/finnish_presidents.json by traversing Wikipedia:
List_of_presidents_of_Finland -> each president's article -> infobox facts.

Runs in WSL with stdlib only. Fields that fail to parse fall back to the
FALLBACK table (flagged in the output so we can see parser coverage).
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "storax-train-test/0.1 (research; contact: local)"}

PRESIDENTS = [  # article titles in ordinal order (from the list page)
    "Kaarlo Juho Ståhlberg", "Lauri Kristian Relander",
    "Pehr Evind Svinhufvud", "Kyösti Kallio", "Risto Ryti",
    "Carl Gustaf Emil Mannerheim", "Juho Kusti Paasikivi",
    "Urho Kekkonen", "Mauno Koivisto", "Martti Ahtisaari",
    "Tarja Halonen", "Sauli Niinistö", "Alexander Stubb",
]

# Verified against the list page table; used only when page parsing misses.
FALLBACK = {
    "Carl Gustaf Emil Mannerheim": {"term_start": 1944, "term_end": 1946,
                                    "party": "independent"},
}

# Infobox party fields are messy (multi-party histories, {{Plainlist}}
# markup). Canonical party-at-election, verified against the list page.
PARTY_CANON = {
    "Kaarlo Juho Ståhlberg": "National Progressive Party",
    "Lauri Kristian Relander": "Agrarian League",
    "Pehr Evind Svinhufvud": "National Coalition Party",
    "Kyösti Kallio": "Agrarian League",
    "Risto Ryti": "National Progressive Party",
    "Carl Gustaf Emil Mannerheim": "independent",
    "Juho Kusti Paasikivi": "National Coalition Party",
    "Urho Kekkonen": "Agrarian League (renamed Centre Party in 1965)",
    "Mauno Koivisto": "Social Democratic Party",
    "Martti Ahtisaari": "Social Democratic Party",
    "Tarja Halonen": "Social Democratic Party",
    "Sauli Niinistö": "National Coalition Party",
    "Alexander Stubb": "National Coalition Party",
}
INCUMBENT = "Alexander Stubb"  # no term_end / successor

# Every party referenced by PARTY_CANON, with its Wikipedia article title.
# "Agrarian League (Finland)" is a redirect: the party was renamed Centre
# Party in 1965 and Wikipedia keeps one article — merged as an alias.
PARTY_PAGES = {
    "National Progressive Party": "National Progressive Party (Finland)",
    "Agrarian League": "Centre Party (Finland)",
    "Centre Party": "Centre Party (Finland)",
    "National Coalition Party": "National Coalition Party",
    "Social Democratic Party": "Social Democratic Party of Finland",
}


def api_get(params):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"still rate-limited: {params}")


def fetch_wikitext(title):
    return api_get({"action": "parse", "page": title, "prop": "wikitext",
                    "format": "json", "formatversion": "2"})["parse"]["wikitext"]


def fetch_plaintext(title):
    """Full article as plain text (TextExtracts; no markup, no refs)."""
    pages = api_get({"action": "query", "prop": "extracts", "titles": title,
                     "explaintext": "1", "format": "json",
                     "formatversion": "2"})["query"]["pages"]
    text = pages[0].get("extract", "")
    # Drop trailing non-content sections.
    for sec in ("== See also ==", "== References ==", "== External links ==",
                "== Bibliography ==", "== Further reading ==", "== Notes =="):
        idx = text.find(sec)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def infobox_field(wt, *names):
    for name in names:
        m = re.search(rf"^\s*\|\s*{name}\s*=\s*(.+)$", wt, re.M)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


def strip_wiki(s):
    if not s:
        return None
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\{\{[^}]*\}\}", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip(" ,;") or None


def year_of(s):
    if not s:
        return None
    m = re.search(r"(1[89]\d\d|20\d\d)", s)
    return int(m.group(1)) if m else None


def main():
    out = []
    for i, title in enumerate(PRESIDENTS, start=1):
        wt = fetch_wikitext(title)
        fb = FALLBACK.get(title, {})
        used_fallback = []

        # Presidency dates: first term_start/term_end pair in the infobox
        # belongs to the highest office listed (the presidency).
        term_start = year_of(infobox_field(wt, "term_start", "term_start1"))
        term_end = year_of(infobox_field(wt, "term_end", "term_end1"))
        birth = year_of(infobox_field(wt, "birth_date"))
        death = year_of(infobox_field(wt, "death_date"))
        birth_place = strip_wiki(infobox_field(wt, "birth_place"))
        party = strip_wiki(infobox_field(wt, "party", "otherparty"))
        successor = strip_wiki(infobox_field(wt, "successor", "successor1"))

        if title == INCUMBENT:
            term_end = None
            successor = None
        rec = {"ordinal": i, "name": title,
               "surname": title.split()[-1],
               "term_start": term_start, "term_end": term_end,
               "birth_year": birth, "death_year": death,
               "birth_place": birth_place,
               "birth_town": (birth_place or "").split(",")[0].strip() or None,
               "party": PARTY_CANON[title],
               "party_raw": party,
               "successor": successor,
               "article": fetch_plaintext(title)}
        for k, v in fb.items():
            if rec.get(k) in (None, ""):
                rec[k] = v
                used_fallback.append(k)
        rec["fallback_fields"] = used_fallback
        out.append(rec)
        print(f"{i:2d}. {title}: term {rec['term_start']}-{rec['term_end']}, "
              f"born {rec['birth_year']} {rec['birth_place']}, "
              f"party {rec['party']}, succ {rec['successor']}"
              + (f"  [fallback: {used_fallback}]" if used_fallback else ""))
        time.sleep(3)

    parties = []
    by_page = {}
    for short, page in PARTY_PAGES.items():
        if page in by_page:
            by_page[page]["aliases"].append(short)
            continue
        art = fetch_plaintext(page)
        rec = {"name": short, "page": page, "aliases": [], "article": art}
        by_page[page] = rec
        parties.append(rec)
        print(f" party {short}: {len(art):,} chars")
        time.sleep(3)

    dest = Path(__file__).resolve().parent.parent / "data" / "finnish_presidents.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps({"presidents": out, "parties": parties},
                               ensure_ascii=False, indent=2))
    print(f"\nwrote {dest} ({len(out)} presidents, {len(parties)} parties)")


if __name__ == "__main__":
    main()
