#!/usr/bin/env python3
"""
Schedule 13D/13G scraper for a given year using the official SEC EDGAR Daily Index.
- Pulls all SC 13D / SC 13G (and amendments) filed in the specified year (or date range).
- Outputs a CSV of filings with key metadata and links.
- Optional: downloads each accession directory's index.json to locate primary XML/HTML, and
  parses common XML fields (issuer name, CUSIP, percent of class) when possible.

USAGE EXAMPLES
--------------
# Basic: list all Schedule 13 filings for 2025 and save to CSV
python schedule13_scraper.py --year 2025 --out schedule13_2025.csv --verbose

# Also download documents and attempt XML parsing to enrich the CSV
python schedule13_scraper.py --year 2025 --download --parse-xml --out schedule13_2025_enriched.csv --verbose

# Limit to only new filings (exclude amendments)
python schedule13_scraper.py --year 2025 --no-amendments --out schedule13_2025_new.csv

# Narrow to a specific day (much faster to test)
python schedule13_scraper.py --year 2025 --start-date 2025-06-24 --end-date 2025-06-24 --verbose

NOTES
-----
- Respects SEC guidance: sets a real User-Agent and throttles requests (default <= 8/sec).
- Uses the official Daily Index: https://www.sec.gov/Archives/edgar/daily-index/YYYY/QTRN/master.YYYYMMDD.idx
- Master index format is pipe-delimited after a header; fields:
  CIK|Company Name|Form Type|Date Filed|Filename
- For each accession, we also fetch .../index.json to discover components and locate primary docs.

DISCLAIMER
----------
For research/backtesting only. Be polite to SEC servers. Heavy historical runs should be staged and cached locally.
"""

import argparse
import csv
import datetime as dt
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

import requests
from xml.etree import ElementTree as ET

BASE_DAILY = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{q}/master.{ymd}.idx"
ARCHIVES = "https://www.sec.gov/Archives/"
HEADERS = {
    # 👉 Use a real email/domain here. SEC may throttle/deny generic UA strings.
    "User-Agent": "EventDrivenBacktest/1.1 (you@yourdomain.com)",
    "Accept-Encoding": "gzip, deflate",
}

# Throttle to stay under SEC's ~10 req/sec guideline
REQUESTS_PER_SECOND = 8.0
REQUEST_INTERVAL = 1.0 / REQUESTS_PER_SECOND

# Robust detection for Schedule 13 forms (SC or SCHEDULE; D or G; optional /A)
S13_PATTERN = re.compile(r'^\s*(SC|SCHEDULE)\s*13[DG]\s*(?:/\s*A)?\s*$', re.IGNORECASE)


@dataclass
class Filing:
    cik: str
    company_name: str
    form_type: str
    date_filed: str  # YYYY-MM-DD
    filename: str    # path under /Archives/
    accession_url: str  # e.g., https://www.sec.gov/Archives/edgar/data/CIK/ACCESSION/
    filing_detail_url: str  # the .txt filing (full submission)
    primary_doc_url: Optional[str] = None
    primary_doc_type: Optional[str] = None
    issuer_name: Optional[str] = None
    issuer_cusip: Optional[str] = None
    percent_of_class: Optional[str] = None


def is_schedule13(form_type: str) -> bool:
    return bool(S13_PATTERN.match(form_type or ""))


def is_amendment(form_type: str) -> bool:
    # Normalize to catch odd spacing like "/ A"
    return "/A" in (form_type or "").upper().replace(" ", "")


def quarter_from_month(m: int) -> int:
    return (m - 1) // 3 + 1


def parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def daterange(year: int, start_date: Optional[dt.date] = None, end_date: Optional[dt.date] = None):
    start = start_date or dt.date(year, 1, 1)
    end = end_date or dt.date(year, 12, 31)
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def polite_get(url: str, headers: Dict[str, str], max_retries: int = 3, backoff: float = 0.6, verbose: bool = False):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                time.sleep(REQUEST_INTERVAL)
                return resp
            elif resp.status_code in (404, 410):
                # Not found for that day (weekend/holiday) — not an error
                time.sleep(REQUEST_INTERVAL)
                return None
            elif resp.status_code in (429, 503):
                # Too many / service unavailable — exponential backoff
                if verbose:
                    print(f"  HTTP {resp.status_code} on {url}; backing off...", flush=True)
                time.sleep(backoff * (2 ** attempt))
            else:
                if verbose:
                    print(f"  HTTP {resp.status_code} on {url}; retrying...", flush=True)
                time.sleep(backoff * (2 ** attempt))
        except requests.RequestException as e:
            if verbose:
                print(f"  Request error on {url}: {e}; retrying...", flush=True)
            time.sleep(backoff * (2 ** attempt))
    if verbose:
        print(f"  WARNING: GET failed {url} after {max_retries} attempts", flush=True)
    return None


def parse_master_idx(text: str, verbose: bool = False) -> List[Filing]:
    """
    Parses a master.idx content and returns schedule 13 filings.
    Lines look like:
      CIK|Company Name|Form Type|Date Filed|Filename
    """
    filings: List[Filing] = []
    lines = text.splitlines()

    # Find the header delimiter line
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("CIK|Company Name|Form Type|Date Filed|Filename"):
            start_idx = i + 1
            break
    if start_idx is None:
        # Some odd files could be malformed; bail gracefully.
        if verbose:
            print("  No header line found in master.idx (skipping).")
        return filings

    for line in lines[start_idx:]:
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, company_name, form_type, date_filed, filename = parts

        if is_schedule13(form_type):
            filing_detail_url = ARCHIVES + filename.strip()
            acc_dir = "/".join(filename.strip().split("/")[:-1]) + "/"
            accession_url = ARCHIVES + acc_dir
            filings.append(Filing(
                cik=cik.strip(),
                company_name=company_name.strip(),
                form_type=form_type.strip(),
                date_filed=date_filed.strip(),
                filename=filename.strip(),
                accession_url=accession_url,
                filing_detail_url=filing_detail_url
            ))
    return filings


def fetch_index_json(accession_url: str, verbose: bool = False) -> Optional[dict]:
    url = accession_url.rstrip("/") + "/index.json"
    resp = polite_get(url, HEADERS, verbose=verbose)
    if not resp:
        return None
    try:
        return resp.json()
    except Exception:
        if verbose:
            print(f"  Could not parse JSON at {url}")
        return None


def choose_primary_doc(index_json: dict) -> Optional[dict]:
    """
    Try to pick the primary Schedule 13 document (XML preferred). Heuristics:
    - filename contains 'primary' and endswith .xml OR contains 'SCHEDULE_13'
    - otherwise first XML; otherwise first HTML/HTM.
    """
    if not index_json or "directory" not in index_json or "item" not in index_json["directory"]:
        return None
    items = index_json["directory"]["item"]

    def score(it):
        name = it.get("name", "").lower()
        s = 0
        if name.endswith(".xml"):
            s += 3
        if "schedule_13" in name:
            s += 2
        if "primary" in name:
            s += 2
        if name.endswith(".htm") or name.endswith(".html"):
            s += 1
        return s

    items_sorted = sorted(items, key=score, reverse=True)
    return items_sorted[0] if items_sorted else None


def extract_xml_fields(xml_bytes: bytes):
    """
    Try to extract issuer name, CUSIP, and percent of class using loose tag matching.
    Namespaces vary; we'll search by substring on tag names.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None, None, None

    def find_first_text(pred):
        for elem in root.iter():
            tag = elem.tag.lower()
            if pred(tag):
                text = (elem.text or "").strip()
                if text:
                    return text
        return None

    issuer_name = find_first_text(lambda t: "name" in t and ("issuer" in t or "subject" in t))
    cusip = find_first_text(lambda t: "cusip" in t)
    percent = find_first_text(lambda t: ("percent" in t and "class" in t) or "percentofclass" in t)
    return issuer_name, cusip, percent


def scrape_year(year: int,
                include_amendments: bool,
                download_docs: bool,
                parse_xml: bool,
                start_date: Optional[dt.date] = None,
                end_date: Optional[dt.date] = None,
                verbose: bool = False) -> List[Filing]:
    all_filings: List[Filing] = []
    day_idx = 0
    for d in daterange(year, start_date, end_date):
        day_idx += 1
        q = quarter_from_month(d.month)
        ymd = d.strftime("%Y%m%d")
        url = BASE_DAILY.format(year=year, q=q, ymd=ymd)
        if verbose:
            print(f"[{d}] GET {url}", flush=True)

        resp = polite_get(url, HEADERS, verbose=verbose)
        if not resp:
            if verbose:
                print("  (no index for this date; weekend/holiday or not yet available)")
            continue

        # Quick peek so you can confirm we got a valid file
        if verbose:
            snippet = resp.text[:120].replace("\n", " \\n ")
            print(f"  first 120 chars: {snippet}")

        filings = parse_master_idx(resp.text, verbose=verbose)

        if not include_amendments:
            filings = [f for f in filings if not is_amendment(f.form_type)]

        if verbose:
            print(f"  found {len(filings)} schedule 13 filings")

        # Enrich with primary doc and optional parsing
        if filings and (download_docs or parse_xml):
            for f in filings:
                idx = fetch_index_json(f.accession_url, verbose=verbose)
                if idx:
                    primary = choose_primary_doc(idx)
                    if primary:
                        f.primary_doc_url = f.accession_url + primary["name"]
                        f.primary_doc_type = primary.get("type")
                        if parse_xml and f.primary_doc_url.lower().endswith(".xml"):
                            doc_resp = polite_get(f.primary_doc_url, HEADERS, verbose=verbose)
                            if doc_resp:
                                issuer, cusip, pct = extract_xml_fields(doc_resp.content)
                                f.issuer_name = issuer
                                f.issuer_cusip = cusip
                                f.percent_of_class = pct

        if filings:
            all_filings.extend(filings)

    return all_filings


def write_csv(path: str, filings: List[Filing]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(asdict(filings[0]).keys()) if filings else [
        "cik","company_name","form_type","date_filed","filename","accession_url",
        "filing_detail_url","primary_doc_url","primary_doc_type","issuer_name",
        "issuer_cusip","percent_of_class"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in filings:
            w.writerow(asdict(rec))


def main():
    global REQUESTS_PER_SECOND, REQUEST_INTERVAL

    parser = argparse.ArgumentParser(description="Fetch Schedule 13D/13G filings from SEC EDGAR Daily Index.")
    parser.add_argument("--year", type=int, default=2025, help="Year to fetch (default: 2025)")
    parser.add_argument("--start-date", type=str, help="Restrict range start (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Restrict range end (YYYY-MM-DD)")
    parser.add_argument("--out", type=str, default="schedule13_filings.csv", help="CSV output path")
    parser.add_argument("--download", action="store_true", help="Fetch accession index.json and locate primary doc")
    parser.add_argument("--parse-xml", action="store_true", help="If XML primary doc found, parse issuer & percent")
    parser.add_argument("--no-amendments", action="store_true", help="Exclude amendments (SC/SCHEDULE 13D/G /A)")
    parser.add_argument("--rps", type=float, default=REQUESTS_PER_SECOND, help="Requests per second throttle (default 8)")
    parser.add_argument("--user-agent", type=str, default=HEADERS["User-Agent"], help="Custom User-Agent per SEC guidance")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    REQUESTS_PER_SECOND = max(1.0, float(args.rps))
    REQUEST_INTERVAL = 1.0 / REQUESTS_PER_SECOND
    HEADERS["User-Agent"] = args.user_agent

    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None

    filings = scrape_year(
        year=args.year,
        include_amendments=not args.no_amendments,
        download_docs=args.download,
        parse_xml=args.parse_xml,
        start_date=start_date,
        end_date=end_date,
        verbose=args.verbose
    )
    write_csv(args.out, filings)
    print(f"Wrote {len(filings)} filings to {args.out}")


if __name__ == "__main__":
    main()
