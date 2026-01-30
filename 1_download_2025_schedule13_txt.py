"""
Download all 2025 Schedule 13D/13G filings as complete submission .txt files from SEC EDGAR.

Approach:
- Use EDGAR quarterly master indexes: /Archives/edgar/full-index/2025/QTR{1..4}/master.idx
- Filter by form type: SC 13D, SC 13D/A, SC 13G, SC 13G/A (plus a couple of variants)
- Each master.idx row contains a "Filename" like: edgar/data/928785/000092878526000001/primary_doc.xml
- Convert that to the complete submission text file URL:
    https://www.sec.gov/Archives/edgar/data/928785/000092878526000001/0000928785-26-000001.txt

Notes:
- SEC fair access: keep request rate low; set a descriptive User-Agent.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests


# -----------------------
# Config
# -----------------------

YEAR = 2025
QTRS = [1,2,3,4]

# Output folder for downloaded .txt filings
OUT_DIR = Path("./sec_13dg_txt_2025").resolve()

# Be conservative; SEC guidance is 10 req/s max, but better to run slower.
REQUESTS_PER_SECOND = 2.0
SLEEP_SECONDS = 1.0 / REQUESTS_PER_SECOND

# You MUST customize this (SEC expects identifying UA with contact info)
USER_AGENT = "YourName YourOrg your.email@example.com"

# Forms to match (some filings use SC prefix, some show SCHEDULE text in some contexts)
FORM_WHITELIST = {
    "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
    "SCHEDULE 13D", "SCHEDULE 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A",
}

MASTER_INDEX_URL_TMPL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/master.idx"


@dataclass
class FilingRow:
    cik: str
    company_name: str
    form_type: str
    date_filed: str
    filename: str  # path in Archives (e.g., edgar/data/.../primary_doc.xml)


def sec_get(session: requests.Session, url: str) -> requests.Response:
    """GET with SEC-friendly headers and basic retry."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    # minimal retry
    for attempt in range(4):
        resp = session.get(url, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp
        if resp.status_code in (403, 429, 500, 502, 503, 504):
            time.sleep(2 + attempt * 2)
            continue
        resp.raise_for_status()
    resp.raise_for_status()
    return resp  # not reached


def parse_master_idx(text: str) -> List[FilingRow]:
    """
    master.idx format: header lines then a pipe-delimited table:
    CIK|Company Name|Form Type|Date Filed|Filename
    """
    lines = text.splitlines()
    # Find the table header line
    start_i = None
    for i, line in enumerate(lines):
        if line.strip().startswith("CIK|Company Name|Form Type|Date Filed|Filename"):
            start_i = i + 1
            break
    if start_i is None:
        raise ValueError("Could not locate master.idx table header")

    rows: List[FilingRow] = []
    reader = csv.reader(lines[start_i:], delimiter="|")
    for parts in reader:
        if len(parts) != 5:
            continue
        cik, cname, ftype, date_filed, filename = [p.strip() for p in parts]
        rows.append(FilingRow(cik=cik, company_name=cname, form_type=ftype, date_filed=date_filed, filename=filename))
    return rows


def is_schedule13(form_type: str) -> bool:
    ft = form_type.strip().upper()
    return ft in FORM_WHITELIST


_ACCESSION_RE = re.compile(r"/(\d{10}-\d{2}-\d{6})\.(txt|html|htm|xml)$", re.IGNORECASE)

def build_complete_txt_url_from_filename(filename: str) -> Optional[str]:
    """
    Robust conversion:
    - If master.idx filename already ends with .txt -> use it directly under https://www.sec.gov/Archives/
    - Else if it points to a filing folder (.../##########YY######/something.*) -> construct accession.txt
    """
    fn = filename.strip().lstrip("/")

    # Case 1: master.idx already points to the complete submission text file
    if fn.lower().endswith(".txt"):
        return f"https://www.sec.gov/Archives/{fn}"

    # Case 2: master.idx points to some doc inside a filing folder (e.g., primary_doc.xml)
    parts = fn.split("/")
    if len(parts) < 4:
        return None

    dir_path = "/".join(parts[:-1])   # edgar/data/.../000092878526000001
    accession_nodash = parts[-2]      # 18-digit folder

    if not (accession_nodash.isdigit() and len(accession_nodash) == 18):
        return None

    accession = f"{accession_nodash[0:10]}-{accession_nodash[10:12]}-{accession_nodash[12:18]}"
    return f"https://www.sec.gov/Archives/{dir_path}/{accession}.txt"


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120]


def download_one_txt(session: requests.Session, row: FilingRow) -> Tuple[bool, str]:
    """
    Download a single .txt and save to:
      OUT_DIR/{form}/{date_filed}/{accession}.txt
    """
    url = build_complete_txt_url_from_filename(row.filename)
    if not url:
        return False, f"Could not build txt URL from filename: {row.filename}"

    # accession from URL
    accession = url.rsplit("/", 1)[-1].replace(".txt", "")

    form_folder = safe_filename(row.form_type.upper().replace("/", "_").replace(" ", "_"))
    out_folder = OUT_DIR / form_folder / row.date_filed
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / f"{accession}.txt"

    if out_path.exists() and out_path.stat().st_size > 0:
        return True, f"SKIP exists {out_path.name}"

    resp = sec_get(session, url)
    out_path.write_bytes(resp.content)
    return True, f"OK {out_path.name}"


def main():
    if "@" not in USER_AGENT:
        raise SystemExit("Please set USER_AGENT to include your contact email before running.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # 1) Gather all matching rows from 2025 quarterly master indexes
    all_rows: List[FilingRow] = []

    print("RUN CONFIG:", "YEAR=", YEAR, "QTRS=", QTRS, "DATE_FILTER=2025-01")

    for qtr in QTRS:
        idx_url = MASTER_INDEX_URL_TMPL.format(year=YEAR, qtr=qtr)
        print(f"Fetching master index: {idx_url}")
        r = sec_get(session, idx_url)
        time.sleep(SLEEP_SECONDS)

        rows = parse_master_idx(r.text)
        sched = [
            row for row in rows
            if is_schedule13(row.form_type)
        ]
        print(f"  Rows total: {len(rows):,} | Schedule13: {len(sched):,}")
        all_rows.extend(sched)

    print(f"\nTotal Schedule 13 rows in {YEAR}: {len(all_rows):,}")
    print("\nDOWNLOAD DEBUG (first 3 rows):")
    for j, row in enumerate(all_rows[:3], start=1):
        url = build_complete_txt_url_from_filename(row.filename)
        print(f"  #{j} form={row.form_type} date={row.date_filed} filename={row.filename}")
        print(f"     txt_url={url}")
    print("\nStarting downloads...\n")

    # 2) Download .txt files
    ok = 0
    err = 0
    for i, row in enumerate(all_rows, start=1):
        try:
            success, msg = download_one_txt(session, row)
            if success:
                ok += 1
            else:
                err += 1
            if i % 200 == 0:
                print(f"[{i}/{len(all_rows)}] ok={ok} err={err} last={msg}")
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            err += 1
            print(f"[{i}/{len(all_rows)}] ERROR downloading {row.filename} -> {e}")
            time.sleep(SLEEP_SECONDS)

    print(f"\nDone. Downloaded/exists OK={ok}, ERR={err}")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
