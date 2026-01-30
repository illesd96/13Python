import pandas as pd
import re
from pathlib import Path

BASE = Path("/Users/szaszvince/Documents/Python/sec_13dg_txt_2025")
IN_FILE = BASE / "scoring_table.csv"
OUT_FILE = BASE / "scoring_table_with_direction.csv"

EPS = 0.01  # threshold in percentage points

print("Loading:", IN_FILE)
df = pd.read_csv(IN_FILE)

def form_family(ft):
    ft = (ft or "")
    return "13G" if "13G" in str(ft).upper() else "13D"

df["form_family"] = df["conformed_submission_type"].apply(form_family)

def norm_name(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    # normalize whitespace + remove punctuation-ish noise
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-&]", "", s)
    return s or None

# --- NEW: stable per-reporting-person key ---
df["reporting_person_name_norm"] = df.get("reporting_person_name", pd.Series([None]*len(df))).apply(norm_name)

def make_rp_key(row):
    cik = row.get("reporting_person_cik")
    name = row.get("reporting_person_name_norm")
    cik_str = None if pd.isna(cik) else str(int(cik)) if str(cik).endswith(".0") else str(cik)
    if cik_str and name:
        return f"{cik_str}::{name}"
    if cik_str:
        return f"{cik_str}::"
    if name:
        return f"::${name}"
    return None

df["reporting_person_key"] = df.apply(make_rp_key, axis=1)

# --- Sorting: prefer time fields over amendment_number ---
# If your scoring_table has these columns, use them; otherwise they’ll be ignored.
sort_cols = ["issuer_cik", "form_family", "reporting_person_key"]
if "filed_as_of_date" in df.columns:
    sort_cols.append("filed_as_of_date")
elif "acceptance_datetime" in df.columns:
    sort_cols.append("acceptance_datetime")
else:
    # fallback to amendment_number then accession_number
    if "amendment_number" in df.columns:
        sort_cols.append("amendment_number")
    if "accession_number" in df.columns:
        sort_cols.append("accession_number")

df = df.sort_values(by=sort_cols, kind="mergesort")  # stable sort

group_cols = ["issuer_cik", "form_family", "reporting_person_key"]

# Previous percent
df["prev_percent_of_class"] = (
    df.groupby(group_cols)["percent_of_class"]
      .shift(1)
)

# Delta
df["delta_percent_of_class"] = df["percent_of_class"] - df["prev_percent_of_class"]

def amendment_direction(row):
    if pd.isna(row["prev_percent_of_class"]):
        return "first"
    if row["delta_percent_of_class"] > EPS:
        return "increase"
    if row["delta_percent_of_class"] < -EPS:
        return "decrease"
    return "flat"

df["amendment_direction"] = df.apply(amendment_direction, axis=1)

DROP_COLS = [
    "form_family",
    "reporting_person_name_norm",
    "reporting_person_key",
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

df.to_csv(OUT_FILE, index=False)

print("Wrote:", OUT_FILE)
print("Rows:", len(df))
print(df["amendment_direction"].value_counts(dropna=False))
