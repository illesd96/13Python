import pandas as pd

from pathlib import Path

BASE = Path("/Users/szaszvince/Documents/Python/sec_13dg_txt_2025")

IN_PATH = BASE / "scoring_table_with_direction.csv"
OUT_PATH = BASE / "scoring_model.csv"

def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()

def to_float(x):
    if pd.isna(x):
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None

def to_int(x, default=None):
    if pd.isna(x):
        return default
    try:
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return default

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))

# ---------- Scoring functions ----------
def score_form_type(conformed_submission_type: str) -> int:
    ft = norm_str(conformed_submission_type)

    # Typical values: "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"
    if ft == "SC 13D":
        return 18
    if ft == "SC 13D/A":
        return 12
    if ft == "SC 13G":
        return 6
    if ft == "SC 13G/A":
        return 2

    # tolerate some variants
    if "13D" in ft and "/A" not in ft:
        return 18
    if "13D" in ft and "/A" in ft:
        return 12
    if "13G" in ft and "/A" not in ft:
        return 6
    if "13G" in ft and "/A" in ft:
        return 2

    return 0

def score_percent_owned(percent_of_class) -> int:
    p = to_float(percent_of_class)
    if p is None:
        return 0
    if p >= 15.0:
        return 14
    if p >= 10.0:
        return 12
    if p >= 7.5:
        return 9
    if p >= 5.0:
        return 6
    return 0

def score_reporting_person_type(type_of_reporting_person: str) -> int:
    t = norm_str(type_of_reporting_person)

    # handle multi codes like "IA,PF" or "IA; PF"
    codes = {c.strip() for c in t.replace(";", ",").split(",") if c.strip()}

    if ("IA" in codes) or ("PF" in codes):
        return 12
    if "IN" in codes:
        return 10
    if ("HC" in codes) or ("CO" in codes):
        return 8
    if "PN" in codes:
        return 6
    if ("BK" in codes) or ("IC" in codes):
        return 4
    if ("OO" in codes) or (len(codes) == 0):
        return 2

    # unknown but present
    return 2

def score_source_of_funds(source_of_funds_code: str) -> int:
    s = norm_str(source_of_funds_code)
    if s == "WC":
        return 6
    if s == "PF":
        return 5
    if s == "BK":
        return 4
    if s == "CM":
        return 3
    if s == "AF":
        return 2
    if s == "OO":
        return 1
    if s == "":
        return 0
    return 0

def score_activist_flag(item4_is_activist) -> int:
    v = to_int(item4_is_activist, default=0)
    return 18 if v == 1 else 0

def score_amendment_number(amendment_number) -> int:
    n = to_int(amendment_number, default=None)
    if n is None:
        return 0
    if n == 0:
        return 4
    if n in (1, 2):
        return 2
    if n >= 3:
        return 1
    return 0

def score_amendment_direction(amendment_direction: str) -> int:
    d = norm_str(amendment_direction)
    if d == "INCREASE":
        return 6
    if d == "DECREASE":
        return -6
    if d == "FLAT":
        return 0
    return 0

def score_purpose_strength(item4_severity) -> int:
    # expected 0/1/2
    v = to_int(item4_severity, default=None)
    if v is None:
        return 0
    if v >= 2:
        return 16
    if v == 1:
        return 10
    if v == 0:
        return 4
    return 0

def score_item6_contracts(item6_severity) -> int:
    # if item6_severity > 0 => contracts/arrangements present
    v = to_int(item6_severity, default=0)
    return 6 if v > 0 else 0

def grade(score_total: int) -> str:
    if score_total >= 80:
        return "A"
    if score_total >= 60:
        return "B"
    if score_total >= 40:
        return "C"
    return "D"

def main():
    df = pd.read_csv(IN_PATH)

    # Components
    df["score_form_type"] = df["conformed_submission_type"].apply(score_form_type)
    df["score_percent_of_class"] = df["percent_of_class"].apply(score_percent_owned)
    df["score_rpt_type"] = df["type_of_reporting_person"].apply(score_reporting_person_type)
    df["score_source_funds"] = df["source_of_funds_code"].apply(score_source_of_funds)
    df["score_activist"] = df["item4_is_activist"].apply(score_activist_flag)
    df["score_amend_no"] = df["amendment_number"].apply(score_amendment_number)
    df["score_purpose"] = df["item4_severity"].apply(score_purpose_strength)
    df["score_item6"] = df["item6_severity"].apply(score_item6_contracts)

    df["score_direction_adj"] = df["amendment_direction"].apply(score_amendment_direction)

    base_cols = [
        "score_form_type",
        "score_percent_of_class",
        "score_rpt_type",
        "score_source_funds",
        "score_activist",
        "score_amend_no",
        "score_purpose",
        "score_item6",
    ]

    df["score_base"] = df[base_cols].sum(axis=1)
    df["score_total_raw"] = df["score_base"] + df["score_direction_adj"]
    df["score_total"] = df["score_total_raw"].apply(lambda x: clamp(int(round(x)), 0, 100))
    df["grade"] = df["score_total"].apply(grade)

    # Output
    out_cols = [
        "accession_number",
        "filed_as_of_date",
        "date_of_event",
        "conformed_submission_type",
        "issuer_cik",
        "issuer_name",
        "issuer_cusip",
        "reporting_person_cik",
        "reporting_person_name",
        "score_total",
        "grade",
        "score_base",
        "score_direction_adj",
        *base_cols,
    ]

    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(OUT_PATH, index=False)
    print(f"Wrote: {OUT_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    main()
