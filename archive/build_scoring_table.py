import json
from pathlib import Path

BASE = Path("/Users/szaszvince/Documents/Python/sec_13dg_txt_2025")
FILINGS = BASE / "filings.jsonl"
RPS = BASE / "reporting_persons.jsonl"
OUT = BASE / "scoring_table.csv"

def to_float(x):
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except:
        return None

def to_int(x):
    f = to_float(x)
    if f is None:
        return None
    # shares mezők sokszor .00-val jönnek
    return int(round(f))

def to_int_or_zero(x):
    if x is None or str(x).strip() == "":
        return 0
    try:
        return int(float(str(x).strip()))
    except:
        return 0

# 1) Load filings keyed by accession
filings = {}
with FILINGS.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        acc = row.get("accession_number")
        if acc:
            filings[acc] = row

# 2) Merge reporting persons with filing row
rows = []
with RPS.open("r", encoding="utf-8") as f:
    for line in f:
        rp = json.loads(line)
        acc = rp.get("accession_number")
        fil = filings.get(acc, {})
        merged = {
            "accession_number": acc,
            "filed_as_of_date": fil.get("filed_as_of_date"),
            "conformed_submission_type": fil.get("conformed_submission_type") or fil.get("submission_type"),
            "date_of_event": fil.get("date_of_event"),
            "amendment_number": to_int_or_zero(fil.get("amendment_number")),
            "is_amendment": 1 if to_int_or_zero(fil.get("amendment_number")) > 0 else 0,

            "issuer_cik": fil.get("issuer_cik") or fil.get("subject_company_cik"),
            "issuer_name": fil.get("issuer_name") or fil.get("subject_company_name"),
            "issuer_cusip": fil.get("issuer_cusip"),

            "reporting_person_cik": rp.get("reporting_person_cik") or fil.get("filed_by_cik"),
            "reporting_person_name": rp.get("reporting_person_name") or fil.get("filed_by_name"),
            "type_of_reporting_person": "|".join(rp.get("type_of_reporting_person") or []),

            "percent_of_class": to_float(rp.get("percent_of_class")),
            "amount_beneficially_owned": to_int(rp.get("aggregate_amount_owned")),
            "sole_voting_power": to_int(rp.get("sole_voting_power")),
            "shared_voting_power": to_int(rp.get("shared_voting_power")),
            "sole_dispositive_power": to_int(rp.get("sole_dispositive_power")),
            "shared_dispositive_power": to_int(rp.get("shared_dispositive_power")),

            "source_of_funds_code": (fil.get("source_of_funds_code") or "").strip(),
            "item4_purpose_of_transaction": (fil.get("item4_purpose_of_transaction") or "").strip(),
            "item6_contracts_arrangements": (fil.get("item6_contracts_arrangements") or "").strip(),
            "amendment_number": to_int_or_zero(fil.get("amendment_number")),
            "is_amendment": 1 if to_int_or_zero(fil.get("amendment_number")) > 0 else 0,

        }
        rows.append(merged)

# 3) If a filing had zero RP rows, optionally keep a “filing-only” row
# (comment out if you don't want these)
rp_accessions = set(r.get("accession_number") for r in rows if r.get("accession_number"))
for acc, fil in filings.items():
    if acc in rp_accessions:
        continue
    rows.append({
        "accession_number": acc,
        "filed_as_of_date": fil.get("filed_as_of_date"),
        "conformed_submission_type": fil.get("conformed_submission_type") or fil.get("submission_type"),
        "date_of_event": fil.get("date_of_event"),
        "issuer_cik": fil.get("issuer_cik") or fil.get("subject_company_cik"),
        "issuer_name": fil.get("issuer_name") or fil.get("subject_company_name"),
        "issuer_cusip": fil.get("issuer_cusip"),
        "reporting_person_cik": fil.get("filed_by_cik"),
        "reporting_person_name": fil.get("filed_by_name"),
        "type_of_reporting_person": "",
        "percent_of_class": None,
        "amount_beneficially_owned": None,
        "sole_voting_power": None,
        "shared_voting_power": None,
        "sole_dispositive_power": None,
        "shared_dispositive_power": None,
        "source_of_funds_code": (fil.get("source_of_funds_code") or "").strip(),
        "item4_purpose_of_transaction": (fil.get("item4_purpose_of_transaction") or "").strip(),
        "item6_contracts_arrangements": (fil.get("item6_contracts_arrangements") or "").strip(),
    })

# 4) Write CSV
cols = [
    "accession_number","filed_as_of_date","conformed_submission_type","date_of_event",
    "is_amendment","amendment_number",
    "issuer_cik","issuer_name","issuer_cusip",
    "reporting_person_cik","reporting_person_name","type_of_reporting_person",
    "percent_of_class","amount_beneficially_owned",
    "sole_voting_power","shared_voting_power","sole_dispositive_power","shared_dispositive_power",
    "source_of_funds_code","item4_purpose_of_transaction","item6_contracts_arrangements",

]

with OUT.open("w", encoding="utf-8") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        def esc(v):
            if v is None:
                return ""
            s = str(v).replace('"', '""')
            # quote if needed
            if any(c in s for c in [",", "\n", "\r"]):
                return f'"{s}"'
            return s
        f.write(",".join(esc(r.get(c, "")) for c in cols) + "\n")

print("Wrote:", OUT)
print("Rows:", len(rows))
