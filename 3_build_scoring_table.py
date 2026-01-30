import json
import re
from pathlib import Path

BASE = Path("/Users/szaszvince/Documents/Python/sec_13dg_txt_2025")
FILINGS = BASE / "filings.jsonl"
RPS = BASE / "reporting_persons.jsonl"
OUT = BASE / "scoring_table.csv"

# -------------------------
# Converters
# -------------------------

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
    return int(round(f))

def to_int_or_zero(x):
    if x is None or str(x).strip() == "":
        return 0
    try:
        return int(float(str(x).strip()))
    except:
        return 0

# -------------------------
# Item 4 / Item 6 interpreter (embedded)
# -------------------------

def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()

def _split_sentences(text: str):
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", text) if s.strip()]

def _compile_patterns(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]

def _find_evidence(sentences, compiled_patterns, limit=8):
    hits = []
    for s in sentences:
        for p in compiled_patterns:
            if p.search(s):
                hits.append(s)
                break
    # dedupe preserve order + cap
    return list(dict.fromkeys(hits))[:limit]

ITEM4_RULES = {
    "activist": [
        r"\bproxy\b", r"\bproxy contest\b", r"\bconsent solicitation\b", r"\bsolicit\b",
        r"\bboard\b", r"\bdirector\b", r"\bboard seat\b", r"\bnominate\b"
    ],
    "mna": [
        r"\bmerger\b", r"\bacquisition\b", r"\btender offer\b", r"\bgo private\b",
        r"\bsale of\b", r"\btransaction\b", r"\bstrategic alternatives?\b", r"\bstrategic review\b"
    ],
    "capital_return": [
        r"\bbuyback\b", r"\brepurchase\b", r"\bdividend\b", r"\breturn of capital\b"
    ],
    "passive": [
        r"\bfor investment purposes only\b",
        r"\bno present plans\b",
        r"\bnot presently\b.*\bplans\b",
        r"\bdoes not have\b.*\bplans\b",
        r"\bno current intention\b",
        r"\bnot currently\b.*\bplans\b",
    ],
}

ITEM4_ACTIONS = {
    "discuss_with_management": [
        r"\bdiscuss\b", r"\bengage\b.*\bmanagement\b", r"\bmeet\b.*\bmanagement\b",
        r"\bcommunicat(e|ed|ing)\b.*\bmanagement\b"
    ],
    "seek_board_seats": [
        r"\bseek\b.*\bboard\b", r"\bnominate\b", r"\bboard seat\b", r"\brepresentation\b.*\bboard\b"
    ],
    "proxy_contest": [
        r"\bproxy contest\b", r"\bsolicit\b.*\bproxy\b", r"\bconsent solicitation\b"
    ],
    "propose_transaction": [
        r"\bpropose\b.*\btransaction\b", r"\bpropose\b.*\bmerger\b", r"\bpropose\b.*\bacquisition\b",
        r"\bseek\b.*\bmerger\b", r"\bseek\b.*\bacquisition\b"
    ],
}

ITEM6_INSTRUMENTS = {
    "swap": [r"\btotal return swap\b", r"\bswap\b", r"\btrs\b"],
    "option": [r"\boption\b", r"\bcall option\b", r"\bput option\b"],
    "forward": [r"\bforward\b"],
    "convertible": [r"\bconvertible\b", r"\bdebenture\b", r"\bconvertible note\b"],
    "margin_loan": [r"\bmargin\b", r"\bpledge\b", r"\bcredit agreement\b", r"\bloan\b", r"\bsecurity interest\b"],
    "voting_agreement": [r"\bvoting agreement\b", r"\birrevocable proxy\b", r"\bproxy\b"],
    "standstill": [r"\bstandstill\b"],
    "lockup": [r"\block[- ]up\b"],
}

def interpret_item4(item4_text: str):
    text_n = _normalize(item4_text)
    sents = _split_sentences(text_n)
    lower = text_n.lower()

    flags = []
    actions = []
    evidence = []

    for flag, pats in ITEM4_RULES.items():
        cp = _compile_patterns(pats)
        if any(p.search(lower) for p in cp):
            flags.append(flag)
            evidence += _find_evidence(sents, cp)

    for act, pats in ITEM4_ACTIONS.items():
        cp = _compile_patterns(pats)
        if any(p.search(lower) for p in cp):
            actions.append(act)
            evidence += _find_evidence(sents, cp)

    # severity 0–3
    severity = 0
    if "passive" in flags and len(flags) == 1 and not actions:
        severity = 0
    else:
        if flags or actions:
            severity = 1
        if any(f in flags for f in ["activist", "mna", "capital_return"]):
            severity = max(severity, 2)
        if any(a in actions for a in ["proxy_contest", "seek_board_seats", "propose_transaction"]):
            severity = 3

    evidence = list(dict.fromkeys(evidence))[:8]

    return {
        "item4_severity": severity,
        "item4_is_passive": 1 if "passive" in flags else 0,
        "item4_is_activist": 1 if "activist" in flags else 0,
        "item4_has_mna": 1 if "mna" in flags else 0,
        "item4_has_capital_return": 1 if "capital_return" in flags else 0,
        "item4_actions": "|".join(actions),
        "item4_evidence": " || ".join(evidence),
    }

def interpret_item6(item6_text: str):
    text_n = _normalize(item6_text)
    sents = _split_sentences(text_n)
    lower = text_n.lower()

    instruments = []
    evidence = []

    for inst, pats in ITEM6_INSTRUMENTS.items():
        cp = _compile_patterns(pats)
        if any(p.search(lower) for p in cp):
            instruments.append(inst)
            evidence += _find_evidence(sents, cp)

    has_derivatives = 1 if any(x in instruments for x in ["swap", "option", "forward"]) else 0

    # severity 0–3
    severity = 0
    if instruments:
        severity = 1
        if has_derivatives or ("margin_loan" in instruments):
            severity = max(severity, 2)
        if ("voting_agreement" in instruments) or ("standstill" in instruments):
            severity = 3

    evidence = list(dict.fromkeys(evidence))[:8]

    return {
        "item6_severity": severity,
        "item6_has_derivatives": has_derivatives,
        "item6_has_voting_agreement": 1 if "voting_agreement" in instruments else 0,
        "item6_has_standstill": 1 if "standstill" in instruments else 0,
        "item6_has_margin_loan": 1 if "margin_loan" in instruments else 0,
        "item6_instruments": "|".join(sorted(set(instruments))),
        "item6_evidence": " || ".join(evidence),
    }

# -------------------------
# 1) Load filings keyed by accession
# -------------------------

filings = {}
with FILINGS.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        acc = row.get("accession_number")
        if acc:
            filings[acc] = row

# -------------------------
# 2) Merge reporting persons with filing row
# -------------------------

rows = []
with RPS.open("r", encoding="utf-8") as f:
    for line in f:
        rp = json.loads(line)
        acc = rp.get("accession_number")
        fil = filings.get(acc, {})

        item4_text = (fil.get("item4_purpose_of_transaction") or "").strip()
        item6_text = (fil.get("item6_contracts_arrangements") or "").strip()

        item4_feat = interpret_item4(item4_text)
        item6_feat = interpret_item6(item6_text)

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
            "item4_purpose_of_transaction": item4_text,
            "item6_contracts_arrangements": item6_text,

            # NEW: interpreted features (Item 4 & Item 6)
            **item4_feat,
            **item6_feat,
        }
        rows.append(merged)

# -------------------------
# 3) If a filing had zero RP rows, keep a “filing-only” row (optional)
# -------------------------

rp_accessions = set(r.get("accession_number") for r in rows if r.get("accession_number"))
for acc, fil in filings.items():
    if acc in rp_accessions:
        continue

    item4_text = (fil.get("item4_purpose_of_transaction") or "").strip()
    item6_text = (fil.get("item6_contracts_arrangements") or "").strip()

    item4_feat = interpret_item4(item4_text)
    item6_feat = interpret_item6(item6_text)

    rows.append({
        "accession_number": acc,
        "filed_as_of_date": fil.get("filed_as_of_date"),
        "conformed_submission_type": fil.get("conformed_submission_type") or fil.get("submission_type"),
        "date_of_event": fil.get("date_of_event"),
        "amendment_number": to_int_or_zero(fil.get("amendment_number")),
        "is_amendment": 1 if to_int_or_zero(fil.get("amendment_number")) > 0 else 0,

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
        "item4_purpose_of_transaction": item4_text,
        "item6_contracts_arrangements": item6_text,

        # NEW: interpreted features
        **item4_feat,
        **item6_feat,
    })

# -------------------------
# 4) Write CSV
# -------------------------

cols = [
    "accession_number","filed_as_of_date","conformed_submission_type","date_of_event",
    "is_amendment","amendment_number",
    "issuer_cik","issuer_name","issuer_cusip",
    "reporting_person_cik","reporting_person_name","type_of_reporting_person",
    "percent_of_class","amount_beneficially_owned",
    "sole_voting_power","shared_voting_power","sole_dispositive_power","shared_dispositive_power",
    "source_of_funds_code","item4_purpose_of_transaction","item6_contracts_arrangements",

    # NEW columns appended
    "item4_severity","item4_is_passive","item4_is_activist","item4_has_mna","item4_has_capital_return",
    "item4_actions","item4_evidence",
    "item6_severity","item6_has_derivatives","item6_has_voting_agreement","item6_has_standstill","item6_has_margin_loan",
    "item6_instruments","item6_evidence",
]

with OUT.open("w", encoding="utf-8") as f:
    f.write(",".join(cols) + "\n")

    def esc(v):
        if v is None:
            return ""
        s = str(v).replace('"', '""')
        # quote if needed
        if any(c in s for c in [",", "\n", "\r"]):
            return f'"{s}"'
        return s

    for r in rows:
        f.write(",".join(esc(r.get(c, "")) for c in cols) + "\n")

print("Wrote:", OUT)
print("Rows:", len(rows))
