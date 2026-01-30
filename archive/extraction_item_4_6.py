#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interpret SEC Schedule 13D/13G Item 4 and Item 6 text from scoring_table.csv

INPUT (FIXED):
  data/parsed/scoring_table.csv

REQUIRED COLUMNS:
  accession_number
  item4_purpose_of_transaction
  item6_contracts_arrangements

OPTIONAL (used if present):
  issuer_cusip
  issuer_cik
  issuer_name
  reporting_person_name
  filed_as_of_date

OUTPUT:
  data/interpreted/item4_item6.jsonl
  data/features/item4_item6_flat.csv
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# =====================
# Text helpers
# =====================

def normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", text) if s.strip()]

def compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]

def find_evidence(sentences: List[str], patterns: List[re.Pattern], limit: int = 12) -> List[str]:
    hits = []
    for s in sentences:
        for p in patterns:
            if p.search(s):
                hits.append(s)
                break
    return list(dict.fromkeys(hits))[:limit]

# =====================
# Item 4 rules
# =====================

ITEM4_RULES = {
    "activist": [
        r"\bproxy\b", r"\bproxy contest\b", r"\bconsent solicitation\b",
        r"\bboard\b", r"\bdirector\b", r"\bnominate\b"
    ],
    "mna": [
        r"\bmerger\b", r"\bacquisition\b", r"\btender offer\b",
        r"\bgo private\b", r"\bsale of\b", r"\bstrategic alternatives?\b"
    ],
    "capital_return": [
        r"\bbuyback\b", r"\brepurchase\b", r"\bdividend\b"
    ],
    "passive": [
        r"\bfor investment purposes only\b",
        r"\bno present plans\b",
        r"\bnot currently\b.*\bplans\b"
    ],
}

ITEM4_ACTIONS = {
    "discuss_with_management": [
        r"\bdiscuss\b", r"\bengage\b.*\bmanagement\b"
    ],
    "seek_board_seats": [
        r"\bseek\b.*\bboard\b", r"\bnominate\b"
    ],
    "proxy_contest": [
        r"\bproxy contest\b", r"\bsolicit\b.*\bproxy\b"
    ],
    "propose_transaction": [
        r"\bpropose\b.*\btransaction\b", r"\bseek\b.*\bmerger\b"
    ],
}

def extract_item4(text: str) -> Dict[str, Any]:
    text_n = normalize(text)
    sents = split_sentences(text_n)
    lower = text_n.lower()

    intent_flags, actions, evidence = [], [], []

    for k, pats in ITEM4_RULES.items():
        cp = compile_patterns(pats)
        if any(p.search(lower) for p in cp):
            intent_flags.append(k)
            evidence += find_evidence(sents, cp)

    for k, pats in ITEM4_ACTIONS.items():
        cp = compile_patterns(pats)
        if any(p.search(lower) for p in cp):
            actions.append(k)
            evidence += find_evidence(sents, cp)

    severity = 0
    if intent_flags or actions:
        severity = 1
    if any(x in intent_flags for x in ("activist", "mna", "capital_return")):
        severity = 2
    if any(x in actions for x in ("proxy_contest", "seek_board_seats", "propose_transaction")):
        severity = 3

    return {
        "intent_flags": intent_flags,
        "actions_mentioned": actions,
        "severity_score": severity,
        "evidence_spans": list(dict.fromkeys(evidence))[:12]
    }

# =====================
# Item 6 rules
# =====================

ITEM6_INSTRUMENTS = {
    "swap": [r"\bswap\b", r"\btotal return swap\b"],
    "option": [r"\boption\b"],
    "margin_loan": [r"\bmargin\b", r"\bpledge\b", r"\bloan\b"],
    "voting_agreement": [r"\bvoting agreement\b", r"\birrevocable proxy\b"],
    "standstill": [r"\bstandstill\b"]
}

def extract_item6(text: str) -> Dict[str, Any]:
    text_n = normalize(text)
    sents = split_sentences(text_n)
    lower = text_n.lower()

    instruments, evidence = [], []

    for k, pats in ITEM6_INSTRUMENTS.items():
        cp = compile_patterns(pats)
        if any(p.search(lower) for p in cp):
            instruments.append(k)
            evidence += find_evidence(sents, cp)

    has_derivatives = any(x in instruments for x in ("swap", "option"))

    severity = 0
    if instruments:
        severity = 1
    if has_derivatives or "margin_loan" in instruments:
        severity = 2
    if "voting_agreement" in instruments or "standstill" in instruments:
        severity = 3

    return {
        "instruments": instruments,
        "economic_exposure": {
            "has_derivatives": has_derivatives
        },
        "severity_score": severity,
        "evidence_spans": list(dict.fromkeys(evidence))[:12]
    }

# =====================
# Main
# =====================

def main():
    input_path = os.path.expanduser("~/Documents/Python/sec_13dg_txt_2025/scoring_table.csv")
    out_jsonl = os.path.expanduser(
    "~/Documents/Python/sec_13dg_txt_2025/data/interpreted/item4_item6.jsonl"
    )
    out_flat = os.path.expanduser(
    "~/Documents/Python/sec_13dg_txt_2025/data/features/item4_item6_flat.csv"
    )

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(out_flat).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    jsonl_rows = []
    flat_rows = []

    for r in rows:
        item4 = extract_item4(r.get("item4_purpose_of_transaction", ""))
        item6 = extract_item6(r.get("item6_contracts_arrangements", ""))

        record = {
            "accession_number": r.get("accession_number"),
            "issuer_cusip": r.get("issuer_cusip"),
            "issuer_cik": r.get("issuer_cik"),
            "issuer_name": r.get("issuer_name"),
            "reporting_person_name": r.get("reporting_person_name"),
            "filed_as_of_date": r.get("filed_as_of_date"),
            "item4": item4,
            "item6": item6
        }

        jsonl_rows.append(record)

        flat_rows.append({
            "accession_number": r.get("accession_number"),
            "item4_severity": item4["severity_score"],
            "item4_is_activist": "activist" in item4["intent_flags"],
            "item4_has_mna": "mna" in item4["intent_flags"],
            "item6_severity": item6["severity_score"],
            "item6_has_derivatives": item6["economic_exposure"]["has_derivatives"],
            "item6_has_voting_agreement": "voting_agreement" in item6["instruments"]
        })

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in jsonl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_flat, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
        w.writeheader()
        w.writerows(flat_rows)

    print("DONE")
    print("Read from:", input_path)
    print("Wrote:", out_jsonl)
    print("Wrote:", out_flat)
    print("Rows:", len(rows))
    print("CWD:", os.getcwd())

if __name__ == "__main__":
    main()
