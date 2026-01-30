"""
Batch-parse SEC Schedule 13 .txt files in a folder.

Outputs:
  - filings.jsonl (one JSON per filing)
  - reporting_persons.jsonl (one JSON per reporting person row)

Usage:
  python3 parse_folder_schedule13_txt.py /path/to/folder_with_txt
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List

from lxml import etree


def extract_block(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    start = text.find(start_tag)
    if start == -1:
        return None
    end = text.find(end_tag, start)
    if end == -1:
        return None
    return text[start + len(start_tag) : end]


def extract_last_block(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    start = text.rfind(start_tag)
    if start == -1:
        return None
    end = text.find(end_tag, start)
    if end == -1:
        return None
    return text[start + len(start_tag) : end]


def header_kv(sec_header: str, key: str) -> Optional[str]:
    m = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", sec_header)
    return m.group(1).strip() if m else None


def header_find_company_pairs(sec_header: str) -> List[Dict[str, Optional[str]]]:
    pattern = re.compile(
        r"(?im)^\s*COMPANY CONFORMED NAME:\s*(.+?)\s*$\s*^\s*CENTRAL INDEX KEY:\s*(.+?)\s*$",
        re.MULTILINE
    )
    pairs = []
    for m in pattern.finditer(sec_header):
        pairs.append({"company_name": m.group(1).strip(), "cik": m.group(2).strip()})
    return pairs


def parse_structured_xml(xml_str: str) -> Dict[str, object]:
    root = etree.fromstring(xml_str.encode("utf-8", errors="ignore"))

    def xfirst(*local_names: str) -> Optional[str]:
        for ln in local_names:
            res = root.xpath(f"//*[local-name()='{ln}']/text()")
            if res:
                v = str(res[0]).strip()
                if v:
                    return v
        return None

    def xall(local_name: str) -> List[str]:
        res = root.xpath(f"//*[local-name()='{local_name}']/text()")
        out = []
        for r in res:
            v = str(r).strip()
            if v:
                out.append(v)
        return out

    data: Dict[str, object] = {}
    data["xml_default_namespace"] = root.nsmap.get(None)

    # Works for both 13D and 13G schemas
    data["submission_type"] = xfirst("submissionType")

    # 13D: dateOfEvent
    # 13G: eventDateRequiresFilingThisStatement (see your sample) :contentReference[oaicite:1]{index=1}
    data["date_of_event"] = xfirst("dateOfEvent", "eventDateRequiresFilingThisStatement")

    # Amendment number is present in many 13D/A XMLs; 13G/A may or may not include it
    data["amendment_number"] = xfirst("amendmentNo", "amendmentNumber")

    # Issuer fields (13D vs 13G casing differences)
    data["issuer_cik"] = xfirst("issuerCIK", "issuerCik")
    data["issuer_cusip"] = xfirst("issuerCUSIP", "issuerCusip")
    data["issuer_name"] = xfirst("issuerName")

    # Replace "item3" with a dedicated "source of funds code" extraction (where present)
    data["source_of_funds_code"] = xfirst("fundType", "fundsSource", "sourceOfFunds")

    # Keep these (mostly 13D-relevant; harmless if missing)
    data["item4_purpose_of_transaction"] = xfirst("transactionPurpose", "purposeOfTransaction")
    data["item6_contracts_arrangements"] = xfirst("contractDescription", "contractsArrangements")

    # Reporting persons:
    # 13D schema often has reportingPersonInfo
    # 13G schema uses coverPageHeaderReportingPersonDetails (as in your sample) :contentReference[oaicite:2]{index=2}
    rp_nodes = root.xpath(
        "//*[local-name()='reportingPersonInfo'] | "
        "//*[local-name()='coverPageHeaderReportingPersonDetails']"
    )

    rp_rows = []
    for rp in rp_nodes:
        def rpfirst(*locals_: str) -> Optional[str]:
            for local in locals_:
                r = rp.xpath(f".//*[local-name()='{local}']/text()")
                if r:
                    v = str(r[0]).strip()
                    if v:
                        return v
            return None

        def rpall(local: str) -> List[str]:
            r = rp.xpath(f".//*[local-name()='{local}']/text()")
            return [str(x).strip() for x in r if str(x).strip()]

        rp_rows.append({
            "reporting_person_name": rpfirst("reportingPersonName", "filingPersonName", "name"),
            "reporting_person_cik": rpfirst("reportingPersonCIK", "cik"),

            # These exist in your 13G sample under reportingPersonBeneficiallyOwnedNumberOfShares :contentReference[oaicite:3]{index=3}
            "sole_voting_power": rpfirst("soleVotingPower", "solePowerOrDirectToVote"),
            "shared_voting_power": rpfirst("sharedVotingPower", "sharedPowerOrDirectToVote"),
            "sole_dispositive_power": rpfirst("soleDispositivePower", "solePowerOrDirectToDispose"),
            "shared_dispositive_power": rpfirst("sharedDispositivePower", "sharedPowerOrDirectToDispose"),

            # 13D often: aggregateAmountOwned + percentOfClass
            # 13G often: reportingPersonBeneficiallyOwnedAggregateNumberOfShares + classPercent :contentReference[oaicite:4]{index=4}
            "aggregate_amount_owned": rpfirst("aggregateAmountOwned", "reportingPersonBeneficiallyOwnedAggregateNumberOfShares"),
            "percent_of_class": rpfirst("percentOfClass", "classPercent"),

            "type_of_reporting_person": rpall("typeOfReportingPerson"),
        })

    data["reporting_persons"] = rp_rows
    return data


def parse_one_txt(path: Path) -> Dict[str, object]:
    raw = path.read_text(errors="ignore")

    sec_header = extract_block(raw, "<SEC-HEADER>", "</SEC-HEADER>")
    if not sec_header:
        raise ValueError("Missing SEC-HEADER")

    accession = header_kv(sec_header, "ACCESSION NUMBER")
    conformed_type = header_kv(sec_header, "CONFORMED SUBMISSION TYPE")
    filed_as_of = header_kv(sec_header, "FILED AS OF DATE")

    is_amendment = bool(conformed_type and conformed_type.strip().upper().endswith("/A"))

    acc_match = re.search(r"(?im)^\s*<ACCEPTANCE-DATETIME>\s*(\d{14})\s*$", raw)
    acceptance_dt = acc_match.group(1) if acc_match else None

    pairs = header_find_company_pairs(sec_header)
    subject = pairs[0] if len(pairs) >= 1 else {"company_name": None, "cik": None}
    filer = pairs[1] if len(pairs) >= 2 else {"company_name": None, "cik": None}

    xml_str = extract_last_block(raw, "<XML>", "</XML>")
    if not xml_str:
        raise ValueError("Missing embedded <XML> block")

    parsed_xml = parse_structured_xml(xml_str.strip())

    filing = {
        "source_file": path.name,
        "accession_number": accession,
        "conformed_submission_type": conformed_type,
        "filed_as_of_date": filed_as_of,
        "acceptance_datetime": acceptance_dt,
        "subject_company_name": subject["company_name"],
        "subject_company_cik": subject["cik"],
        "filed_by_name": filer["company_name"],
        "filed_by_cik": filer["cik"],

        "xml_namespace": parsed_xml.get("xml_default_namespace"),
        "submission_type": parsed_xml.get("submission_type"),

        "is_amendment": is_amendment,
        "date_of_event": parsed_xml.get("date_of_event"),
        "amendment_number": parsed_xml.get("amendment_number"),

        "issuer_cik": parsed_xml.get("issuer_cik"),
        "issuer_cusip": parsed_xml.get("issuer_cusip"),
        "issuer_name": parsed_xml.get("issuer_name"),

        # Item3 helyett ez:
        "source_of_funds_code": parsed_xml.get("source_of_funds_code"),

        "item4_purpose_of_transaction": parsed_xml.get("item4_purpose_of_transaction"),
        "item6_contracts_arrangements": parsed_xml.get("item6_contracts_arrangements"),
    }

    reporting_persons = parsed_xml.get("reporting_persons", [])
    return {"filing": filing, "reporting_persons": reporting_persons}

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_folder_schedule13_txt.py /path/to/folder_with_txt")
        sys.exit(1)

    folder = Path(sys.argv[1]).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Not a folder: {folder}")

    out_filings = folder / "filings.jsonl"
    out_rps = folder / "reporting_persons.jsonl"
    out_errors = folder / "errors.jsonl"

    txt_files = sorted(folder.rglob("*.txt"))
    print(f"Found {len(txt_files)} .txt files in {folder}")

    n_ok = 0
    n_err = 0

    with out_filings.open("w", encoding="utf-8") as f_fil, \
         out_rps.open("w", encoding="utf-8") as f_rp, \
         out_errors.open("w", encoding="utf-8") as f_err:

        for p in txt_files:
            try:
                res = parse_one_txt(p)
                filing = res["filing"]
                rps = res["reporting_persons"]

                f_fil.write(json.dumps(filing, ensure_ascii=False) + "\n")
                for rp in rps:
                    rp_row = {"accession_number": filing["accession_number"], **rp}
                    f_rp.write(json.dumps(rp_row, ensure_ascii=False) + "\n")

                n_ok += 1
            except Exception as e:
                n_err += 1
                f_err.write(json.dumps({"file": p.name, "error": str(e)}, ensure_ascii=False) + "\n")

    print(f"Done. OK={n_ok}, ERR={n_err}")
    print(f"Wrote: {out_filings.name}, {out_rps.name}, {out_errors.name}")


if __name__ == "__main__":
    main()
