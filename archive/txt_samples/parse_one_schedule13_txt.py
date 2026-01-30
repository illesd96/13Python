"""
Parse ONE SEC Schedule 13 .txt filing (SEC-DOCUMENT container) and extract:
1) Header fields from <SEC-HEADER> (accession, filed date, form type, subject company, filer)
2) Structured XML from <XML>...</XML> inside <TEXT>
3) Key Schedule 13 fields from the structured XML (issuer, reporting persons, item 3/4/6, etc.)

Usage:
  python parse_one_schedule13_txt.py /path/to/0000928785-26-000001.txt
"""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List


def extract_block(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    start = text.find(start_tag)
    if start == -1:
        return None
    end = text.find(end_tag, start)
    if end == -1:
        return None
    return text[start + len(start_tag) : end]


def extract_last_block(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Useful for <XML>...</XML> where there may be multiple docs; take the last one."""
    start = text.rfind(start_tag)
    if start == -1:
        return None
    end = text.find(end_tag, start)
    if end == -1:
        return None
    return text[start + len(start_tag) : end]


def header_kv(sec_header: str, key: str) -> Optional[str]:
    """
    Extracts KEY: <value> lines from SEC-HEADER text (tab/spaces tolerant).
    Example: 'ACCESSION NUMBER:\t\t0000928785-26-000001'
    """
    # Match "KEY:" then anything until end-of-line
    m = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", sec_header)
    return m.group(1).strip() if m else None


def header_section(sec_header: str, section_title: str) -> Optional[str]:
    """
    Extracts the block after e.g. 'SUBJECT COMPANY:' up to the next all-caps-ish label line.
    Not perfect, but works well for SEC-HEADER layouts.
    """
    # Find the section title line
    m = re.search(rf"(?im)^\s*{re.escape(section_title)}\s*:\s*$", sec_header)
    if not m:
        return None
    start = m.end()

    # Section ends at next line that looks like "SOMETHING:" at column start (e.g., FILED BY:)
    m2 = re.search(r"(?im)^\s*[A-Z0-9][A-Z0-9 \-/&]+\s*:\s*$", sec_header[start:])
    end = start + m2.start() if m2 else len(sec_header)
    return sec_header[start:end].strip()


def header_company_data(section_text: str) -> Dict[str, Optional[str]]:
    """
    From a section (SUBJECT COMPANY or FILED BY), pull:
      - COMPANY CONFORMED NAME
      - CENTRAL INDEX KEY
      - FORM TYPE (if present inside the section)
    """
    out = {
        "company_name": None,
        "cik": None,
        "form_type": None,
    }
    if not section_text:
        return out
    out["company_name"] = header_kv(section_text, "COMPANY CONFORMED NAME")
    out["cik"] = header_kv(section_text, "CENTRAL INDEX KEY")
    out["form_type"] = header_kv(section_text, "FORM TYPE")
    return out


def parse_structured_xml(xml_str: str) -> Dict[str, object]:
    """
    Parse the Schedule 13 structured XML (your example uses namespace schedule13D).
    Tries lxml first; falls back to ElementTree if lxml isn't available.
    """
    try:
        from lxml import etree  # type: ignore
        parser = "lxml"
        root = etree.fromstring(xml_str.encode("utf-8", errors="ignore"))
        default_ns = root.nsmap.get(None)
        ns = {"s": default_ns} if default_ns else {}

        def x1(path: str) -> Optional[str]:
            res = root.xpath(path, namespaces=ns) if ns else root.xpath(path)
            if not res:
                return None
            el = res[0]
            if isinstance(el, str):
                return el.strip() or None
            return ((el.text or "").strip()) or None

        def xall(path: str) -> List[str]:
            res = root.xpath(path, namespaces=ns) if ns else root.xpath(path)
            out: List[str] = []
            for el in res:
                if isinstance(el, str):
                    v = el.strip()
                else:
                    v = (el.text or "").strip()
                if v:
                    out.append(v)
            return out

        # Build paths with namespace prefix (your example shows schedule13D default ns)
        def p(x: str) -> str:
            return x.replace("/", "/s:") if ns else x

        base = "/edgarSubmission"
        if ns:
            base = "/s:edgarSubmission"

        data: Dict[str, object] = {"_xml_parser": parser, "_xml_default_ns": default_ns}

        data["submission_type"] = x1(f"{base}/s:headerData/s:submissionType")
        data["securities_class_title"] = x1(f"{base}/s:formData/s:coverPageHeader/s:securitiesClassTitle")
        data["date_of_event"] = x1(f"{base}/s:formData/s:coverPageHeader/s:dateOfEvent")
        data["previously_filed_flag"] = x1(f"{base}/s:formData/s:coverPageHeader/s:previouslyFiledFlag")

        data["issuer_cik"] = x1(f"{base}/s:formData/s:coverPageHeader/s:issuerInfo/s:issuerCIK")
        data["issuer_cusip"] = x1(f"{base}/s:formData/s:coverPageHeader/s:issuerInfo/s:issuerCUSIP")
        data["issuer_name"] = x1(f"{base}/s:formData/s:coverPageHeader/s:issuerInfo/s:issuerName")

        # Items 3/4/6
        data["item3_source_of_funds"] = x1(f"{base}/s:formData/s:items1To7/s:item3/s:fundsSource")
        data["item4_purpose_of_transaction"] = x1(f"{base}/s:formData/s:items1To7/s:item4/s:transactionPurpose")
        data["item6_contracts_arrangements"] = x1(f"{base}/s:formData/s:items1To7/s:item6/s:contractDescription")

        # Reporting persons
        rps = root.xpath(f"{base}/s:formData/s:reportingPersons/s:reportingPersonInfo", namespaces=ns)
        rp_rows = []
        for rp in rps:
            def rp1(rel: str) -> Optional[str]:
                r = rp.xpath(rel, namespaces=ns)
                if not r:
                    return None
                return ((r[0].text or "").strip()) or None

            def rpall(rel: str) -> List[str]:
                r = rp.xpath(rel, namespaces=ns)
                return [((el.text or "").strip()) for el in r if (el.text or "").strip()]

            rp_rows.append({
                "reporting_person_name": rp1("./s:reportingPersonName"),
                "reporting_person_cik": rp1("./s:reportingPersonCIK"),
                "no_cik_flag": rp1("./s:reportingPersonNoCIK"),
                "member_of_group": rp1("./s:memberOfGroup"),
                "citizenship_or_org": rp1("./s:citizenshipOrOrganization"),
                "sole_voting_power": rp1("./s:soleVotingPower"),
                "shared_voting_power": rp1("./s:sharedVotingPower"),
                "sole_dispositive_power": rp1("./s:soleDispositivePower"),
                "shared_dispositive_power": rp1("./s:sharedDispositivePower"),
                "aggregate_amount_owned": rp1("./s:aggregateAmountOwned"),
                "percent_of_class": rp1("./s:percentOfClass"),
                "type_of_reporting_person": rpall("./s:typeOfReportingPerson"),
            })

        data["reporting_persons"] = rp_rows
        return data

    except ModuleNotFoundError:
        # Fallback: ElementTree (less convenient with namespaces, but works)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_str)

        def lname(tag: str) -> str:
            return tag.split("}")[-1]

        def find_first_by_localname(local: str) -> Optional[str]:
            for el in root.iter():
                if lname(el.tag) == local and el.text and el.text.strip():
                    return el.text.strip()
            return None

        def find_path_text(path: List[str]) -> Optional[str]:
            cur = root
            for step in path:
                nxt = None
                for child in cur:
                    if lname(child.tag) == step:
                        nxt = child
                        break
                if nxt is None:
                    return None
                cur = nxt
            return (cur.text or "").strip() or None

        data: Dict[str, object] = {"_xml_parser": "ElementTree", "_xml_default_ns": None}

        # Basic fields (best-effort)
        data["submission_type"] = find_path_text(["headerData", "submissionType"])
        data["securities_class_title"] = find_path_text(["formData", "coverPageHeader", "securitiesClassTitle"])
        data["date_of_event"] = find_path_text(["formData", "coverPageHeader", "dateOfEvent"])
        data["previously_filed_flag"] = find_path_text(["formData", "coverPageHeader", "previouslyFiledFlag"])

        data["issuer_cik"] = find_path_text(["formData", "coverPageHeader", "issuerInfo", "issuerCIK"])
        data["issuer_cusip"] = find_path_text(["formData", "coverPageHeader", "issuerInfo", "issuerCUSIP"])
        data["issuer_name"] = find_path_text(["formData", "coverPageHeader", "issuerInfo", "issuerName"])

        data["item3_source_of_funds"] = find_path_text(["formData", "items1To7", "item3", "fundsSource"])
        data["item4_purpose_of_transaction"] = find_path_text(["formData", "items1To7", "item4", "transactionPurpose"])
        data["item6_contracts_arrangements"] = find_path_text(["formData", "items1To7", "item6", "contractDescription"])

        # Reporting persons
        rp_rows = []
        # Locate reportingPersons node
        formData = None
        for el in root:
            if lname(el.tag) == "formData":
                formData = el
                break
        if formData is not None:
            reportingPersons = None
            for el in formData:
                if lname(el.tag) == "reportingPersons":
                    reportingPersons = el
                    break
            if reportingPersons is not None:
                for rp in reportingPersons:
                    if lname(rp.tag) != "reportingPersonInfo":
                        continue
                    def child_text(rp_el, child_local) -> Optional[str]:
                        for c in rp_el:
                            if lname(c.tag) == child_local and (c.text or "").strip():
                                return c.text.strip()
                        return None

                    def child_texts(rp_el, child_local) -> List[str]:
                        out = []
                        for c in rp_el:
                            if lname(c.tag) == child_local and (c.text or "").strip():
                                out.append(c.text.strip())
                        return out

                    rp_rows.append({
                        "reporting_person_name": child_text(rp, "reportingPersonName"),
                        "reporting_person_cik": child_text(rp, "reportingPersonCIK"),
                        "no_cik_flag": child_text(rp, "reportingPersonNoCIK"),
                        "member_of_group": child_text(rp, "memberOfGroup"),
                        "citizenship_or_org": child_text(rp, "citizenshipOrOrganization"),
                        "sole_voting_power": child_text(rp, "soleVotingPower"),
                        "shared_voting_power": child_text(rp, "sharedVotingPower"),
                        "sole_dispositive_power": child_text(rp, "soleDispositivePower"),
                        "shared_dispositive_power": child_text(rp, "sharedDispositivePower"),
                        "aggregate_amount_owned": child_text(rp, "aggregateAmountOwned"),
                        "percent_of_class": child_text(rp, "percentOfClass"),
                        "type_of_reporting_person": child_texts(rp, "typeOfReportingPerson"),
                    })

        data["reporting_persons"] = rp_rows
        return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_one_schedule13_txt.py /path/to/filing.txt")
        sys.exit(1)

    txt_path = Path(sys.argv[1])
    raw = txt_path.read_text(errors="ignore")

    # 1) SEC-HEADER block
    sec_header = extract_block(raw, "<SEC-HEADER>", "</SEC-HEADER>")
    if not sec_header:
        raise ValueError("Could not find <SEC-HEADER>...</SEC-HEADER> block")

    accession = header_kv(sec_header, "ACCESSION NUMBER")
    conformed_type = header_kv(sec_header, "CONFORMED SUBMISSION TYPE")
    filed_as_of = header_kv(sec_header, "FILED AS OF DATE")
    acceptance_dt = re.search(r"(?im)^\s*<ACCEPTANCE-DATETIME>\s*(\d{14})\s*$", raw)
    acceptance_dt = acceptance_dt.group(1) if acceptance_dt else None

    def header_find_company_pairs(sec_header: str):
        """
        Finds all occurrences of:
          COMPANY CONFORMED NAME: <name>
          CENTRAL INDEX KEY: <cik>
        Returns list of dicts in appearance order.
        """
        pattern = re.compile(
            r"(?im)^\s*COMPANY CONFORMED NAME:\s*(.+?)\s*$\s*^\s*CENTRAL INDEX KEY:\s*(.+?)\s*$",
            re.MULTILINE
        )
        pairs = []
        for m in pattern.finditer(sec_header):
            name = m.group(1).strip()
            cik = m.group(2).strip()
            pairs.append({"company_name": name, "cik": cik})
        return pairs

    pairs = header_find_company_pairs(sec_header)

    # Usually: first pair = SUBJECT COMPANY, second pair = FILED BY
    subject = pairs[0] if len(pairs) >= 1 else {"company_name": None, "cik": None}
    filer = pairs[1] if len(pairs) >= 2 else {"company_name": None, "cik": None}

    print("=== HEADER (SEC-HEADER) ===")
    print("accession_number:         ", accession)
    print("conformed_submission_type:", conformed_type)
    print("filed_as_of_date:         ", filed_as_of)
    print("acceptance_datetime:      ", acceptance_dt)
    print("--- SUBJECT COMPANY ---")
    print("subject_company_name:     ", subject["company_name"])
    print("subject_company_cik:      ", subject["cik"])
    print("--- FILED BY ---")
    print("filed_by_name:            ", filer["company_name"])
    print("filed_by_cik:             ", filer["cik"])


    # 2) Extract embedded XML (take last <XML> block, safest if multiple docs)
    xml_str = extract_last_block(raw, "<XML>", "</XML>")
    if not xml_str:
        raise ValueError("Could not find <XML>...</XML> block inside the .txt")

    xml_str = xml_str.strip()
    parsed = parse_structured_xml(xml_str)

    print("\n=== STRUCTURED XML (extracted from TXT) ===")
    print("xml_parser:               ", parsed.get("_xml_parser"))
    print("xml_default_namespace:    ", parsed.get("_xml_default_ns"))
    print("submission_type:          ", parsed.get("submission_type"))
    print("securities_class_title:   ", parsed.get("securities_class_title"))
    print("date_of_event:            ", parsed.get("date_of_event"))
    print("previously_filed_flag:    ", parsed.get("previously_filed_flag"))
    print("issuer_cik:               ", parsed.get("issuer_cik"))
    print("issuer_cusip:             ", parsed.get("issuer_cusip"))
    print("issuer_name:              ", parsed.get("issuer_name"))

    print("\n--- Items (text) ---")
    for k in ["item3_source_of_funds", "item4_purpose_of_transaction", "item6_contracts_arrangements"]:
        v = parsed.get(k)
        if isinstance(v, str) and len(v) > 220:
            v = v[:220] + "..."
        print(f"{k}: {v}")

    print("\n--- Reporting Persons ---")
    rps = parsed.get("reporting_persons", [])
    if not rps:
        print("Found 0 reporting persons")
    else:
        print(f"Found {len(rps)} reporting person(s)\n")
        for i, rp in enumerate(rps, start=1):
            print(f"RP #{i}")
            print("  name:                   ", rp.get("reporting_person_name"))
            print("  cik:                    ", rp.get("reporting_person_cik"))
            print("  type(s):                ", rp.get("type_of_reporting_person"))
            print("  amount beneficially owned:", rp.get("aggregate_amount_owned"))
            print("  percent of class:       ", rp.get("percent_of_class"))
            print("  sole/shared voting:     ", rp.get("sole_voting_power"), "/", rp.get("shared_voting_power"))
            print("  sole/shared dispositive:", rp.get("sole_dispositive_power"), "/", rp.get("shared_dispositive_power"))
            print()

    # Cross-check: which filed date to prefer?
    # - SEC-HEADER filed_as_of_date is YYYYMMDD
    # - XML has date_of_event, not filed date
    print("=== NOTE ===")
    print("Use SEC-HEADER FILED AS OF DATE as filed_date; XML dateOfEvent as event_date.")


if __name__ == "__main__":
    main()
