"""
Tests for empirical verification of M1 literature deliverables:
- research/literature_matrix.csv
- research/references.bib
- research/literature_review.md
- research/research_gap_assessment.md
"""

import csv
import re
import unicodedata
from pathlib import Path
import pytest

RESEARCH_DIR = Path(__file__).resolve().parent.parent / "research"
CSV_PATH = RESEARCH_DIR / "literature_matrix.csv"
BIB_PATH = RESEARCH_DIR / "references.bib"
REVIEW_PATH = RESEARCH_DIR / "literature_review.md"
GAP_PATH = RESEARCH_DIR / "research_gap_assessment.md"

EXPECTED_CSV_COLUMNS = [
    "Paper_ID",
    "Citation_Key",
    "Authors",
    "Year",
    "Title",
    "Venue",
    "DOI_or_URL",
    "Demand_Regime",
    "Modeling_Paradigm",
    "Forecasting_Methods",
    "Inventory_Setting",
    "Evaluation_Metrics",
    "Key_Empirical_Findings",
    "Relevance_to_Our_Study",
]


def strip_accents(text: str) -> str:
    """Normalize and strip unicode diacritics for robust textual matching."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def parse_bibtex_entries(bib_content: str):
    """
    Robust regex-based parser for BibTeX entries.
    Extracts entry type, citation key, and fields dictionary.
    """
    entries = {}
    pattern = re.compile(r'@(\w+)\s*\{\s*([^,]+),', re.MULTILINE)
    matches = list(pattern.finditer(bib_content))
    
    for i, match in enumerate(matches):
        entry_type = match.group(1).lower()
        key = match.group(2).strip()
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(bib_content)
        entry_body = bib_content[start_idx:end_idx]
        
        # Extract field key-value pairs
        fields = {}
        field_matches = re.findall(
            r'([a-zA-Z_]+)\s*=\s*\{([^}]*)\}', entry_body, re.DOTALL
        )
        for f_name, f_val in field_matches:
            fields[f_name.lower().strip()] = f_val.strip()
        
        entries[key] = {
            "type": entry_type,
            "fields": fields,
        }
    return entries


def test_m1_deliverable_files_exist():
    assert CSV_PATH.exists(), f"Missing {CSV_PATH}"
    assert BIB_PATH.exists(), f"Missing {BIB_PATH}"
    assert REVIEW_PATH.exists(), f"Missing {REVIEW_PATH}"
    assert GAP_PATH.exists(), f"Missing {GAP_PATH}"
    
    assert CSV_PATH.stat().st_size > 0, "literature_matrix.csv is empty"
    assert BIB_PATH.stat().st_size > 0, "references.bib is empty"
    assert REVIEW_PATH.stat().st_size > 0, "literature_review.md is empty"
    assert GAP_PATH.stat().st_size > 0, "research_gap_assessment.md is empty"


def test_literature_matrix_count_and_columns():
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    
    assert headers == EXPECTED_CSV_COLUMNS, f"Headers mismatch: {headers} vs {EXPECTED_CSV_COLUMNS}"
    assert len(rows) >= 20, f"Expected >= 20 papers, found {len(rows)}"
    assert len(rows) == 29, f"Expected 29 surveyed papers, found {len(rows)}"


def test_literature_matrix_no_missing_or_blank_fields():
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row_idx, row in enumerate(rows, start=1):
        for col in EXPECTED_CSV_COLUMNS:
            val = row.get(col)
            assert val is not None, f"Row {row_idx} column '{col}' is None"
            val_clean = val.strip()
            assert val_clean != "", f"Row {row_idx} column '{col}' is empty or whitespace"
            assert val_clean.lower() != "nan", f"Row {row_idx} column '{col}' is NaN"
            assert val_clean.lower() != "null", f"Row {row_idx} column '{col}' is Null"


def test_literature_matrix_keys_and_ids_integrity():
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    seen_ids = set()
    seen_keys = set()
    for row_idx, row in enumerate(rows, start=1):
        paper_id = row["Paper_ID"].strip()
        citation_key = row["Citation_Key"].strip()
        year_str = row["Year"].strip()
        
        # Check ID is numeric and contiguous
        assert paper_id.isdigit(), f"Paper_ID '{paper_id}' in row {row_idx} is not an integer"
        assert int(paper_id) == row_idx, f"Paper_ID {paper_id} != expected index {row_idx}"
        assert paper_id not in seen_ids, f"Duplicate Paper_ID {paper_id}"
        seen_ids.add(paper_id)
        
        # Check Citation_Key is alphanumeric and unique
        assert re.match(r'^[a-zA-Z0-9_-]+$', citation_key), f"Invalid Citation_Key format '{citation_key}'"
        assert citation_key not in seen_keys, f"Duplicate Citation_Key '{citation_key}'"
        seen_keys.add(citation_key)
        
        # Check Year
        assert year_str.isdigit(), f"Year '{year_str}' is not integer"
        year_int = int(year_str)
        assert 1950 <= year_int <= 2026, f"Year {year_int} out of valid range"


def test_literature_matrix_doi_and_url_format():
    doi_regex = re.compile(r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$')
    url_regex = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
    
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row_idx, row in enumerate(rows, start=1):
        doi_or_url = row["DOI_or_URL"].strip()
        is_doi = bool(doi_regex.match(doi_or_url))
        is_url = bool(url_regex.match(doi_or_url))
        assert is_doi or is_url, (
            f"Row {row_idx} ('{row['Citation_Key']}'): DOI_or_URL '{doi_or_url}' "
            f"neither matches valid standard DOI pattern nor valid URL pattern"
        )


def test_bibtex_parsing_and_entry_count():
    bib_content = BIB_PATH.read_text(encoding="utf-8")
    entries = parse_bibtex_entries(bib_content)
    
    assert len(entries) >= 20, f"Expected >= 20 BibTeX entries, found {len(entries)}"
    assert len(entries) == 29, f"Expected 29 BibTeX entries, found {len(entries)}"
    
    # Check each entry has essential fields
    for key, data in entries.items():
        fields = data["fields"]
        assert "author" in fields, f"BibTeX entry '{key}' missing author"
        assert "title" in fields, f"BibTeX entry '{key}' missing title"
        assert "year" in fields, f"BibTeX entry '{key}' missing year"
        assert fields["author"].strip() != "", f"BibTeX entry '{key}' author is empty"
        assert fields["title"].strip() != "", f"BibTeX entry '{key}' title is empty"
        assert fields["year"].strip().isdigit(), f"BibTeX entry '{key}' year is non-numeric"


def test_100_percent_key_match_between_csv_and_bibtex():
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_keys = {row["Citation_Key"].strip() for row in reader}
    
    bib_content = BIB_PATH.read_text(encoding="utf-8")
    bib_entries = parse_bibtex_entries(bib_content)
    bib_keys = set(bib_entries.keys())
    
    csv_missing_in_bib = csv_keys - bib_keys
    bib_missing_in_csv = bib_keys - csv_keys
    
    assert len(csv_missing_in_bib) == 0, f"Keys in CSV but missing in BibTeX: {csv_missing_in_bib}"
    assert len(bib_missing_in_csv) == 0, f"Keys in BibTeX but missing in CSV: {bib_missing_in_csv}"
    assert csv_keys == bib_keys, "Set of CSV keys does not exactly match BibTeX keys"


def test_metadata_consistency_csv_and_bibtex():
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_data = {row["Citation_Key"].strip(): row for row in reader}
    
    bib_content = BIB_PATH.read_text(encoding="utf-8")
    bib_entries = parse_bibtex_entries(bib_content)
    
    for key, c_row in csv_data.items():
        b_entry = bib_entries[key]
        b_fields = b_entry["fields"]
        
        # Check Year matches exactly
        assert c_row["Year"].strip() == b_fields["year"].strip(), (
            f"Year mismatch for {key}: CSV={c_row['Year']} vs BibTeX={b_fields['year']}"
        )
        
        # Check DOI / URL correspondence
        c_doi_url = c_row["DOI_or_URL"].strip()
        b_doi = b_fields.get("doi", "").strip()
        b_url = b_fields.get("url", "").strip()
        
        if c_doi_url.startswith("10."):
            assert b_doi.lower() == c_doi_url.lower(), (
                f"DOI mismatch for {key}: CSV={c_doi_url} vs BibTeX={b_doi}"
            )
        elif c_doi_url.startswith("http"):
            assert b_url == c_doi_url, (
                f"URL mismatch for {key}: CSV={c_doi_url} vs BibTeX={b_url}"
            )


def test_literature_review_markdown_depth_and_structure():
    content = REVIEW_PATH.read_text(encoding="utf-8")
    
    # Word count and size checks
    words = content.split()
    assert len(words) >= 3000, f"Literature review too brief: {len(words)} words (expected >= 3000)"
    assert REVIEW_PATH.stat().st_size >= 25000, f"Literature review file too small: {REVIEW_PATH.stat().st_size} bytes"
    
    # Pillar coverage checks
    required_pillars = [
        "Pillar 1: The Forecast Accuracy vs. Inventory Performance Disconnect",
        "Pillar 2: Intermittent & Zero-Inflated Retail Demand Dynamics",
        "Pillar 3: Loss Functions, Pinball Loss, and Asymmetric Operational Objectives",
        "Pillar 4: Machine Learning Ensembles & Deep Learning in Retail Replenishment",
        "Pillar 5: Continuous-Review",
    ]
    for pillar in required_pillars:
        assert pillar.lower() in content.lower(), f"Missing required pillar: '{pillar}'"
    
    # Mathematical formulation checks
    math_tokens = [
        "C_{total}",
        "C_{holding}",
        "C_{shortage}",
        "ROP",
        "EOQ",
        "Z_\\alpha",
        "\\sigma_e",
        "\\sigma_L",
        "\\rho_q",
        "Tracking Signal",
        "WAPE",
        "Bias\\%",
    ]
    for token in math_tokens:
        assert token in content, f"Missing mathematical formulation token: '{token}'"


def test_policy_definitions_across_m1_deliverables():
    rev_content = REVIEW_PATH.read_text(encoding="utf-8")
    gap_content = GAP_PATH.read_text(encoding="utf-8")
    combined = rev_content + "\n" + gap_content
    
    required_policies = [
        "Policy A",
        "Policy B",
        "Policy C",
        "Policy D",
    ]
    for pol in required_policies:
        assert pol in combined, f"Missing required policy definition across deliverables: '{pol}'"
    
    # Ensure specific mathematical formulas for Policy C and Policy D exist
    assert "Bias%" in combined
    assert "5%" in combined or "5.0%" in combined
    assert "MAE" in combined


def test_research_gap_assessment_depth_and_structure():
    content = GAP_PATH.read_text(encoding="utf-8")
    
    # Word count and size checks
    words = content.split()
    assert len(words) >= 1500, f"Gap assessment too brief: {len(words)} words (expected >= 1500)"
    assert GAP_PATH.stat().st_size >= 12000, f"Gap assessment file too small: {GAP_PATH.stat().st_size} bytes"
    
    # Check all four gaps
    assert "Gap 1:" in content, "Missing Gap 1 specification"
    assert "Gap 2:" in content, "Missing Gap 2 specification"
    assert "Gap 3:" in content, "Missing Gap 3 specification"
    assert "Gap 4:" in content, "Missing Gap 4 specification"
    
    # Check all research questions
    assert "RQ1" in content, "Missing RQ1"
    assert "RQ2" in content, "Missing RQ2"
    assert "RQ3" in content, "Missing RQ3"
    assert "RQ4" in content, "Missing RQ4"
    
    # Check policy definitions and thresholds
    assert "Policy A" in content
    assert "Policy B" in content
    assert "Policy C" in content
    assert "Policy D" in content
    assert "5.0%" in content or "5%" in content
    assert "Jensen's Inequality" in content


def test_citation_coverage_across_deliverables():
    rev_content = strip_accents(REVIEW_PATH.read_text(encoding="utf-8")).lower()
    gap_content = strip_accents(GAP_PATH.read_text(encoding="utf-8")).lower()
    combined_content = rev_content + "\n" + gap_content
    
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    covered_in_rev = 0
    covered_in_any = 0
    uncited_any = []
    
    for row in rows:
        key = row["Citation_Key"].strip().lower()
        authors_raw = row["Authors"].strip()
        first_author = authors_raw.split(",")[0].strip()
        first_author_clean = strip_accents(first_author).lower()
        
        in_rev = key in rev_content or first_author_clean in rev_content
        in_any = key in combined_content or first_author_clean in combined_content
        
        if in_rev:
            covered_in_rev += 1
        if in_any:
            covered_in_any += 1
        else:
            uncited_any.append((key, first_author))
    
    total = len(rows)
    cov_rev_pct = covered_in_rev / total
    cov_any_pct = covered_in_any / total
    
    # In review alone, at least 80% of papers are cited
    assert cov_rev_pct >= 0.80, f"Coverage in literature review {cov_rev_pct:.1%} < 80%"
    # Across both deliverables, at least 90% of papers are cited
    assert cov_any_pct >= 0.90, f"Coverage across deliverables {cov_any_pct:.1%} < 90%, uncited: {uncited_any}"
