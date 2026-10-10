"""
generate_zenodo_archive.py
Packages the complete research software, manuscript, replication package,
and verification assets into a release archive for Zenodo DOI minting.
"""

import os
import zipfile
import hashlib
import json

ARCHIVE_NAME = "ML_Inventory_Decision_Support_Zenodo_Release.zip"

CORE_FILES_TO_PACKAGE = [
    # Manuscripts & Documents
    "ML_Inventory_Decision_Support_Paper.pdf",
    "ML_Inventory_Decision_Support_Paper.docx",
    "Supplementary_Materials.docx",
    "Journal_Submission_Cover_Letter.docx",
    "EXECUTIVE_SUMMARY_HANDOUT.docx",
    "Research_Presentation_Deck.pptx",
    "Academic_Conference_Poster.pptx",
    "DEFENSE_SPEECH_MAPPING.md",
    "DEFENSE_PRESENTATION_SCRIPT.md",
    "PEER_REVIEW_RESPONSE_TEMPLATE.docx",
    "Reviewer_Defense_Dossier.md",
    "HIGHLIGHTS.md",
    "AUDIT_CERTIFICATE.md",
    "REPLICATION_PACKAGE.md",
    "CITATION.cff",
    "ZENODO_METADATA.json",
    # Tools & Dashboards
    "interactive_dashboard.html",
    "validate_submission_package.py",
    "simulate_oral_defense_panel.py",
    "run_all.ps1",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt" if os.path.exists("requirements.txt") else None,
]

CORE_DIRS_TO_PACKAGE = [
    "src",
    "tests",
    "paper_figures",
    "latex",
]

def main():
    print("=" * 70)
    print("   PACKAGING ZENODO OPEN-ACCESS RESEARCH RELEASE ARCHIVE")
    print("=" * 70)

    packaged_count = 0

    with zipfile.ZipFile(ARCHIVE_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Package core files
        for f in CORE_FILES_TO_PACKAGE:
            if f and os.path.exists(f):
                z.write(f, f)
                packaged_count += 1
                print(f" [FILE] Added {f}")
            elif f:
                print(f" [SKIP] {f} not found")

        # 2. Package core directories
        for d in CORE_DIRS_TO_PACKAGE:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for file in files:
                        full_path = os.path.join(root, file)
                        # Skip pycache and temp files
                        if "__pycache__" in full_path or file.endswith(".pyc"):
                            continue
                        rel_path = os.path.relpath(full_path, ".")
                        z.write(full_path, rel_path)
                        packaged_count += 1
                print(f" [DIR]  Added directory tree: {d}/")

    archive_size = os.path.getsize(ARCHIVE_NAME)
    with open(ARCHIVE_NAME, "rb") as fp:
        sha256 = hashlib.sha256(fp.read()).hexdigest()

    print("-" * 70)
    print(f"SUCCESS: Created Zenodo Archive -> {ARCHIVE_NAME}")
    print(f"Total Packaged Items : {packaged_count}")
    print(f"Archive Size         : {archive_size:,} bytes ({archive_size / (1024*1024):.2f} MB)")
    print(f"SHA-256 Digest       : {sha256}")
    print("=" * 70)

if __name__ == "__main__":
    main()
