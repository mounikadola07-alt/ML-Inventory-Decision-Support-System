# Makefile for Enterprise Inventory Decision Support System
.PHONY: all test paper slides poster latex audit clean dashboard

all: test paper slides poster latex audit

test:
	python -m pytest tests/ -v

paper:
	python generate_reference_matched_paper.py

slides:
	python build_presentation_deck.py

poster:
	python build_academic_poster.py

latex:
	python build_latex_package.py
	python -c "import zipfile, os; z = 'latex_submission_package.zip'; zipf = zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED); [zipf.write(os.path.join(r, f), os.path.relpath(os.path.join(r, f), 'latex')) for r, d, files in os.walk('latex') for f in files]; zipf.close()"

audit:
	python verify_submission_compliance.py

dashboard:
	python src/interactive_dashboard.py
