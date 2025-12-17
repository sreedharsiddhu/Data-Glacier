# Analysis and Submission Report Generator

This repository contains a script to analyze a dataset (Excel) located in the same directory and produce a Markdown report and plots. If Pandoc is installed the script can also convert the generated Markdown to PDF.

Files added:
- `analyze_and_report.py` — main analysis script (looks for the first Excel file in the script folder).
- `requirements.txt` — Python package requirements.
- `submission_template.md` — blank submission template (already present).

Usage (Windows / PowerShell):

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Place your Excel dataset in this folder (the script will pick the first `.xlsx` or `.xls` file).

3. Run the script:

```powershell
python analyze_and_report.py
```

4. Outputs are written to the `report_output` folder. If `pandoc` is installed, the script will attempt to produce `submission.pdf` inside that folder. If not, convert the `report.md` with:

```powershell
pandoc report_output\report.md -o report_output\submission.pdf
```

If you want, tell me the Team Member details and any problem description text you want included; I can update the generated Markdown and run the Pandoc conversion for you.
