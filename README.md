# Grain Elevator Directory Extractor

Extracts structured data from scanned grain elevator directory pages using **Google Document AI** (OCR) and **Claude** (structured extraction).

## Pipeline

| Step | Flag | What it does |
|------|------|-------------|
| **1 — OCR** | `do_ocr` | Sends scanned page images to Google Document AI and saves raw text to a CSV. |
| **2 — Extraction** | `do_extraction` | Sends OCR text to Claude to extract structured elevator entries (JSONL). |
| **3 — CSV** | `do_csv` | Flattens the JSONL into a tabular CSV. |

## Setup

```bash
pip install -r requirements.txt
```

### API keys

1. **Copy the example env file** and fill in your keys:
   ```bash
   cp .env.example .env
   ```
2. **Anthropic** — set `ANTHROPIC_API_KEY` in `.env`.
3. **Google Cloud Document AI** — set the `GCP_*` variables in `.env` and place your service-account JSON file in the `keys/` folder. The credential file is auto-detected from `keys/*.json`.

> Both `.env` and `keys/` are git-ignored and will not be committed.

## Usage

### CLI

```bash
# Run the full pipeline (OCR → Extraction → CSV)
python run_pipeline.py --images-dir scans/2013 --output-dir outputs --year 2013

# Run only specific steps
python run_pipeline.py --output-dir outputs --year 2013 --do-extraction --do-csv

# OCR only
python run_pipeline.py --images-dir scans/2013 --output-dir outputs --year 2013 --do-ocr
```

### Python

```python
from run_pipeline import run_pipeline

run_pipeline(
    images_dir="scans/2013",
    output_dir="outputs",
    year="2013",
    do_ocr=True,
    do_extraction=True,
    do_csv=True,
)
```

### Standalone subcommands (extract_elevators.py)

```bash
# Extract from an existing OCR CSV
python extract_elevators.py extract inputs/2013.csv outputs/2013.jsonl

# Convert JSONL to CSV
python extract_elevators.py tocsv outputs/2013.jsonl
```

### Jupyter Notebook

Open **[example.ipynb](example.ipynb)** for an interactive walkthrough. The notebook lets you:

1. Configure paths and select which pipeline steps to run (OCR, Extraction, CSV)
2. Execute the pipeline
3. Inspect results in a DataFrame

## Output files

For a run with `--year 2013`, the `outputs/` folder will contain:

| File | Description |
|------|-------------|
| `2013_ocr.csv` | Raw OCR text per image |
| `2013.jsonl` | Structured elevator entries |
| `2013.errors.jsonl` | Entries that failed validation |
| `2013.csv` | Final flat CSV |

## Project structure

```
run_pipeline.py          # Main entry point — orchestrates all steps
extract_elevators.py     # Claude extraction + Pydantic validation
scan_tools.py            # Google Document AI OCR + image preprocessing
example.ipynb            # Interactive notebook example
requirements.txt
.env.example             # Template for API keys
```
