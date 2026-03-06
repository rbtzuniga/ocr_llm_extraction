"""
Grain Elevator Pipeline
=======================
Unified pipeline: OCR → Extraction → CSV

Usage (CLI):
    python run_pipeline.py --images-dir scans/2013 --output-dir outputs --year 2013

Usage (function):
    from run_pipeline import run_pipeline
    run_pipeline(
        images_dir="scans/2013",
        output_dir="outputs",
        year="2013",
        do_ocr=True,
        do_extraction=True,
        do_csv=True,
    )
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Step 1 — OCR
# ---------------------------------------------------------------------------

def run_ocr(images_dir: Path, ocr_csv_path: Path) -> None:
    """OCR all images in *images_dir* and write results to *ocr_csv_path*."""
    import scan_tools as st  # lazy import — only needed when OCR is requested

    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
    if not images:
        log.warning("No images found in %s", images_dir)
        return

    # Resume support: load already-processed names
    processed: set[str] = set()
    img_names, texts, statuses, errors = [], [], [], []
    if ocr_csv_path.exists():
        existing = pd.read_csv(ocr_csv_path, dtype=str).fillna("")
        for _, row in existing.iterrows():
            img_names.append(row["img_name"])
            texts.append(row["text"])
            statuses.append(row["status"])
            errors.append(row["error"])
            processed.add(row["img_name"])
        log.info("Resuming OCR — %d already processed", len(processed))

    remaining = [img for img in images if img.name not in processed]
    log.info("OCR: %d images to process (%d skipped)", len(remaining), len(processed))

    for i, img in enumerate(tqdm(remaining, desc="OCR"), 1):
        try:
            # Rename .tif → .tiff (Document AI requirement)
            if img.suffix.lower() == ".tif":
                img_tiff = img.with_suffix(".tiff")
                img.rename(img_tiff)
                img = img_tiff

            text = st.ocr_document(str(img))
            img_names.append(img.name)
            texts.append(text)
            statuses.append("success")
            errors.append("")
        except Exception as exc:
            img_names.append(img.name)
            texts.append("")
            statuses.append("error")
            errors.append(str(exc))

        # Checkpoint every 50 images
        if i % 50 == 0:
            _save_ocr_csv(ocr_csv_path, img_names, texts, statuses, errors)
            log.info("Checkpoint at %d images", i)

    _save_ocr_csv(ocr_csv_path, img_names, texts, statuses, errors)
    success = statuses.count("success")
    log.info("OCR done — %d success, %d errors", success, len(statuses) - success)


def _save_ocr_csv(path, img_names, texts, statuses, errors):
    df = pd.DataFrame({
        "img_name": img_names,
        "text": texts,
        "status": statuses,
        "error": errors,
    })
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Step 2 — Extraction  (Claude)
# ---------------------------------------------------------------------------

def run_extraction(ocr_csv_path: Path, jsonl_path: Path, model: str, backend: str) -> None:
    """Extract structured elevator entries from OCR CSV using Claude or Ollama."""
    from extract_elevators import process_csv
    log.info("Extraction: %s → %s (backend=%s)", ocr_csv_path, jsonl_path, backend)
    process_csv(str(ocr_csv_path), str(jsonl_path), model=model, backend=backend)


# ---------------------------------------------------------------------------
# Step 3 — JSONL → CSV
# ---------------------------------------------------------------------------

def run_csv_conversion(jsonl_path: Path, csv_path: Path) -> None:
    """Flatten extraction JSONL into a tabular CSV."""
    from extract_elevators import jsonl_to_csv
    log.info("CSV conversion: %s → %s", jsonl_path, csv_path)
    jsonl_to_csv(str(jsonl_path), str(csv_path))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    images_dir: str | None = None,
    output_dir: str = "outputs",
    year: str = "2013",
    do_ocr: bool = True,
    do_extraction: bool = True,
    do_csv: bool = True,
    model: str = "claude-sonnet-4-6",
    backend: str = "claude",
) -> None:
    """Run the full or partial pipeline.

    Parameters
    ----------
    images_dir : str | None
        Directory with scanned page images. Required when *do_ocr* is True.
    output_dir : str
        Directory for all outputs (OCR CSV, JSONL, final CSV).
    year : str
        Label used for naming output files, e.g. "2013".
    do_ocr : bool
        Step 1 — Run Google Document AI OCR on images.
    do_extraction : bool
        Step 2 — Extract structured data from OCR text via Claude.
    do_csv : bool
        Step 3 — Convert extraction JSONL to a flat CSV.
    model : str
        Model to use for extraction (e.g., "claude-sonnet-4-6" or "llama3.1:8b").
    backend : str
        Backend to use: "claude" or "ollama".
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ocr_csv = out / f"{year}_ocr.csv"
    jsonl = out / f"{year}.jsonl"
    csv = out / f"{year}.csv"

    if do_ocr:
        if images_dir is None:
            raise ValueError("images_dir is required when do_ocr=True")
        img_dir = Path(images_dir)
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {img_dir}")
        run_ocr(img_dir, ocr_csv)

    if do_extraction:
        if not ocr_csv.exists():
            raise FileNotFoundError(
                f"OCR CSV not found: {ocr_csv}  — run with do_ocr=True first"
            )
        run_extraction(ocr_csv, jsonl, model=model, backend=backend)

    if do_csv:
        if not jsonl.exists():
            raise FileNotFoundError(
                f"JSONL not found: {jsonl}  — run with do_extraction=True first"
            )
        run_csv_conversion(jsonl, csv)

    log.info("Pipeline finished.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Grain elevator pipeline: OCR → Extraction → CSV"
    )
    p.add_argument("--images-dir", default=None, help="Directory with scanned images (required for OCR step)")
    p.add_argument("--output-dir", default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--year", default="2013", help="Year label for output filenames")
    p.add_argument("--do-ocr", action="store_true", help="Run OCR step")
    p.add_argument("--do-extraction", action="store_true", help="Run Claude extraction step")
    p.add_argument("--do-csv", action="store_true", help="Run JSONL→CSV conversion step")
    p.add_argument("--model", default="claude-sonnet-4-6", help="Claude model for extraction")
    args = p.parse_args()

    # If no step flags are given, run all steps
    if not (args.do_ocr or args.do_extraction or args.do_csv):
        args.do_ocr = True
        args.do_extraction = True
        args.do_csv = True

    run_pipeline(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        year=args.year,
        do_ocr=args.do_ocr,
        do_extraction=args.do_extraction,
        do_csv=args.do_csv,
        model=args.model,
    )


if __name__ == "__main__":
    main()
