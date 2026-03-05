"""
Grain Elevator Directory Data Extraction
Uses Claude API to extract structured fields from OCR text.

Usage:
    python extract_elevators.py <input_csv> <output_jsonl>

Requirements:
    pip install anthropic pydantic pandas tqdm
"""

import json
import re
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
import anthropic

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extraction.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class Representative(BaseModel):
    position: str
    name: str

class CapacityEntry(BaseModel):
    amount: int
    type: str  # "up" (upright) or "fl" (flat)

class ElevatorEntry(BaseModel):
    img_name: str
    text: str
    elevator_type: Optional[str] = Field(
        None,
        description="Single uppercase letter: P, T, R, S, or F. Always the first token in text.",
    )
    company: Optional[str] = None
    annotations: list[str] = Field(
        default_factory=list,
        description="Parenthetical notes after company name (not mailing address).",
    )
    mailing_address: Optional[str] = Field(
        None,
        description="Address in parentheses preceded by 'mailing address'.",
    )
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    phone: list[str] = Field(default_factory=list)
    fax: list[str] = Field(default_factory=list)
    email: list[str] = Field(default_factory=list)
    internet: list[str] = Field(default_factory=list)
    representatives: list[Representative] = Field(
        default_factory=list,
        description="Pairs of (position, name) for people listed with a title.",
    )
    cap: list[CapacityEntry] = Field(
        default_factory=list,
        description="Elevator capacity entries, e.g. '189,718 bus up' → {amount:189718, type:'up'}.",
    )
    ld_cap: Optional[int] = Field(
        None,
        description="Loading capacity integer only, e.g. '35,000 bph' → 35000.",
    )
    li: list[str] = Field(
        default_factory=list,
        description="License types: 'Fed', 'St', or both.",
    )
    rec: list[str] = Field(
        default_factory=list,
        description="Receiving modes from: Truck, Barge, Rail.",
    )
    ld: list[str] = Field(
        default_factory=list,
        description="Loading modes from: Truck, Barge, Rail.",
    )
    svc: list[str] = Field(
        default_factory=list,
        description="Services from: Automatic sampling, Drying, Storage, Cleaning, Scalping.",
    )
    rr: list[str] = Field(
        default_factory=list,
        description="Railroad services from: BNSF, CN, CP, CCP, CSX, FNM, GTW, KCS, NS, TM, UP, other.",
    )
    swc: list[str] = Field(
        default_factory=list,
        description="Switching carriers from: BNSF, CN, CP, CCP, CSX, FNM, GTW, KCS, NS, TM, UP, other.",
    )


SCHEMA_JSON = json.dumps(ElevatorEntry.model_json_schema(), indent=2)

ELEVATOR_TYPES = {"P", "T", "R", "S", "F"}


def parse_img_name(img_name: str) -> tuple[str, int, int]:
    """Parse 'year_page_entry.png' → (year, page, entry)."""
    stem = img_name.replace(".png", "").replace(".jpg", "")
    parts = stem.split("_")
    # Format: year_page_entry  (e.g. "2013_10_0")
    year = parts[0]
    page = int(parts[1])
    entry = int(parts[2])
    return year, page, entry


def starts_with_elevator_type(text: str) -> bool:
    """Return True if text begins with a valid elevator type letter."""
    stripped = text.strip()
    if not stripped:
        return False
    # First token must be a single letter in ELEVATOR_TYPES
    first_token = stripped.split()[0] if stripped.split() else ""
    return first_token in ELEVATOR_TYPES


def is_consecutive(prev_name: str, curr_name: str) -> bool:
    """Return True if curr_name is the image immediately following prev_name.

    Consecutive means:
      - same year & page, entry incremented by 1
      - OR same year, page incremented by 1, entry restarted to 0
    """
    try:
        y1, p1, e1 = parse_img_name(prev_name)
        y2, p2, e2 = parse_img_name(curr_name)
    except (ValueError, IndexError):
        return False
    if y1 != y2:
        return False
    # Same page, next entry
    if p1 == p2 and e2 == e1 + 1:
        return True
    # Next page, entry restarted to 0
    if p2 == p1 + 1 and e2 == 0:
        return True
    return False


def preprocess_entries(
    ok: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Detect continuation rows (text without elevator-type prefix) and merge
    them with the previous entry.  Return (rows_to_process, flagged_errors).

    Each item in rows_to_process is a dict with keys: img_name, text.
    For merged rows, img_name lists both names (e.g. "2013_10_5+2013_10_6")
    and text is the concatenation.

    flagged_errors are rows that lack an elevator type but aren't consecutive
    with the previous entry.
    """
    rows = []
    for _, r in ok.iterrows():
        # Strip leading non-letter chars (spaces, newlines, hyphens, etc.)
        # so OCR artifacts don't cause a valid entry to look like a continuation.
        cleaned_text = re.sub(r'^[^a-zA-Z]+', '', r["text"].strip())
        rows.append({
            "img_name": r["img_name"].strip(),
            "text": cleaned_text,
        })

    # Sort by parsed (year, page, entry) so ordering is reliable
    def sort_key(row):
        try:
            y, p, e = parse_img_name(row["img_name"])
            return (y, p, e)
        except (ValueError, IndexError):
            return ("", 0, 0)

    rows.sort(key=sort_key)

    merged: list[dict] = []
    errors: list[dict] = []

    i = 0
    while i < len(rows):
        row = rows[i]
        if starts_with_elevator_type(row["text"]):
            # Normal entry — but check if the NEXT row is a continuation
            combined_img = row["img_name"]
            combined_text = row["text"]
            while i + 1 < len(rows):
                nxt = rows[i + 1]
                if not starts_with_elevator_type(nxt["text"]) and is_consecutive(
                    rows[i]["img_name"], nxt["img_name"]
                ):
                    log.info(
                        "Merging continuation %s into %s",
                        nxt["img_name"],
                        combined_img,
                    )
                    combined_img += "+" + nxt["img_name"]
                    combined_text += "\n" + nxt["text"]
                    i += 1
                else:
                    break
            merged.append({"img_name": combined_img, "text": combined_text})
        else:
            # Doesn't start with elevator type and isn't following a normal row
            # (already consumed above) — flag as error
            errors.append({
                "img_name": row["img_name"],
                "text": row["text"],
                "error": "no_elevator_type",
                "detail": (
                    "Text does not start with an elevator type (P/T/R/S/F) "
                    "and is not a continuation of the previous entry."
                ),
            })
            log.warning("Flagging %s — no elevator type and not a continuation", row["img_name"])
        i += 1

    return merged, errors


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a precise data extractor for grain elevator directory entries.
Extract structured data from OCR-scanned text into the JSON schema below.
Return ONLY valid JSON — no markdown, no explanation, no code fences.

Schema:
{SCHEMA_JSON}

Extraction rules:
1. elevator_type — always the FIRST character/token in the text (P, T, R, S, or F).
2. company — the name immediately following elevator_type.
3. annotations — parenthetical content after the company name that is NOT a mailing address and NOT the street address. May appear more than once.
4. mailing_address — parenthetical content explicitly preceded by the words "mailing address".
5. address / city / state / zipcode — parse from the street address line.
6. phone / fax — include full formatted number strings. "Fax:" prefix identifies fax.
7. email — any email address present.
8. internet — any URL or www address present.
9. representatives — names preceded by a title keyword (manager, superintendent, supervisor, terminal manager, branch manager, office manager, director, agent, etc.). Store as {{position, name}}.
10. cap — e.g. "189,718 bus up" → {{amount: 189718, type: "up"}}; "fl" for flat. Multiple entries possible.
11. ld_cap — integer only from "X bph".
12. li — normalize to "Fed" and/or "St".
13. rec / ld — normalize entries to: Truck, Barge, Rail.
14. svc — normalize to: Automatic sampling, Drying, Storage, Cleaning, Scalping.
15. rr — railroad service abbreviations.
16. swc — switching carrier abbreviations (labeled "SWC:" in text).
17. img_name and text — copy verbatim from the input.
18. Use null for absent optional fields; empty list [] for absent list fields.
19. Do NOT invent values not present in the text.
"""


def sanitize_text(text: str) -> str:
    """Replace non-ASCII characters with safe equivalents to avoid
    content-filter false positives from OCR artifacts."""
    replacements = {
        "\u25aa": "-",   # ▪ small black square
        "\u2015": "-",   # ― horizontal bar
        "\u2014": "-",   # — em dash
        "\u2013": "-",   # – en dash
        "\u2022": "-",   # • bullet
        "\u25cf": "-",   # ● black circle
        "\u00b7": "-",   # · middle dot
        "\u2019": "'",   # ' right single quotation
        "\u2018": "'",   # ' left single quotation
        "\u201c": '"',   # " left double quotation
        "\u201d": '"',   # " right double quotation
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Drop any remaining non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return text


def make_user_prompt(img_name: str, text: str) -> str:
    return f"img_name: {img_name}\ntext: {text}"


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_claude(
    client: anthropic.Anthropic,
    img_name: str,
    text: str,
    model: str = "claude-sonnet-4-6",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict:
    """Call Claude and return parsed dict, or raise on persistent failure."""
    sanitized = False
    for attempt in range(1, max_retries + 1):
        try:
            prompt_text = sanitize_text(text) if sanitized else text
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": make_user_prompt(img_name, prompt_text)}
                ],
            )
            # Check for empty / content-filtered response
            if (
                not response.content
                or not response.content[0].text.strip()
                or response.stop_reason not in ("end_turn", "stop_sequence")
            ):
                raise RuntimeError(
                    f"Response cleared by content safety filter "
                    f"(stop_reason={response.stop_reason})"
                )

            raw = response.content[0].text.strip()
            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except (json.JSONDecodeError, anthropic.APIError, RuntimeError) as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, img_name, exc)
            # On first failure, switch to sanitized text for subsequent retries
            if not sanitized:
                log.info("Switching to sanitized (ASCII-only) text for %s", img_name)
                sanitized = True
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                raise


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_csv(input_path: str, output_path: str, model: str = "claude-sonnet-4-6") -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    error_path = output_path.with_suffix(".errors.jsonl")

    df = pd.read_csv(input_path, dtype=str).fillna("")

    # Filter to successful OCR rows only
    ok = df[df["status"].str.strip().str.lower() == "success"].copy()
    log.info("Total rows: %d  |  status=success: %d", len(df), len(ok))

    # Determine already-processed img_names to allow resuming
    processed: set[str] = set()
    if output_path.exists():
        # Try utf-8 first; fall back to system default for legacy files
        for enc in ("utf-8", None):
            try:
                with open(output_path, encoding=enc) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            processed.add(rec.get("img_name", ""))
                        except json.JSONDecodeError:
                            pass
                break  # success — stop trying encodings
            except UnicodeDecodeError:
                processed.clear()
    log.info("Already processed: %d rows — will skip these.", len(processed))

    # ------------------------------------------------------------------
    # Pre-process: detect continuations (merge) and orphan rows (flag)
    # ------------------------------------------------------------------
    entries_to_process, flagged_errors = preprocess_entries(ok)
    log.info(
        "After merge: %d entries to process, %d flagged errors",
        len(entries_to_process),
        len(flagged_errors),
    )

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    with open(output_path, "a", encoding="utf-8") as out_f, open(error_path, "a", encoding="utf-8") as err_f:
        # Write out pre-processing errors first
        for ferr in flagged_errors:
            err_f.write(json.dumps(ferr) + "\n")
        if flagged_errors:
            err_f.flush()

        for row in tqdm(entries_to_process, desc="Extracting"):
            img_name = row["img_name"]
            text = row["text"]

            # For merged entries, check if ANY constituent img_name was already processed
            constituent_names = img_name.split("+")
            if any(n in processed for n in constituent_names):
                continue

            try:
                raw_dict = call_claude(client, img_name, text, model=model)

                # Claude may return a list when the text contains multiple entries
                items = raw_dict if isinstance(raw_dict, list) else [raw_dict]
                for item in items:
                    entry = ElevatorEntry.model_validate(item)
                    out_f.write(entry.model_dump_json() + "\n")
                out_f.flush()
                processed.add(img_name)

            except ValidationError as exc:
                log.error("Validation error for %s: %s", img_name, exc)
                err_f.write(json.dumps({
                    "img_name": img_name,
                    "text": text,
                    "error": "validation",
                    "detail": str(exc),
                }) + "\n")
                err_f.flush()

            except Exception as exc:
                log.error("Unexpected error for %s: %s", img_name, exc)
                err_f.write(json.dumps({
                    "img_name": img_name,
                    "text": text,
                    "error": "api_or_parse",
                    "detail": str(exc),
                }) + "\n")
                err_f.flush()

            # Polite rate-limit pause
            time.sleep(0.3)

    log.info("Done. Output: %s  |  Errors: %s", output_path, error_path)


# ---------------------------------------------------------------------------
# JSONL → CSV conversion
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "img_name",
    "elevator_type",
    "company",
    "annotations",
    "mailing_address",
    "address",
    "city",
    "state",
    "zipcode",
    "phone",
    "fax",
    "email",
    "internet",
    "representatives",
    "cap_up",
    "cap_fl",
    "ld_cap",
    "li",
    "rec",
    "ld",
    "svc",
    "rr",
    "swc",
]


def flatten_entry(record: dict) -> dict:
    """Flatten a single JSONL record into a flat dict for CSV output."""
    row: dict = {}

    # Scalar fields — copy as-is
    for key in ("img_name", "elevator_type", "company", "mailing_address",
                "address", "city", "state", "zipcode", "ld_cap"):
        row[key] = record.get(key) or ""

    # List-of-string fields — join with "; "
    for key in ("annotations", "phone", "fax", "email", "internet",
                "li", "rec", "ld", "svc", "rr", "swc"):
        vals = record.get(key, []) or []
        row[key] = "; ".join(str(v) for v in vals)

    # Representatives — "Position: Name; Position: Name"
    reps = record.get("representatives", []) or []
    row["representatives"] = "; ".join(
        f"{r.get('position', '')}: {r.get('name', '')}" for r in reps
    )

    # Capacity — split into cap_up and cap_fl (sum if multiple of same type)
    cap_up = 0
    cap_fl = 0
    for c in record.get("cap", []) or []:
        amt = c.get("amount", 0) or 0
        ctype = c.get("type", "")
        if ctype == "up":
            cap_up += amt
        elif ctype == "fl":
            cap_fl += amt
    row["cap_up"] = cap_up if cap_up else ""
    row["cap_fl"] = cap_fl if cap_fl else ""

    return row


def jsonl_to_csv(jsonl_path: str, csv_path: str) -> None:
    """Read a JSONL file and write a flattened CSV."""
    jsonl_path = Path(jsonl_path)
    csv_path = Path(csv_path)

    records = []
    for enc in ("utf-8", None):
        try:
            with open(jsonl_path, encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            break
        except UnicodeDecodeError:
            records.clear()

    rows = [flatten_entry(r) for r in records]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("Wrote %d rows to %s", len(df), csv_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract grain elevator data using Claude.")
    sub = parser.add_subparsers(dest="command")

    # --- extract subcommand (default behaviour) ---
    p_extract = sub.add_parser("extract", help="Extract elevator data from CSV via Claude API.")
    p_extract.add_argument("input_csv", help="Path to input CSV file")
    p_extract.add_argument("output_jsonl", help="Path to output JSONL file")
    p_extract.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model string (default: claude-sonnet-4-6)",
    )

    # --- tocsv subcommand ---
    p_csv = sub.add_parser("tocsv", help="Convert extraction JSONL to a flat CSV.")
    p_csv.add_argument("input_jsonl", help="Path to input JSONL file")
    p_csv.add_argument("output_csv", nargs="?", default=None,
                       help="Path to output CSV (default: same name with .csv extension)")

    args = parser.parse_args()

    if args.command == "tocsv":
        csv_out = args.output_csv or str(Path(args.input_jsonl).with_suffix(".csv"))
        jsonl_to_csv(args.input_jsonl, csv_out)
    elif args.command == "extract":
        process_csv(args.input_csv, args.output_jsonl, model=args.model)
    else:
        # Backwards-compatible: no subcommand → treat positional args as extract
        parser.print_help()



# Example usage:
"""
python extract_elevators.py extract inputs/2013_10.csv outputs/2013_10.jsonl
python extract_elevators.py tocsv outputs/2013_10.jsonl outputs/2013_10.csv
"""

"""
python extract_elevators.py tocsv outputs/2013_10.jsonl
python extract_elevators.py tocsv outputs/2013_10.jsonl outputs/custom_name.csv
"""