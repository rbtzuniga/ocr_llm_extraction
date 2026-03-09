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
import requests
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
import anthropic

from dotenv import load_dotenv
load_dotenv()

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.1:8b"

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
    text: Optional[str] = Field(
        None,
        description="Original OCR text (omitted when include_text=False to reduce costs).",
    )
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


# Full schema for reference (not pasted in prompt when using structured outputs)
SCHEMA_JSON = json.dumps(ElevatorEntry.model_json_schema(), indent=2)

# Schema for extraction (excludes text field to reduce output tokens)
def get_extraction_schema(include_text: bool = False) -> dict:
    """Return JSON schema for extraction, optionally excluding text field."""
    schema = ElevatorEntry.model_json_schema()
    if not include_text:
        # Remove text from required and properties to save output tokens
        if "required" in schema and "text" in schema["required"]:
            schema["required"].remove("text")
        if "properties" in schema and "text" in schema["properties"]:
            del schema["properties"]["text"]
    return schema

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

# System prompt WITH schema embedded (for Ollama / legacy mode)
SYSTEM_PROMPT_WITH_SCHEMA = f"""You are a precise data extractor for grain elevator directory entries.
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

# System prompt WITHOUT schema (for Claude structured outputs — schema passed separately)
SYSTEM_PROMPT = """You are a precise data extractor for grain elevator directory entries.
Extract structured data from OCR-scanned text following these rules:

Extraction rules:
1. elevator_type — always the FIRST character/token in the text (P, T, R, S, or F).
2. company — the name immediately following elevator_type.
3. annotations — parenthetical content after the company name that is NOT a mailing address and NOT the street address. May appear more than once.
4. mailing_address — parenthetical content explicitly preceded by the words "mailing address".
5. address / city / state / zipcode — parse from the street address line.
6. phone / fax — include full formatted number strings. "Fax:" prefix identifies fax.
7. email — any email address present.
8. internet — any URL or www address present.
9. representatives — names preceded by a title keyword (manager, superintendent, supervisor, terminal manager, branch manager, office manager, director, agent, etc.). Store as {position, name}.
10. cap — e.g. "189,718 bus up" → {amount: 189718, type: "up"}; "fl" for flat. Multiple entries possible.
11. ld_cap — integer only from "X bph".
12. li — normalize to "Fed" and/or "St".
13. rec / ld — normalize entries to: Truck, Barge, Rail.
14. svc — normalize to: Automatic sampling, Drying, Storage, Cleaning, Scalping.
15. rr — railroad service abbreviations.
16. swc — switching carrier abbreviations (labeled "SWC:" in text).
17. img_name — copy verbatim from the input.
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
# Ollama API call
# ---------------------------------------------------------------------------

def call_ollama(
    img_name: str,
    text: str,
    model: str = OLLAMA_DEFAULT_MODEL,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Call Ollama local LLM and return parsed dict, or raise on persistent failure."""
    sanitized = False
    for attempt in range(1, max_retries + 1):
        try:
            prompt_text = sanitize_text(text) if sanitized else text
            # Use schema-embedded prompt for Ollama (no structured outputs support)
            full_prompt = f"{SYSTEM_PROMPT_WITH_SCHEMA}\n\nUser: {make_user_prompt(img_name, prompt_text)}\n\nRespond with ONLY valid JSON:"
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2048,
                    },
                },
                timeout=120,  # 2 minute timeout for slow generations
            )
            response.raise_for_status()
            
            result = response.json()
            raw = result.get("response", "").strip()
            
            if not raw:
                raise RuntimeError("Empty response from Ollama")
            
            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            # Also handle trailing fences
            if "```" in raw:
                raw = raw.split("```")[0]
            
            return json.loads(raw.strip())
            
        except (json.JSONDecodeError, requests.RequestException, RuntimeError) as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, img_name, exc)
            if not sanitized:
                log.info("Switching to sanitized (ASCII-only) text for %s", img_name)
                sanitized = True
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                raise


# ---------------------------------------------------------------------------
# Claude API call with retry, prompt caching, and structured outputs
# ---------------------------------------------------------------------------

def call_claude(
    client: anthropic.Anthropic,
    img_name: str,
    text: str,
    model: str = "claude-sonnet-4-6",
    max_retries: int = 3,
    retry_delay: float = 5.0,
    include_text: bool = False,
    use_structured_output: bool = True,
) -> dict:
    """Call Claude and return parsed dict, or raise on persistent failure.
    
    Args:
        client: Anthropic client instance
        img_name: Image filename for the entry
        text: OCR text to extract from
        model: Claude model to use
        max_retries: Number of retry attempts
        retry_delay: Base delay between retries (multiplied by attempt number)
        include_text: If True, include original text in output (increases costs)
        use_structured_output: If True, use JSON schema response format
    """
    sanitized = False
    extraction_schema = get_extraction_schema(include_text=include_text)
    
    for attempt in range(1, max_retries + 1):
        try:
            prompt_text = sanitize_text(text) if sanitized else text
            
            # Build system message with cache_control for prompt caching
            system_content = [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}  # Cache the system prompt
                }
            ]
            
            # Build request kwargs
            request_kwargs = {
                "model": model,
                "max_tokens": 2048,
                "system": system_content,
                "messages": [
                    {"role": "user", "content": make_user_prompt(img_name, prompt_text)}
                ],
            }
            
            # Add structured output response format if enabled
            if use_structured_output:
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "elevator_entry",
                        "strict": True,
                        "schema": extraction_schema,
                    }
                }
            
            # Add beta header for extended prompt caching
            response = client.beta.messages.create(
                betas=["prompt-caching-2024-07-31"],
                **request_kwargs
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
            # Strip accidental markdown fences (shouldn't happen with structured output)
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
# Batch processing (Anthropic Message Batches API — 50% cost savings)
# ---------------------------------------------------------------------------

def create_batch_requests(
    entries: list[dict],
    model: str,
    include_text: bool = False,
) -> list[dict]:
    """Create a list of batch request objects for the Anthropic Batches API."""
    extraction_schema = get_extraction_schema(include_text=include_text)
    requests = []
    
    for entry in entries:
        img_name = entry["img_name"]
        text = sanitize_text(entry["text"])
        
        request = {
            "custom_id": img_name,
            "params": {
                "model": model,
                "max_tokens": 2048,
                "system": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [
                    {"role": "user", "content": make_user_prompt(img_name, text)}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "elevator_entry",
                        "strict": True,
                        "schema": extraction_schema,
                    }
                }
            }
        }
        requests.append(request)
    
    return requests


def process_batch(
    client: anthropic.Anthropic,
    entries: list[dict],
    model: str,
    include_text: bool = False,
    poll_interval: float = 30.0,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Process entries using Anthropic Message Batches API (50% cost savings).
    
    Returns:
        Tuple of (results_dict, errors_dict) where:
        - results_dict maps img_name to extracted data dict
        - errors_dict maps img_name to error message
    """
    log.info("Creating batch with %d requests...", len(entries))
    
    requests = create_batch_requests(entries, model, include_text)
    
    # Create the batch
    batch = client.beta.messages.batches.create(
        betas=["prompt-caching-2024-07-31"],
        requests=requests
    )
    
    batch_id = batch.id
    log.info("Batch created: %s — polling for completion...", batch_id)
    
    # Poll for completion
    while True:
        batch_status = client.beta.messages.batches.retrieve(batch_id)
        status = batch_status.processing_status
        
        if status == "ended":
            break
        elif status in ("canceling", "canceled"):
            raise RuntimeError(f"Batch {batch_id} was canceled")
        
        log.info(
            "Batch %s: %s (requests: %d/%d completed)",
            batch_id, status,
            batch_status.request_counts.succeeded + batch_status.request_counts.errored,
            batch_status.request_counts.total,
        )
        time.sleep(poll_interval)
    
    # Retrieve results
    log.info("Batch completed. Retrieving results...")
    results = {}
    errors = {}
    
    for result in client.beta.messages.batches.results(batch_id):
        custom_id = result.custom_id
        
        if result.result.type == "succeeded":
            message = result.result.message
            if message.content and message.content[0].text:
                try:
                    raw = message.content[0].text.strip()
                    # Strip markdown fences if present
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    results[custom_id] = json.loads(raw)
                except json.JSONDecodeError as e:
                    errors[custom_id] = f"JSON decode error: {e}"
            else:
                errors[custom_id] = "Empty response content"
        else:
            error_type = result.result.type
            error_msg = getattr(result.result, 'error', {})
            errors[custom_id] = f"{error_type}: {error_msg}"
    
    log.info("Batch results: %d succeeded, %d failed", len(results), len(errors))
    return results, errors


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_csv(
    input_path: str,
    output_path: str,
    model: str = "claude-sonnet-4-6",
    backend: str = "claude",
    batching: bool = False,
    include_text: bool = False,
) -> None:
    """Process CSV with LLM extraction.
    
    Args:
        input_path: Path to OCR CSV
        output_path: Path for output JSONL
        model: Model name to use
        backend: "claude" or "ollama"
        batching: If True, use Anthropic Message Batches API (50% cost savings)
        include_text: If True, include original OCR text in output (increases token usage)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    error_path = output_path.with_suffix(".errors.jsonl")

    df = pd.read_csv(input_path, dtype=str).fillna("")

    # Filter to successful OCR rows only
    ok = df[df["status"].str.strip().str.lower() == "success"].copy()
    log.info("Total rows: %d  |  status=success: %d", len(df), len(ok))
    log.info("Using backend: %s  |  model: %s", backend, model)

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

    # Initialize client based on backend
    client = None
    if backend == "claude":
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    elif backend == "ollama":
        # Verify Ollama is running
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            r.raise_for_status()
            available_models = [m["name"] for m in r.json().get("models", [])]
            if model not in available_models and not any(model in m for m in available_models):
                log.warning("Model '%s' not found. Available: %s", model, available_models)
                log.info("Run: ollama pull %s", model)
        except requests.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                f"Make sure Ollama is running: ollama serve"
            ) from e

    with open(output_path, "a", encoding="utf-8") as out_f, open(error_path, "a", encoding="utf-8") as err_f:
        # Write out pre-processing errors first
        for ferr in flagged_errors:
            err_f.write(json.dumps(ferr) + "\n")
        if flagged_errors:
            err_f.flush()

        # Filter out already-processed entries
        entries_remaining = []
        for row in entries_to_process:
            img_name = row["img_name"]
            constituent_names = img_name.split("+")
            if not any(n in processed for n in constituent_names):
                entries_remaining.append(row)
        
        if not entries_remaining:
            log.info("All entries already processed.")
            return
        
        log.info("Entries to process: %d", len(entries_remaining))

        # ----------------------------------------------------------------
        # Batch processing mode (Claude only, 50% cost savings)
        # ----------------------------------------------------------------
        if batching and backend == "claude":
            log.info("Using batch processing mode (50%% cost savings)...")
            
            # Process in batches of up to 10,000 (API limit)
            batch_size = 10000
            for batch_start in range(0, len(entries_remaining), batch_size):
                batch_entries = entries_remaining[batch_start:batch_start + batch_size]
                log.info(
                    "Processing batch %d-%d of %d",
                    batch_start + 1,
                    batch_start + len(batch_entries),
                    len(entries_remaining)
                )
                
                results, batch_errors = process_batch(
                    client, batch_entries, model, include_text=include_text
                )
                
                # Write successful results
                for img_name, raw_dict in results.items():
                    try:
                        items = raw_dict if isinstance(raw_dict, list) else [raw_dict]
                        for item in items:
                            entry = ElevatorEntry.model_validate(item)
                            out_f.write(entry.model_dump_json() + "\n")
                        out_f.flush()
                        processed.add(img_name)
                    except ValidationError as exc:
                        # Find the original text for error logging
                        original_text = next(
                            (e["text"] for e in batch_entries if e["img_name"] == img_name),
                            ""
                        )
                        log.error("Validation error for %s: %s", img_name, exc)
                        err_f.write(json.dumps({
                            "img_name": img_name,
                            "text": original_text if include_text else "(omitted)",
                            "error": "validation",
                            "detail": str(exc),
                        }) + "\n")
                        err_f.flush()
                
                # Write batch errors
                for img_name, error_msg in batch_errors.items():
                    original_text = next(
                        (e["text"] for e in batch_entries if e["img_name"] == img_name),
                        ""
                    )
                    log.error("Batch error for %s: %s", img_name, error_msg)
                    err_f.write(json.dumps({
                        "img_name": img_name,
                        "text": original_text if include_text else "(omitted)",
                        "error": "batch_api",
                        "detail": error_msg,
                    }) + "\n")
                    err_f.flush()
        else:
            # ----------------------------------------------------------------
            # Sequential processing mode (original behavior)
            # ----------------------------------------------------------------
            if batching and backend != "claude":
                log.warning("Batching is only supported for Claude backend. Using sequential mode.")
            
            for row in tqdm(entries_remaining, desc="Extracting"):
                img_name = row["img_name"]
                text = row["text"]

                try:
                    if backend == "claude":
                        raw_dict = call_claude(
                            client, img_name, text, model=model,
                            include_text=include_text
                        )
                    else:  # ollama
                        raw_dict = call_ollama(img_name, text, model=model)

                    # LLM may return a list when the text contains multiple entries
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
                        "text": text if include_text else "(omitted)",
                        "error": "validation",
                        "detail": str(exc),
                    }) + "\n")
                    err_f.flush()

                except Exception as exc:
                    log.error("Unexpected error for %s: %s", img_name, exc)
                    err_f.write(json.dumps({
                        "img_name": img_name,
                        "text": text if include_text else "(omitted)",
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
    p_extract = sub.add_parser("extract", help="Extract elevator data from CSV via LLM.")
    p_extract.add_argument("input_csv", help="Path to input CSV file")
    p_extract.add_argument("output_jsonl", help="Path to output JSONL file")
    p_extract.add_argument(
        "--backend",
        choices=["claude", "ollama"],
        default="claude",
        help="LLM backend: 'claude' (API) or 'ollama' (local). Default: claude",
    )
    p_extract.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults: claude-sonnet-4-6 (claude), llama3.1:8b (ollama)",
    )
    p_extract.add_argument(
        "--batching",
        action="store_true",
        help="Use Anthropic Message Batches API for 50%% cost savings (Claude only). "
             "Processes all entries in a single batch request.",
    )
    p_extract.add_argument(
        "--include-text",
        action="store_true",
        help="Include original OCR text in output JSON. "
             "Omit to reduce output tokens and costs.",
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
        # Set default model based on backend if not specified
        model = args.model
        if model is None:
            model = OLLAMA_DEFAULT_MODEL if args.backend == "ollama" else "claude-sonnet-4-6"
        process_csv(
            args.input_csv,
            args.output_jsonl,
            model=model,
            backend=args.backend,
            batching=args.batching,
            include_text=args.include_text,
        )
    else:
        # Backwards-compatible: no subcommand → treat positional args as extract
        parser.print_help()



# Example usage:
"""
# Standard extraction (uses structured outputs + prompt caching):
python extract_elevators.py extract inputs/2013_10.csv outputs/2013_10.jsonl

# With batch processing (50% cost savings, Claude only):
python extract_elevators.py extract inputs/2013_10.csv outputs/2013_10.jsonl --batching

# Include original text in output (increases token costs):
python extract_elevators.py extract inputs/2013_10.csv outputs/2013_10.jsonl --include-text

# Full cost-optimized run:
python extract_elevators.py extract inputs/2013_10.csv outputs/2013_10.jsonl --batching

# Convert JSONL to CSV:
python extract_elevators.py tocsv outputs/2013_10.jsonl outputs/2013_10.csv
"""

"""
Cost optimization summary:
1. Structured Outputs: JSON schema passed via response_format (not in prompt)
2. Prompt Caching: System prompt cached with cache_control for reuse
3. Text Omission: Original text excluded from output by default (--include-text to keep)
4. Batch Processing: Use --batching flag for 50% cost savings on large runs
"""