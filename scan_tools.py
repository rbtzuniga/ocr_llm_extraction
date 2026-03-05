from __future__ import annotations


from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import io
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
from dotenv import load_dotenv

load_dotenv()

# Configuration — loaded from .env
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
LOCATION = os.environ.get("GCP_LOCATION", "us")
PROCESSOR_ID = os.environ.get("GCP_PROCESSOR_ID", "")

_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
if not _creds:
    _keys_dir = Path("keys")
    _json_files = list(_keys_dir.glob("*.json")) if _keys_dir.is_dir() else []
    if _json_files:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_json_files[0])


# ----------------------------
# Helpers (preprocess)
# ----------------------------

def split_pdf(input_path, output_dir, max_pages=15):
    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for start in range(0, total_pages, max_pages):
        writer = PdfWriter()
        for i in range(start, min(start + max_pages, total_pages)):
            writer.add_page(reader.pages[i])
        out_path = output_dir / f"{input_path.stem}_part{start//max_pages+1}.pdf"
        with open(out_path, "wb") as f:
            writer.write(f)
        print(f"Saved {out_path}")


def ocr_document(file_path: str) -> str:
    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
    
    with open(file_path, "rb") as f:
        content = f.read()
    
    # Detect mime type
    suffix = Path(file_path).suffix.lower()
    mime_types = {".pdf": "application/pdf", ".png": "image/png", 
                  ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".tiff": "image/tiff"}
    mime_type = mime_types.get(suffix, "application/pdf")
    
    raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    
    result = client.process_document(request=request)
    return result.document.text


def _crop_black_border(img_bgr: np.ndarray, pad: int = 20) -> np.ndarray:
    """
    Crops black scanner borders by detecting where the paper edge is.
    Scans inward from each edge looking for the transition from black to paper.
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # The paper is bright (typically > 150), the border is dark (< 50)
    # We scan from each edge looking for a significant jump in brightness
    
    def find_paper_edge_from_left(gray_img, sample_rows=20):
        """Scan from left, return x where paper starts"""
        h, w = gray_img.shape
        # Sample multiple rows to be robust
        step = max(1, h // (sample_rows + 1))
        edges = []
        for row in range(step, h - step, step):
            line = gray_img[row, :]
            # Find first position where brightness exceeds threshold
            # Use a rolling mean to smooth noise
            kernel_size = 10
            if len(line) > kernel_size:
                smoothed = np.convolve(line, np.ones(kernel_size)/kernel_size, mode='valid')
                for i, val in enumerate(smoothed):
                    if val > 100:  # Paper threshold
                        edges.append(i)
                        break
        return int(np.median(edges)) if edges else 0
    
    def find_paper_edge_from_right(gray_img, sample_rows=20):
        """Scan from right, return x where paper ends"""
        h, w = gray_img.shape
        step = max(1, h // (sample_rows + 1))
        edges = []
        for row in range(step, h - step, step):
            line = gray_img[row, ::-1]  # Reverse
            kernel_size = 10
            if len(line) > kernel_size:
                smoothed = np.convolve(line, np.ones(kernel_size)/kernel_size, mode='valid')
                for i, val in enumerate(smoothed):
                    if val > 100:
                        edges.append(w - i)
                        break
        return int(np.median(edges)) if edges else w
    
    def find_paper_edge_from_top(gray_img, sample_cols=20):
        """Scan from top, return y where paper starts"""
        h, w = gray_img.shape
        step = max(1, w // (sample_cols + 1))
        edges = []
        for col in range(step, w - step, step):
            line = gray_img[:, col]
            kernel_size = 10
            if len(line) > kernel_size:
                smoothed = np.convolve(line, np.ones(kernel_size)/kernel_size, mode='valid')
                for i, val in enumerate(smoothed):
                    if val > 100:
                        edges.append(i)
                        break
        return int(np.median(edges)) if edges else 0
    
    def find_paper_edge_from_bottom(gray_img, sample_cols=20):
        """Scan from bottom, return y where paper ends"""
        h, w = gray_img.shape
        step = max(1, w // (sample_cols + 1))
        edges = []
        for col in range(step, w - step, step):
            line = gray_img[::-1, col]  # Reverse
            kernel_size = 10
            if len(line) > kernel_size:
                smoothed = np.convolve(line, np.ones(kernel_size)/kernel_size, mode='valid')
                for i, val in enumerate(smoothed):
                    if val > 100:
                        edges.append(h - i)
                        break
        return int(np.median(edges)) if edges else h
    
    left = find_paper_edge_from_left(gray)
    right = find_paper_edge_from_right(gray)
    top = find_paper_edge_from_top(gray)
    bottom = find_paper_edge_from_bottom(gray)
    
    # Sanity checks - don't crop more than 20% from any side
    max_crop = 0.20
    left = min(left, int(W * max_crop))
    right = max(right, int(W * (1 - max_crop)))
    top = min(top, int(H * max_crop))
    bottom = max(bottom, int(H * (1 - max_crop)))
    
    # Ensure valid crop region
    if right <= left or bottom <= top:
        return img_bgr
    
    # Apply padding (negative to crop more into the paper edge for clean cut)
    x0 = max(0, left - pad)
    y0 = max(0, top - pad)
    x1 = min(W, right + pad)
    y1 = min(H, bottom + pad)
    
    # Final sanity: must be at least 60% of original
    if (x1 - x0) * (y1 - y0) < H * W * 0.6:
        return img_bgr
    
    return img_bgr[y0:y1, x0:x1].copy()


def _deskew(img_bgr: np.ndarray, max_skew_deg: float = 15.0) -> np.ndarray:
    """
    Estimates skew from binarized text pixels and rotates image to reduce it.
    Only corrects small skew angles (up to max_skew_deg) to avoid unintended
    large rotations like 90 degrees.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu binarize: text becomes dark; invert so text is white
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = 255 - bw

    # Remove small noise
    ink = cv2.medianBlur(ink, 3)

    coords = np.column_stack(np.where(ink > 0))
    if coords.shape[0] < 500:
        return img_bgr  # not enough signal

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # OpenCV minAreaRect returns angle in [-90, 0) range
    # We need to convert this to a small skew correction angle
    # The rect angle is measured from the x-axis to the first side of the rect
    # For text that's roughly horizontal, we want a small correction angle
    
    # Normalize angle to range [-45, 45]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    # Only apply correction for small skew angles to avoid large unintended rotations
    if abs(angle) > max_skew_deg:
        return img_bgr  # Angle too large, likely misdetection; skip deskew

    (h, w) = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ----------------------------
# Helpers (Document AI parsing)
# ----------------------------

def _layout_to_text(layout: documentai.Document.Page.Layout, full_text: str) -> str:
    """
    Converts Document AI text anchors to a string (same idea as Google's sample). :contentReference[oaicite:2]{index=2}
    """
    if not layout.text_anchor.text_segments:
        return ""
    parts = []
    for seg in layout.text_anchor.text_segments:
        start = int(seg.start_index) if seg.start_index is not None else 0
        end = int(seg.end_index) if seg.end_index is not None else 0
        if end > start:
            parts.append(full_text[start:end])
    return "".join(parts)


def _poly_to_bbox_norm(poly: documentai.BoundingPoly) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns (x0, y0, x1, y1) in normalized [0,1] coordinates if available,
    else None.
    """
    verts = list(poly.normalized_vertices)
    if not verts:
        return None
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _kmeans_1d(x: np.ndarray, k: int, max_iter: int = 50, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple 1D k-means (no sklearn).
    Returns (labels, centers).
    """
    x = x.astype(np.float64)
    n = x.size
    if k <= 0:
        raise ValueError("k must be >= 1")
    if n == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    k = min(k, n)

    # Initialize centers by quantiles
    qs = np.linspace(0, 1, k + 2)[1:-1]
    centers = np.quantile(x, qs) if k > 1 else np.array([x.mean()])

    for _ in range(max_iter):
        # Assign
        d = np.abs(x[:, None] - centers[None, :])
        labels = d.argmin(axis=1).astype(np.int32)

        new_centers = centers.copy()
        for j in range(k):
            members = x[labels == j]
            if members.size == 0:
                # Re-seed empty cluster at farthest point
                farthest_idx = np.argmax(np.min(d, axis=1))
                new_centers[j] = x[farthest_idx]
            else:
                new_centers[j] = members.mean()

        if np.max(np.abs(new_centers - centers)) < tol:
            centers = new_centers
            break
        centers = new_centers

    return labels, centers


# ----------------------------
# Main function
# ----------------------------

def split_columns_with_docai(
    image_path: str,
    n_cols: int,
    *,
    project_id: str = PROJECT_ID,
    location: str = LOCATION,
    processor_id: str = PROCESSOR_ID,
    mime_type: str = "image/jpeg",
    crop_black_border: bool = True,
    deskew: bool = True,
    header_width_norm: float = 0.75,
    crop_padding_px: int = 12,
    language_hints: Optional[List[str]] = None,
    # payload control
    max_payload_bytes: int = 40 * 1024 * 1024,
) -> Dict[str, Any]:
    """
    Pipeline:
      1) Load image
      2) Crop black border
      3) Deskew
      4) Encode for Document AI under size limit (JPEG + possible downscale)
      5) Document AI OCR
      6) Cluster paragraphs into n_cols by x-center
      7) Return per-column crops + text, plus images for visualization
    """
    if n_cols < 1:
        raise ValueError("n_cols must be >= 1")

    original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = original.copy()
    if crop_black_border:
        img = _crop_black_border(img, pad=10)
    if deskew:
        img = _deskew(img)

    # Ensure request fits under Document AI raw content limit
    content, mime_type, encode_info, img_used = _encode_under_limit(
        img,
        limit_bytes=max_payload_bytes,
        start_quality=90,
        start_max_dim=2600,
    )
    # IMPORTANT: use img_used for downstream cropping/geometry consistency
    img = img_used
    H, W = img.shape[:2]

    # Call Document AI
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project_id, location, processor_id)

    ocr_config = None
    if language_hints:
        ocr_config = documentai.OcrConfig(
            hints=documentai.OcrConfig.Hints(language_hints=language_hints)
        )
    process_options = documentai.ProcessOptions(ocr_config=ocr_config) if ocr_config else None

    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=content, mime_type=mime_type),
        process_options=process_options,
    )
    result = client.process_document(request=request)
    doc = result.document
    full_text = doc.text or ""

    if not doc.pages:
        raise RuntimeError("Document AI returned no pages")
    page0 = doc.pages[0]

    # Collect paragraph items
    items = []
    for p in page0.paragraphs:
        bbox = _poly_to_bbox_norm(p.layout.bounding_poly)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        if (x1 - x0) >= header_width_norm:
            continue
        text = _layout_to_text(p.layout, full_text)
        text = " ".join(text.split())
        items.append({
            "bbox_norm": (x0, y0, x1, y1),
            "x_center": 0.5 * (x0 + x1),
            "y_top": y0,
            "text": text,
        })

    # Fallback to lines if paragraphs are empty
    if not items:
        for ln in page0.lines:
            bbox = _poly_to_bbox_norm(ln.layout.bounding_poly)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            if (x1 - x0) >= header_width_norm:
                continue
            text = _layout_to_text(ln.layout, full_text)
            text = " ".join(text.split())
            items.append({
                "bbox_norm": (x0, y0, x1, y1),
                "x_center": 0.5 * (x0 + x1),
                "y_top": y0,
                "text": text,
            })

    if not items:
        raise RuntimeError("No usable paragraphs/lines found for column splitting")

    # Cluster x-centers into columns
    x = np.array([it["x_center"] for it in items], dtype=np.float64)
    labels, centers = _kmeans_1d(x, k=n_cols)

    # Order clusters left->right
    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    for it, lab in zip(items, labels):
        it["col"] = remap[int(lab)]

    # Build per-column outputs
    columns: List[Dict[str, Any]] = []
    n_used = int(min(n_cols, len(centers)))
    for col in range(n_used):
        col_items = [it for it in items if it["col"] == col]
        if not col_items:
            columns.append({"col_index": col, "bbox_px": None, "crop_bgr": None, "text": "", "n_items": 0})
            continue

        x0 = min(it["bbox_norm"][0] for it in col_items)
        y0 = min(it["bbox_norm"][1] for it in col_items)
        x1 = max(it["bbox_norm"][2] for it in col_items)
        y1 = max(it["bbox_norm"][3] for it in col_items)

        X0 = max(0, int(np.floor(x0 * W)) - crop_padding_px)
        Y0 = max(0, int(np.floor(y0 * H)) - crop_padding_px)
        X1 = min(W, int(np.ceil(x1 * W)) + crop_padding_px)
        Y1 = min(H, int(np.ceil(y1 * H)) + crop_padding_px)

        crop = img[Y0:Y1, X0:X1].copy()

        col_items_sorted = sorted(col_items, key=lambda d: d["y_top"])
        col_text = "\n".join([it["text"] for it in col_items_sorted if it["text"]])

        columns.append({
            "col_index": col,
            "bbox_px": (X0, Y0, X1, Y1),
            "crop_bgr": crop,
            "text": col_text,
            "n_items": len(col_items),
        })

    return {
        "original_bgr": original,
        "preprocessed_bgr": img,          # this is the image actually sent (after any resizing)
        "encode_info": encode_info,        # shows jpeg_quality/max_dim/payload_bytes used to fit under 40MB
        "page_size": (W, H),
        "n_cols_requested": n_cols,
        "n_cols_used": len(columns),
        "columns": columns,
        "debug_items": items,
    }



# ----------------------------
# Payload size fix (40MB limit)
# ----------------------------

def _resize_to_max_dim(img_bgr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img_bgr
    scale = max_dim / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_under_limit(
    img_bgr: np.ndarray,
    *,
    limit_bytes: int = 40 * 1024 * 1024,
    safety_margin_bytes: int = 512 * 1024,
    start_quality: int = 90,
    min_quality: int = 35,
    start_max_dim: int = 2600,
    min_max_dim: int = 1100,
    force_grayscale_if_needed: bool = True,
) -> Tuple[bytes, str, Dict[str, Any], np.ndarray]:
    """
    Encodes to JPEG (and if needed downsizes / reduces quality) to fit under Document AI limit.
    Returns: (content_bytes, mime_type, info_dict, image_used_bgr)
    """
    target = limit_bytes - safety_margin_bytes
    if target <= 0:
        raise ValueError("limit_bytes must be > safety_margin_bytes")

    work = img_bgr.copy()
    q = start_quality
    max_dim = start_max_dim
    used_gray = False

    def try_encode(img: np.ndarray, quality: int) -> Optional[bytes]:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return buf.tobytes()

    # First apply initial resize (helps a lot for huge scans)
    work = _resize_to_max_dim(work, max_dim=max_dim)

    for _ in range(200):
        payload = try_encode(work, q)
        if payload is None:
            raise RuntimeError("Failed to JPEG-encode image")

        if len(payload) <= target:
            return payload, "image/jpeg", {
                "jpeg_quality": q,
                "max_dim": max_dim,
                "used_grayscale": used_gray,
                "payload_bytes": len(payload),
                "limit_bytes": limit_bytes,
            }, work

        # Too big -> reduce quality first
        if q > min_quality:
            q = max(min_quality, q - 5)
            continue

        # If already at low quality, downscale
        if max_dim > min_max_dim:
            max_dim = max(min_max_dim, int(round(max_dim * 0.85)))
            work = _resize_to_max_dim(work, max_dim=max_dim)
            q = start_quality  # reset quality after resizing
            continue

        # Last resort: grayscale to shrink more
        if force_grayscale_if_needed and not used_gray:
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            work = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            used_gray = True
            q = start_quality
            continue

        raise RuntimeError(
            f"Could not fit image under {limit_bytes} bytes even after downscaling/quality reduction. "
            f"Final bytes={len(payload)}."
        )

    raise RuntimeError("Encoding loop did not converge")




# ----------------------------
# Visualization
# ----------------------------

def visualize_original_and_preprocessed(
    original_bgr: np.ndarray,
    preprocessed_bgr: np.ndarray,
    *,
    figsize: Tuple[int, int] = (14, 10),
    titles: Tuple[str, str] = ("Original", "Preprocessed"),
) -> None:
    """Show original and preprocessed images side-by-side."""
    def bgr2rgb(x: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(bgr2rgb(original_bgr))
    ax1.set_title(titles[0])
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(bgr2rgb(preprocessed_bgr))
    ax2.set_title(titles[1])
    ax2.axis("off")

    plt.tight_layout()
    plt.show()