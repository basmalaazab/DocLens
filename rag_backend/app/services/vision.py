import logging
from pathlib import Path

import cv2
import pytesseract
from ultralytics import YOLO

from app.core.config import settings
from app.services.generation import build_prompt
from app.services.retrieval import retrieve_documents

logger = logging.getLogger(__name__)

# DocLayNet labels that contain text which is useful to OCR.
TEXT_LABELS = {
    "Text",
    "Title",
    "Caption",
    "List-item",
    "Section-header",
    "Footnote",
    "Table",
    "Formula",
}

_yolo_model = None
_ocr_engine = None


def init_vision_model() -> None:
    """Load the YOLO document-layout model once at application startup."""
    global _yolo_model

    model_path = Path(settings.yolo_model_path)
    if not model_path.exists():
        try:
            from huggingface_hub import hf_hub_download

            logger.info(
                "YOLO model not found locally. Downloading %s from Hugging Face...",
                settings.yolo_model_repo,
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=settings.yolo_model_repo,
                filename=settings.yolo_model_filename,
                local_dir=str(model_path.parent),
            )
            model_path = Path(downloaded)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load YOLO model. Put '{settings.yolo_model_filename}' "
                f"at {settings.yolo_model_path} or allow the app to download it."
            ) from exc

    _yolo_model = YOLO(str(model_path))
    logger.info("YOLO model loaded from %s", model_path)


def _init_ocr() -> None:
    global _ocr_engine
    try:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
        logger.info("RapidOCR initialized")
    except Exception as exc:
        _ocr_engine = None
        logger.warning("RapidOCR unavailable; pytesseract will be used: %s", exc)


def detect_regions(image_path: str, conf: float | None = None) -> list[dict]:
    if _yolo_model is None:
        raise RuntimeError("Vision model not initialised")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    confidence_threshold = conf if conf is not None else settings.yolo_confidence
    result = _yolo_model(image_path, conf=confidence_threshold, verbose=False)[0]

    regions = []
    height, width = image.shape[:2]
    for box in result.boxes:
        label = result.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        crop = image[y1:y2, x1:x2]
        regions.append(
            {
                "label": label,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "crop": crop,
            }
        )
    return regions


def _rapidocr_text(crop) -> str:
    if _ocr_engine is None:
        return ""
    try:
        result, _ = _ocr_engine(crop)
        if not result:
            return ""

        # Reading order: top-to-bottom, then left-to-right.
        items = []
        for box in result:
            text = str(box[1]).strip()
            if not text:
                continue
            points = box[0]
            x_center = sum(point[0] for point in points) / len(points)
            y_center = sum(point[1] for point in points) / len(points)
            items.append((y_center, x_center, text))

        items.sort(key=lambda item: (round(item[0] / 25), item[1]))
        return "\n".join(item[2] for item in items)
    except Exception as exc:
        logger.warning("RapidOCR extraction failed: %s", exc)
        return ""


def _tesseract_text(crop) -> str:
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )
        return pytesseract.image_to_string(processed).strip()
    except Exception as exc:
        logger.warning("pytesseract extraction failed: %s", exc)
        return ""


def extract_text_from_regions(regions: list[dict]) -> str:
    """OCR text-bearing regions; RapidOCR first, pytesseract as fallback."""
    if _ocr_engine is None:
        _init_ocr()

    text_parts = []
    for region in regions:
        if region["label"] not in TEXT_LABELS or region["crop"].size == 0:
            continue

        extracted = _rapidocr_text(region["crop"])
        if not extracted:
            extracted = _tesseract_text(region["crop"])
        if extracted:
            text_parts.append(extracted)

    return "\n".join(text_parts)


def classify_content_type(regions: list[dict]) -> str:
    labels = {region["label"] for region in regions}
    if "Table" in labels:
        return "table"
    if "Formula" in labels:
        return "formula"
    if "Picture" in labels:
        return "diagram"
    if labels & TEXT_LABELS:
        return "text"
    return "unknown"


def describe_image(image_path: str) -> tuple[str, str, str]:
    regions = detect_regions(image_path)
    content_type = classify_content_type(regions)
    ocr_text = extract_text_from_regions(regions)

    # If layout detection misses the page or OCR returns nothing, inspect the
    # complete image rather than silently returning an empty result.
    if not ocr_text.strip():
        image = cv2.imread(image_path)
        if image is not None and image.size:
            fallback_region = [{
                "label": "Text",
                "confidence": 1.0,
                "box": (0, 0, image.shape[1], image.shape[0]),
                "crop": image,
            }]
            ocr_text = extract_text_from_regions(fallback_region)

    labels = sorted({region["label"] for region in regions})
    description_parts = [f"Image content type: {content_type}."]
    if labels:
        description_parts.append(f"Detected elements: {', '.join(labels)}.")
    if ocr_text:
        description_parts.append(f"Text extracted from image (OCR):\n{ocr_text[:3000]}")

    return "\n".join(description_parts), content_type, ocr_text


def query_with_image(question: str, image_path: str, n_results: int | None = None) -> dict:
    """Answer using the user's question plus factual OCR text from the image."""
    import ollama

    image_description, content_type, ocr_text = describe_image(image_path)

    # Retrieval should prioritize the actual question and extracted text, not
    # descriptive metadata such as "Detected elements: Table".
    search_query = f"{question}\n{ocr_text}".strip() if ocr_text else question
    retrieved_docs = retrieve_documents(search_query, n_results)

    # Make OCR available directly to the LLM, because it may contain details
    # that are not present in the indexed PDFs.
    if ocr_text.strip():
        retrieved_docs = [
            {
                "text": ocr_text,
                "source": "Uploaded Image (OCR)",
                "page": 1,
            },
            *retrieved_docs,
        ]

    prompt = build_prompt(
        question=(
            f"{question}\n\nThe user also attached an image. "
            f"Image analysis metadata:\n{image_description}"
        ),
        retrieved_docs=retrieved_docs,
    )

    client = ollama.Client(host=settings.ollama_host)
    response = client.chat(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response["message"]["content"],
        "content_type": content_type,
        "detected_elements": image_description,
        "sources": retrieved_docs,
    }
