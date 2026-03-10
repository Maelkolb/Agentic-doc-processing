"""Shared utilities: MIME mapping, LLM JSON cleaning, skew angle detection."""

import numpy as np

# Shared MIME type mapping for image file extensions
MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def clean_llm_json(text: str) -> str:
    """Strip markdown code fences and extract JSON from LLM responses."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
    return text


def detect_skew_angle(
    gray: np.ndarray,
    *,
    canny_low: int = 50,
    canny_high: int = 150,
    min_line_length: int = 0,
    max_angle: float = 45.0,
) -> float:
    """Detect document skew angle using HoughLinesP on Canny edges.

    Args:
        gray: Grayscale image as numpy array.
        canny_low: Lower Canny threshold.
        canny_high: Upper Canny threshold.
        min_line_length: Minimum line length for HoughLinesP. If 0, defaults to width // 10.
        max_angle: Maximum absolute angle to include in median calculation.

    Returns:
        Median skew angle in degrees, or 0.0 if no lines detected.
    """
    import cv2

    height, width = gray.shape[:2]
    if min_line_length <= 0:
        min_line_length = width // 10

    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=min_line_length, maxLineGap=20)

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < max_angle:
                angles.append(angle)

    return float(np.median(angles)) if angles else 0.0
