"""Document assessment: CV quality metrics + Gemini vision content analysis."""

import json
import os
from typing import Any, Dict

import cv2
import numpy as np

# MIME type by extension for Gemini image parts (must match actual bytes)
_MIME_BY_EXT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}


class DocumentAssessor:
    """Initial document assessment using computer vision + LLM analysis."""

    def __init__(self, llm_client: Any, model_id: str) -> None:
        self.client = llm_client
        self.model_id = model_id

    def _analyze_image_quality(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        estimated_dpi = max(width / 8.5, height / 11) if width < height else max(height / 8.5, width / 11)
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 128.0
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = min(laplacian.var() / 500, 1.0)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.mean(np.abs(gray.astype(float) - blur.astype(float))) / 255.0
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=width // 10, maxLineGap=20)
        skew_angle = 0.0
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(angle) < 45:
                        angles.append(angle)
            if angles:
                skew_angle = float(np.median(angles))
        return {
            "dimensions": {"width": width, "height": height},
            "estimated_dpi": round(estimated_dpi),
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "sharpness": round(sharpness, 3),
            "noise_level": round(noise, 4),
            "skew_angle": round(skew_angle, 2),
        }

    def _analyze_content(self, image_path: str) -> Dict[str, Any]:
        from google.genai import types

        with open(image_path, "rb") as f:
            img_bytes = f.read()
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = _MIME_BY_EXT.get(ext, "image/jpeg")
        analysis_prompt = """Analyze this document image and provide:
1. **Script Type**: Identify the writing system (e.g., Latin, Fraktur, Kurrent, Sütterlin, Greek, Cyrillic, Hebrew, Arabic, etc.)
2. **Estimated Period**: Approximate date/era
3. **Primary Language**: Main language of the text
4. **Document Type**: What kind of document (letter, book page, legal document, newspaper, etc.)
5. **Is Printed**: Is this printed text or handwritten?
6. **Layout Features**: Has tables? Marginalia? Footnotes? Multiple columns? Decorative elements?
7. **Layout Complexity**: simple/moderate/complex
8. **Text Density**: sparse/normal/dense
9. **Condition Notes**: Any damage, staining, fading?

Return as JSON:
{"script_type": "...", "estimated_period": "...", "primary_language": "...", "document_type": "...",
 "is_printed": true/false, "has_tables": true/false, "has_marginalia": true/false, "has_footnotes": true/false,
 "has_multiple_columns": true/false, "has_decorative_elements": true/false,
 "layout_complexity": "simple/moderate/complex", "text_density": "sparse/normal/dense", "condition_notes": "..."}
Return ONLY valid JSON, no markdown."""
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    analysis_prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            return json.loads(response_text)
        except Exception as e:
            print(f"Content analysis error: {e}")
            return {
                "script_type": "unknown",
                "estimated_period": "unknown",
                "primary_language": "unknown",
                "document_type": "unknown",
                "is_printed": True,
                "layout_complexity": "moderate",
                "error": str(e),
            }

    def assess(self, image_path: str) -> Dict[str, Any]:
        quality_metrics = self._analyze_image_quality(image_path)
        content_analysis = self._analyze_content(image_path)
        preprocessors = []
        if abs(quality_metrics["skew_angle"]) > 0.5:
            preprocessors.append("deskew")
        if quality_metrics["noise_level"] > 0.03:
            preprocessors.append("denoise")
        if quality_metrics["contrast"] < 0.3:
            preprocessors.append("enhance_contrast")
        if quality_metrics["brightness"] < 0.3 or quality_metrics["brightness"] > 0.85:
            preprocessors.append("enhance_contrast")
        script = content_analysis.get("script_type", "Latin")
        is_printed = content_analysis.get("is_printed", True)
        if script in ["Fraktur", "Kurrent", "Sütterlin", "Antiqua"] or not is_printed:
            rec_tool = "llm_transcriber"
        elif is_printed and quality_metrics["sharpness"] > 0.4:
            rec_tool = "tesseract"
        else:
            rec_tool = "trocr"
        recommendations = {
            "needs_preprocessing": len(preprocessors) > 0,
            "preprocessors": preprocessors,
            "recognition_tool": rec_tool,
            "use_llm_refinement": content_analysis.get("script_type") in ["Fraktur", "Kurrent", "Sütterlin"],
        }
        return {
            "status": "success",
            "image_path": image_path,
            "quality_metrics": quality_metrics,
            "content_analysis": content_analysis,
            "recommendations": recommendations,
        }
