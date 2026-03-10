"""Region detection and classification using Gemini vision."""

import json
import os
import traceback
from typing import Any, Dict, List

from PIL import Image

from ..utils import MIME_BY_EXT, clean_llm_json


class RegionDetector:
    """Region detection and classification using Gemini vision. Optimized prompt for bboxes."""

    VALID_REGION_TYPES = [
        "TitleRegion", "HeadingRegion", "SubheadingRegion",
        "ParagraphRegion", "TextRegion",
        "HeaderRegion", "FooterRegion", "PageNumberRegion",
        "FootnoteRegion", "MarginaliaRegion", "AnnotationRegion",
        "TableRegion", "ImageRegion", "DiagramRegion",
        "DecorationRegion", "InitialRegion",
        "SignatureRegion", "StampRegion", "CaptionRegion",
        "UnknownRegion",
    ]
    REGION_MARGIN = 0.005

    def __init__(self, client: Any, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def _add_margin(
        self, bbox: Dict, img_width: int, img_height: int, margin_frac: float
    ) -> Dict:
        margin_x = int(img_width * margin_frac)
        margin_y = int(img_height * margin_frac)
        new_x = max(0, bbox["x"] - margin_x)
        new_y = max(0, bbox["y"] - margin_y)
        new_w = min(bbox["width"] + 2 * margin_x, img_width - new_x)
        new_h = min(bbox["height"] + 2 * margin_y, img_height - new_y)
        return {"x": new_x, "y": new_y, "width": new_w, "height": new_h}

    def detect_and_classify(
        self, image_path: str, document_context: Dict = None
    ) -> Dict[str, Any]:
        from google.genai import types

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = MIME_BY_EXT.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        context_lines = []
        if document_context:
            if document_context.get("script_type"):
                context_lines.append(f"Script: {document_context['script_type']}")
            if document_context.get("estimated_period"):
                context_lines.append(f"Period: {document_context['estimated_period']}")
            if document_context.get("primary_language"):
                context_lines.append(f"Language: {document_context['primary_language']}")
            if document_context.get("is_printed") is not None:
                context_lines.append(f"Type: {'Printed' if document_context['is_printed'] else 'Handwritten'}")
        context_str = f"Document info: {', '.join(context_lines)}" if context_lines else ""

        detection_prompt = f"""Detect all text and content regions in this document image.
{context_str}

TASK: Find every distinct content block and return precise bounding boxes.
COORDINATE SYSTEM: Use normalized 0-1000 scale. (0,0)=top-left, (1000,1000)=bottom-right. bbox format: {{"x": left, "y": top, "width": w, "height": h}}

REGION TYPES (use exactly these names): ParagraphRegion, HeadingRegion, TitleRegion, PageNumberRegion, HeaderRegion, FooterRegion, FootnoteRegion, MarginaliaRegion, TableRegion, ImageRegion, DiagramRegion, SignatureRegion, TextRegion.

CRITICAL: SEPARATE PARAGRAPHS. TIGHT BOXES. FULL PAGE SCAN. NO OVERLAPS. READING ORDER = natural sequence.

OUTPUT FORMAT (JSON only):
{{"regions": [{{"id": "r1", "type": "PageNumberRegion", "bbox": {{"x": 920, "y": 30, "width": 60, "height": 35}}, "reading_order": 1}}, ...], "total_regions": N}}
Return the JSON:"""

        response_text = ""
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    detection_prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=8192,
                    thinking_config=types.ThinkingConfig(thinking_level="medium"),
                ),
            )
            response_text = clean_llm_json(response.text)

            result = json.loads(response_text)
            raw_regions = result.get("regions", [])

            validated_regions: List[Dict] = []
            for i, region in enumerate(raw_regions):
                region_type = region.get("type", "TextRegion")
                if region_type not in self.VALID_REGION_TYPES:
                    for valid in self.VALID_REGION_TYPES:
                        if region_type.lower() in valid.lower() or valid.lower() in region_type.lower():
                            region_type = valid
                            break
                    else:
                        region_type = "TextRegion"
                bbox = region.get("bbox", {})
                nx = float(bbox.get("x", 0))
                ny = float(bbox.get("y", 0))
                nw = float(bbox.get("width", 100))
                nh = float(bbox.get("height", 50))
                px = int(nx * width / 1000)
                py = int(ny * height / 1000)
                pw = int(nw * width / 1000)
                ph = int(nh * height / 1000)
                px = max(0, min(px, width - 1))
                py = max(0, min(py, height - 1))
                pw = max(10, min(pw, width - px))
                ph = max(10, min(ph, height - py))
                raw_bbox = {"x": px, "y": py, "width": pw, "height": ph}
                final_bbox = self._add_margin(raw_bbox, width, height, self.REGION_MARGIN)
                validated_regions.append({
                    "id": f"region_{i+1:03d}",
                    "type": region_type,
                    "bbox": final_bbox,
                    "reading_order": int(region.get("reading_order", i + 1)),
                    "confidence": float(region.get("confidence", 0.9)),
                    "description": region.get("description", ""),
                })
            validated_regions.sort(key=lambda r: r["reading_order"])
            for i, r in enumerate(validated_regions):
                r["id"] = f"region_{i+1:03d}"
                r["reading_order"] = i + 1

            return {
                "status": "success",
                "image_path": image_path,
                "image_dimensions": {"width": width, "height": height},
                "regions": validated_regions,
                "total_regions": len(validated_regions),
                "reading_order": [r["id"] for r in validated_regions],
                "region_types_detected": list(set(r["type"] for r in validated_regions)),
            }
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"JSON parse error: {e}", "response_preview": response_text[:500]}
        except Exception as e:
            return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
