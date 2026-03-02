"""Tesseract OCR for line-level printed text."""

from typing import Any, Dict, List

import numpy as np
from PIL import Image


class TesseractOCR:
    """Tesseract OCR with language selection; processes line crops."""

    LANGUAGES = {
        "german": "deu", "english": "eng", "french": "fra", "latin": "lat",
        "italian": "ita", "spanish": "spa", "dutch": "nld", "polish": "pol",
        "russian": "rus", "greek": "grc", "hebrew": "heb", "fraktur": "deu_frak",
    }

    def transcribe_line(self, line_image: Image.Image, languages: List[str] = None) -> Dict[str, Any]:
        if languages:
            lang_codes = [self.LANGUAGES.get(l.lower(), l) for l in languages]
            lang = "+".join(lang_codes)
        else:
            lang = "deu+lat+eng"
        try:
            import pytesseract
            text = pytesseract.image_to_string(line_image, lang=lang).strip()
            data = pytesseract.image_to_data(line_image, lang=lang, output_type=pytesseract.Output.DICT)
            confs = [c for c in data["conf"] if c > 0]
            confidence = float(np.mean(confs) / 100) if confs else 0.5
            return {"status": "success", "text": text, "confidence": confidence}
        except Exception as e:
            return {"status": "error", "error": str(e), "text": "", "confidence": 0}

    def transcribe_lines(
        self, image: Image.Image, lines: List[Dict], languages: List[str] = None
    ) -> Dict[str, Any]:
        results = []
        all_text = []
        total_conf = 0.0
        for idx, line in enumerate(lines):
            bbox = line.get("bbox", {})
            x, y = bbox.get("x", 0), bbox.get("y", 0)
            w, h = bbox.get("width", 0), bbox.get("height", 0)
            if w <= 0 or h <= 0:
                results.append({
                    "line_id": line.get("id", f"line_{idx:03d}"),
                    "bbox": bbox,
                    "polygon": line.get("polygon", []),
                    "text": "",
                    "confidence": 0,
                })
                continue
            line_crop = image.crop((x, y, x + w, y + h))
            result = self.transcribe_line(line_crop, languages)
            results.append({
                "line_id": line.get("id", f"line_{idx:03d}"),
                "bbox": bbox,
                "polygon": line.get("polygon", []),
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0),
            })
            if result.get("text"):
                all_text.append(result["text"])
                total_conf += result.get("confidence", 0)
        return {
            "status": "success",
            "lines": results,
            "text": "\n".join(all_text),
            "confidence": total_conf / len(lines) if lines else 0,
        }
