"""TrOCR for handwriting/print recognition; processes line crops."""

from typing import Any, Dict, List

from PIL import Image


class TrOCRHTR:
    """TrOCR for handwriting/print; lazy-loads model by key."""

    AVAILABLE_MODELS = {
        "handwritten": {"model": "microsoft/trocr-base-handwritten", "processor": "microsoft/trocr-base-handwritten"},
        "handwritten-large": {"model": "microsoft/trocr-large-handwritten", "processor": "microsoft/trocr-large-handwritten"},
        "printed": {"model": "microsoft/trocr-base-printed", "processor": "microsoft/trocr-base-printed"},
        "printed-large": {"model": "microsoft/trocr-large-printed", "processor": "microsoft/trocr-large-printed"},
        "kurrent": {"model": "dh-unibe/trocr-kurrent", "processor": "microsoft/trocr-base-handwritten"},
        "kurrent-xvi-xvii": {"model": "dh-unibe/trocr-kurrent-XVI-XVII", "processor": "microsoft/trocr-base-handwritten"},
    }

    def __init__(self) -> None:
        self.processors: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.device = None
        self.failed_models: Dict[str, str] = {}
        self._device_initialized = False

    def _init_device(self) -> None:
        if self._device_initialized:
            return
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device_initialized = True

    def _load_model(self, model_key: str) -> bool:
        if model_key in self.models and model_key in self.processors:
            return True
        if model_key in self.failed_models:
            return False
        if model_key not in self.AVAILABLE_MODELS:
            model_key = "handwritten"
        config = self.AVAILABLE_MODELS[model_key]
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self._init_device()
            processor = TrOCRProcessor.from_pretrained(config["processor"], use_fast=False)
            model = VisionEncoderDecoderModel.from_pretrained(config["model"])
            model = model.to(self.device).eval()
            self.processors[model_key] = processor
            self.models[model_key] = model
            return True
        except Exception as e:
            self.failed_models[model_key] = str(e)
            return False

    def transcribe_line(self, line_image: Image.Image, model: str = "handwritten") -> Dict[str, Any]:
        if not self._load_model(model):
            return {"status": "error", "error": "Model unavailable", "text": "", "confidence": 0}
        try:
            import torch
            processor = self.processors[model]
            model_obj = self.models[model]
            if line_image.mode != "RGB":
                line_image = line_image.convert("RGB")
            if line_image.width < 10 or line_image.height < 10:
                return {"status": "error", "error": "Image too small", "text": "", "confidence": 0}
            pixel_values = processor(images=line_image, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = model_obj.generate(pixel_values, max_length=256, num_beams=4, early_stopping=True)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return {"status": "success", "text": text, "confidence": 0.8 if text else 0.1}
        except Exception as e:
            return {"status": "error", "error": str(e), "text": "", "confidence": 0}

    def transcribe_lines(
        self, image: Image.Image, lines: List[Dict], model: str = "handwritten"
    ) -> Dict[str, Any]:
        if not self._load_model(model):
            return {"status": "error", "error": "Model failed", "text": "", "lines": []}
        if image.mode != "RGB":
            image = image.convert("RGB")
        results = []
        all_text = []
        total_conf = 0.0
        errors = 0
        img_w, img_h = image.size
        for idx, line in enumerate(lines):
            bbox = line.get("bbox", {})
            x, y = bbox.get("x", 0), bbox.get("y", 0)
            w, h = bbox.get("width", 0), bbox.get("height", 0)
            if w <= 0 or h <= 0:
                results.append({"line_id": line.get("id", f"line_{idx:03d}"), "bbox": bbox, "text": "", "confidence": 0})
                errors += 1
                continue
            x, y = max(0, x), max(0, y)
            x2, y2 = min(x + w, img_w), min(y + h, img_h)
            try:
                line_crop = image.crop((x, y, x2, y2))
                if line_crop.width < 5 or line_crop.height < 5:
                    results.append({"line_id": line.get("id", f"line_{idx:03d}"), "bbox": bbox, "text": "", "confidence": 0})
                    errors += 1
                    continue
            except Exception:
                results.append({"line_id": line.get("id", f"line_{idx:03d}"), "bbox": bbox, "text": "", "confidence": 0})
                errors += 1
                continue
            result = self.transcribe_line(line_crop, model)
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
        successful = len(lines) - errors
        return {
            "status": "success" if successful > 0 else "error",
            "lines": results,
            "text": "\n".join(all_text),
            "confidence": total_conf / successful if successful > 0 else 0,
        }
