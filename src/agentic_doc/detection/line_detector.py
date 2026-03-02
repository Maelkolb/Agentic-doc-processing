"""
Line detection using Surya's DetectionPredictor.

Supports both dict-based and object-based Surya API responses (e.g. surya-ocr 0.17.x and newer).
Use DETECTOR_BATCH_SIZE, DETECTOR_BLANK_THRESHOLD, DETECTOR_TEXT_THRESHOLD env vars for tuning.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


def _get_bboxes_from_page(page: Any) -> List[Any]:
    """Get bboxes list from a single-page prediction (dict or object)."""
    if isinstance(page, dict):
        return page.get("bboxes", [])
    return getattr(page, "bboxes", [])


def _get_polygon_from_bbox(bbox: Any) -> List[List[int]]:
    """
    Extract polygon from a single bbox (dict or object).
    Surya: 4 points clockwise from top-left; normalize to list of [x, y].
    """
    if isinstance(bbox, dict):
        raw = bbox.get("polygon", [])
    else:
        raw = getattr(bbox, "polygon", [])
    if not raw:
        return []
    return [[int(pt[0]), int(pt[1])] for pt in raw]


def _get_confidence_from_bbox(bbox: Any) -> float:
    """Extract confidence from a single bbox (dict or object)."""
    if isinstance(bbox, dict):
        return float(bbox.get("confidence", 0.0))
    return float(getattr(bbox, "confidence", 0.0))


class LineDetector:
    """
    Line detection using Surya's DetectionPredictor.
    Detects text line polygons within detected regions.
    Supports both dict and object Surya API responses for compatibility across versions.
    """

    LINE_MARGIN = 0.0005

    def __init__(self, use_layout_fallback: bool = False):
        self.predictor = None
        self._layout_predictor = None
        self._foundation_predictor = None
        self._initialized = False
        self._use_layout_fallback = use_layout_fallback

    def _initialize(self) -> None:
        """Lazy initialization of Surya DetectionPredictor."""
        if self._initialized:
            return
        print("Loading Surya detection model...")
        from surya.detection import DetectionPredictor

        self.predictor = DetectionPredictor()
        self._initialized = True
        print("Surya DetectionPredictor loaded")

    def _init_layout_fallback(self) -> None:
        """Lazy init for layout-based fallback (optional)."""
        if self._layout_predictor is not None:
            return
        try:
            from surya.foundation import FoundationPredictor
            from surya.layout import LayoutPredictor
            from surya.settings import settings

            self._foundation_predictor = FoundationPredictor(
                checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
            )
            self._layout_predictor = LayoutPredictor(self._foundation_predictor)
        except Exception as e:
            print(f"Layout fallback not available: {e}")
            self._use_layout_fallback = False

    def _add_margin_to_polygon(
        self, polygon: List[List[int]], img_width: int, img_height: int
    ) -> List[List[int]]:
        """Add small margin to line polygon by expanding outward."""
        if not polygon or len(polygon) < 3:
            return polygon
        pts = np.array(polygon, dtype=np.float32)
        centroid = pts.mean(axis=0)
        margin_factor = 1.0 + self.LINE_MARGIN * 2
        expanded = centroid + (pts - centroid) * margin_factor
        expanded[:, 0] = np.clip(expanded[:, 0], 0, max(0, img_width - 1))
        expanded[:, 1] = np.clip(expanded[:, 1], 0, max(0, img_height - 1))
        return expanded.astype(int).tolist()

    def _polygon_to_bbox(self, polygon: List[List[int]]) -> Dict[str, int]:
        """Convert polygon to axis-aligned bounding box."""
        if not polygon:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        return {
            "x": min(x_coords),
            "y": min(y_coords),
            "width": max(x_coords) - min(x_coords),
            "height": max(y_coords) - min(y_coords),
        }

    def _normalize_predictions_to_lines(
        self,
        predictions: List[Any],
        img_width: int,
        img_height: int,
        region_bbox: Optional[Dict[str, int]] = None,
        region_id: str = "full_page",
    ) -> List[Dict[str, Any]]:
        """
        Normalize Surya output (list of dicts or objects, one per image) to our line format.
        If region_bbox is set, predictions are in crop coords; we offset to full image.
        """
        if not predictions:
            return []
        page = predictions[0]
        bboxes = _get_bboxes_from_page(page)
        lines = []
        for i, det_bbox in enumerate(bboxes):
            polygon = _get_polygon_from_bbox(det_bbox)
            if not polygon:
                continue
            if region_bbox is not None:
                polygon = [
                    [pt[0] + region_bbox["x"], pt[1] + region_bbox["y"]]
                    for pt in polygon
                ]
            margined = self._add_margin_to_polygon(polygon, img_width, img_height)
            line_bbox = self._polygon_to_bbox(margined)
            conf = _get_confidence_from_bbox(det_bbox)
            line_id = f"{region_id}_line_{i + 1:03d}" if region_id != "full_page" else f"line_{i + 1:03d}"
            lines.append({
                "id": line_id,
                "polygon": margined,
                "bbox": line_bbox,
                "confidence": conf,
            })
        lines.sort(key=lambda l: (l["bbox"]["y"], l["bbox"]["x"]))
        return lines

    def _detect_with_layout_fallback(
        self, image: Image.Image, img_width: int, img_height: int
    ) -> List[Dict[str, Any]]:
        """Use Layout model bboxes as line-level boxes when detection returns nothing."""
        self._init_layout_fallback()
        if self._layout_predictor is None:
            return []
        try:
            layout_predictions = self._layout_predictor([image])
            if not layout_predictions:
                return []
            page = layout_predictions[0]
            bboxes = _get_bboxes_from_page(page)
            lines = []
            for i, lb in enumerate(bboxes):
                polygon = _get_polygon_from_bbox(lb)
                if not polygon:
                    continue
                margined = self._add_margin_to_polygon(polygon, img_width, img_height)
                line_bbox = self._polygon_to_bbox(margined)
                conf = _get_confidence_from_bbox(lb)
                lines.append({
                    "id": f"line_{i + 1:03d}",
                    "polygon": margined,
                    "bbox": line_bbox,
                    "confidence": conf,
                })
            lines.sort(key=lambda l: (l["bbox"]["y"], l["bbox"]["x"]))
            return lines
        except Exception as e:
            print(f"Layout fallback failed: {e}")
            return []

    def detect(
        self, image_path: str, regions: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Detect text lines within regions using Surya DetectionPredictor.
        If regions is None, detects on the full image.
        Normalizes both dict and object Surya responses.
        """
        self._initialize()
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        if not regions:
            predictions = self.predictor([image])
            lines = self._normalize_predictions_to_lines(
                predictions, img_width, img_height, region_bbox=None, region_id="full_page"
            )
            if not lines and self._use_layout_fallback:
                lines = self._detect_with_layout_fallback(image, img_width, img_height)
            return {
                "status": "success",
                "tool": "surya",
                "image_path": image_path,
                "regions": [{
                    "id": "full_page",
                    "type": "TextRegion",
                    "bbox": {"x": 0, "y": 0, "width": img_width, "height": img_height},
                    "lines": lines,
                    "line_count": len(lines),
                }],
                "total_lines": len(lines),
            }

        results = []
        total_lines = 0
        for region in regions:
            region_bbox = region.get("bbox")
            if not region_bbox:
                results.append({**region, "lines": [], "line_count": 0})
                continue
            region_type = region.get("type", "TextRegion")
            if region_type in ("ImageRegion", "DiagramRegion"):
                results.append({**region, "lines": [], "line_count": 0})
                continue

            crop = image.crop((
                region_bbox["x"],
                region_bbox["y"],
                region_bbox["x"] + region_bbox["width"],
                region_bbox["y"] + region_bbox["height"],
            ))
            if crop.width < 10 or crop.height < 10:
                results.append({**region, "lines": [], "line_count": 0})
                continue

            try:
                predictions = self.predictor([crop])
            except Exception as e:
                print(f"Line detection failed for region {region.get('id', 'unknown')}: {e}")
                results.append({**region, "lines": [], "line_count": 0})
                continue

            lines = self._normalize_predictions_to_lines(
                predictions,
                img_width,
                img_height,
                region_bbox=region_bbox,
                region_id=region.get("id", "region"),
            )
            if not lines and self._use_layout_fallback:
                lines_crop = self._detect_with_layout_fallback(crop, crop.width, crop.height)
                lines = []
                for j, ln in enumerate(lines_crop):
                    bbox = ln["bbox"]
                    poly = ln["polygon"]
                    lines.append({
                        "id": f"{region.get('id', 'region')}_line_{j + 1:03d}",
                        "polygon": [[p[0] + region_bbox["x"], p[1] + region_bbox["y"]] for p in poly],
                        "bbox": {
                            "x": bbox["x"] + region_bbox["x"],
                            "y": bbox["y"] + region_bbox["y"],
                            "width": bbox["width"],
                            "height": bbox["height"],
                        },
                        "confidence": ln["confidence"],
                    })

            results.append({
                "id": region.get("id", "region"),
                "type": region_type,
                "bbox": region_bbox,
                "reading_order": region.get("reading_order", 0),
                "confidence": region.get("confidence", 0.9),
                "description": region.get("description", ""),
                "lines": lines,
                "line_count": len(lines),
            })
            total_lines += len(lines)

        return {
            "status": "success",
            "tool": "surya",
            "image_path": image_path,
            "regions": results,
            "total_lines": total_lines,
        }
