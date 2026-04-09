"""
Line detection using Surya's DetectionPredictor.

Detects text lines within each region by cropping from the full image
and running Surya on each crop.  Coordinates are offset back to
full-image space.

IMPORTANT: Surya 0.17.x requires ``transformers>=4.56.1,<5``.
Using transformers 5.x causes the model to produce garbage output.
The ``pyproject.toml`` pins this dependency correctly.

Supports both dict-based and object-based Surya API responses.
Env vars: DETECTOR_BATCH_SIZE, DETECTOR_BLANK_THRESHOLD, DETECTOR_TEXT_THRESHOLD.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Surya response helpers (dict / object agnostic)
# ---------------------------------------------------------------------------


def _get_bboxes_from_page(page: Any) -> List[Any]:
    """Get bboxes list from a single-page prediction (dict or object)."""
    if isinstance(page, dict):
        return page.get("bboxes", [])
    return getattr(page, "bboxes", [])


def _get_polygon_from_bbox(bbox: Any) -> List[List[int]]:
    """Extract polygon from a single bbox (dict or object).
    Returns 4 points clockwise from top-left as [[x,y], ...]."""
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


# ---------------------------------------------------------------------------
# LineDetector
# ---------------------------------------------------------------------------


class LineDetector:
    """
    Detect text lines within document regions using Surya's DetectionPredictor.

    For each text region the detector crops from the full-page image,
    runs Surya on the crop, and maps coordinates back to full-image space.
    """

    LINE_MARGIN = 0.0005

    def __init__(self, use_layout_fallback: bool = False) -> None:
        self.predictor = None
        self._initialized = False
        self._use_layout_fallback = use_layout_fallback

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Lazy-load the Surya DetectionPredictor."""
        if self._initialized:
            return
        try:
            print("Loading Surya detection model ...")
            from surya.detection import DetectionPredictor

            self.predictor = DetectionPredictor()
            self._initialized = True
            print("Surya DetectionPredictor loaded")
        except Exception as e:
            print(f"Surya DetectionPredictor failed to load: {e}")
            self._initialized = True
            self.predictor = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _add_margin_to_polygon(
        self, polygon: List[List[int]], img_width: int, img_height: int
    ) -> List[List[int]]:
        """Slightly expand a polygon outward from its centroid."""
        if not polygon or len(polygon) < 3:
            return polygon
        pts = np.array(polygon, dtype=np.float32)
        centroid = pts.mean(axis=0)
        factor = 1.0 + self.LINE_MARGIN * 2
        expanded = centroid + (pts - centroid) * factor
        expanded[:, 0] = np.clip(expanded[:, 0], 0, max(0, img_width - 1))
        expanded[:, 1] = np.clip(expanded[:, 1], 0, max(0, img_height - 1))
        return expanded.astype(int).tolist()

    @staticmethod
    def _polygon_to_bbox(polygon: List[List[int]]) -> Dict[str, int]:
        """Convert polygon to axis-aligned bounding box dict."""
        if not polygon:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return {
            "x": min(xs), "y": min(ys),
            "width": max(xs) - min(xs), "height": max(ys) - min(ys),
        }

    # ------------------------------------------------------------------
    # Surya output normalisation
    # ------------------------------------------------------------------

    def _normalize_predictions_to_lines(
        self,
        predictions: List[Any],
        img_width: int,
        img_height: int,
        region_bbox: Optional[Dict[str, int]] = None,
        region_id: str = "full_page",
    ) -> List[Dict[str, Any]]:
        """Convert Surya predictions (one entry per image) into our line format.

        If *region_bbox* is set the predictions are assumed to be in crop
        coordinates and are offset to full-image space.
        """
        if not predictions:
            return []
        page = predictions[0]
        bboxes = _get_bboxes_from_page(page)
        lines: List[Dict[str, Any]] = []
        for i, det_bbox in enumerate(bboxes):
            polygon = _get_polygon_from_bbox(det_bbox)
            if not polygon:
                continue
            # offset crop coords → full image
            if region_bbox is not None:
                polygon = [
                    [pt[0] + region_bbox["x"], pt[1] + region_bbox["y"]]
                    for pt in polygon
                ]
            margined = self._add_margin_to_polygon(polygon, img_width, img_height)
            line_bbox = self._polygon_to_bbox(margined)
            conf = _get_confidence_from_bbox(det_bbox)
            line_id = (
                f"{region_id}_line_{i + 1:03d}"
                if region_id != "full_page"
                else f"line_{i + 1:03d}"
            )
            lines.append({
                "id": line_id,
                "polygon": margined,
                "bbox": line_bbox,
                "confidence": conf,
            })
        lines.sort(key=lambda l: (l["bbox"]["y"], l["bbox"]["x"]))
        return lines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, image_path: str, regions: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Detect text lines.

        If *regions* is ``None``, detect on the full image.
        Otherwise crop each region and run Surya on the crop.
        """
        self._initialize()
        if self.predictor is None:
            return {
                "status": "error",
                "error": "Surya DetectionPredictor not available. "
                         "Install surya-ocr and transformers>=4.56.1,<5.",
            }

        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        # ---- full-page mode (no regions) ----
        if not regions:
            predictions = self.predictor([image])
            lines = self._normalize_predictions_to_lines(
                predictions, img_width, img_height,
            )
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

        # ---- per-region crop mode ----
        _skip_types = {"ImageRegion", "DiagramRegion", "DecorationRegion"}
        results: List[Dict[str, Any]] = []
        total_lines = 0

        for i, region in enumerate(regions):
            rid = region.get("id", f"region_{i + 1:03d}")
            rtype = region.get("type", "TextRegion")
            rbbox = region.get("bbox")

            # skip non-text / invalid regions
            if not rbbox or rtype in _skip_types:
                results.append(self._build_region(region, i, []))
                continue

            crop = image.crop((
                rbbox["x"], rbbox["y"],
                rbbox["x"] + rbbox["width"],
                rbbox["y"] + rbbox["height"],
            ))
            if crop.width < 10 or crop.height < 10:
                results.append(self._build_region(region, i, []))
                continue

            try:
                predictions = self.predictor([crop])
            except Exception as e:
                print(f"Line detection failed for region {rid}: {e}")
                results.append(self._build_region(region, i, []))
                continue

            lines = self._normalize_predictions_to_lines(
                predictions, img_width, img_height,
                region_bbox=rbbox, region_id=rid,
            )
            results.append(self._build_region(region, i, lines))
            total_lines += len(lines)

        return {
            "status": "success",
            "tool": "surya",
            "image_path": image_path,
            "regions": results,
            "total_lines": total_lines,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_region(
        region: Dict, index: int, lines: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "id": region.get("id", f"region_{index + 1:03d}"),
            "type": region.get("type", "TextRegion"),
            "bbox": region.get("bbox", {}),
            "reading_order": region.get("reading_order", index + 1),
            "confidence": region.get("confidence", 0.9),
            "description": region.get("description", ""),
            "lines": lines,
            "line_count": len(lines),
        }
