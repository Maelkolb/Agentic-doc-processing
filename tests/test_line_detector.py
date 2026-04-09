"""Tests for LineDetector: Surya response helpers and detection logic."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_doc.detection.line_detector import (
    LineDetector,
    _get_bboxes_from_page,
    _get_confidence_from_bbox,
    _get_polygon_from_bbox,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _make_temp_image(width=400, height=300):
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.new("RGB", (width, height), "white").save(f.name)
    f.close()
    return f.name


def _mock_detector(mock_predictions):
    """Create a LineDetector with mocked Surya predictor."""
    det = LineDetector()
    det._initialized = True
    det.predictor = MagicMock(return_value=mock_predictions)
    return det


# ── Surya response helpers ───────────────────────────────────────────────


class TestSuryaHelpers:
    def test_get_bboxes_dict(self):
        page = {"bboxes": [{"polygon": [[0, 0], [10, 0], [10, 5], [0, 5]]}]}
        assert len(_get_bboxes_from_page(page)) == 1
        assert _get_bboxes_from_page({}) == []

    def test_get_bboxes_object(self):
        obj = MagicMock()
        obj.bboxes = [MagicMock()]
        assert len(_get_bboxes_from_page(obj)) == 1

    def test_get_polygon_dict(self):
        assert _get_polygon_from_bbox({"polygon": [[1, 2], [3, 4], [5, 6], [7, 8]]}) == [
            [1, 2], [3, 4], [5, 6], [7, 8]
        ]
        assert _get_polygon_from_bbox({}) == []

    def test_get_polygon_object(self):
        bbox = MagicMock()
        bbox.polygon = [(0, 0), (10, 0), (10, 5), (0, 5)]
        assert _get_polygon_from_bbox(bbox) == [[0, 0], [10, 0], [10, 5], [0, 5]]

    def test_get_confidence_dict(self):
        assert _get_confidence_from_bbox({"confidence": 0.85}) == 0.85
        assert _get_confidence_from_bbox({}) == 0.0

    def test_get_confidence_object(self):
        bbox = MagicMock()
        bbox.confidence = 0.77
        assert _get_confidence_from_bbox(bbox) == 0.77


# ── Full page detection ──────────────────────────────────────────────────


class TestFullPage:
    def test_full_page_returns_lines(self):
        mock_pred = [{
            "bboxes": [
                {"polygon": [[5, 5], [50, 5], [50, 15], [5, 15]], "confidence": 0.88},
                {"polygon": [[5, 25], [50, 25], [50, 35], [5, 35]], "confidence": 0.92},
            ]
        }]
        det = _mock_detector(mock_pred)
        path = _make_temp_image()
        try:
            result = det.detect(path, regions=None)
        finally:
            os.unlink(path)
        assert result["status"] == "success"
        assert result["tool"] == "surya"
        assert result["total_lines"] == 2
        assert result["regions"][0]["id"] == "full_page"

    def test_object_based_response(self):
        bbox1 = MagicMock()
        bbox1.polygon = [[0, 0], [100, 0], [100, 12], [0, 12]]
        bbox1.confidence = 0.9
        page = MagicMock()
        page.bboxes = [bbox1]
        det = _mock_detector([page])
        path = _make_temp_image()
        try:
            result = det.detect(path, regions=None)
        finally:
            os.unlink(path)
        assert result["total_lines"] == 1
        assert result["regions"][0]["lines"][0]["confidence"] == 0.9


# ── Region crop detection ────────────────────────────────────────────────


class TestRegionCrops:
    def test_lines_offset_to_full_image(self):
        """Lines detected in crop should be offset by region origin."""
        mock_pred = [{
            "bboxes": [
                {"polygon": [[10, 10], [180, 10], [180, 25], [10, 25]], "confidence": 0.9},
            ]
        }]
        det = _mock_detector(mock_pred)
        path = _make_temp_image()
        regions = [
            {"id": "r1", "type": "TextRegion",
             "bbox": {"x": 50, "y": 100, "width": 200, "height": 80}},
        ]
        try:
            result = det.detect(path, regions=regions)
        finally:
            os.unlink(path)
        assert result["status"] == "success"
        r1 = result["regions"][0]
        assert r1["line_count"] == 1
        line = r1["lines"][0]
        # polygon should be offset: x + 50, y + 100
        assert line["bbox"]["x"] >= 50
        assert line["bbox"]["y"] >= 100

    def test_multiple_regions(self):
        mock_pred = [{
            "bboxes": [
                {"polygon": [[5, 5], [190, 5], [190, 20], [5, 20]], "confidence": 0.9},
            ]
        }]
        det = _mock_detector(mock_pred)
        path = _make_temp_image()
        regions = [
            {"id": "r1", "type": "HeadingRegion",
             "bbox": {"x": 0, "y": 0, "width": 200, "height": 50}},
            {"id": "r2", "type": "ParagraphRegion",
             "bbox": {"x": 0, "y": 60, "width": 200, "height": 200}},
        ]
        try:
            result = det.detect(path, regions=regions)
        finally:
            os.unlink(path)
        # Surya is called once per region (2 calls)
        assert det.predictor.call_count == 2
        assert result["total_lines"] == 2

    def test_image_region_skipped(self):
        mock_pred = [{"bboxes": []}]
        det = _mock_detector(mock_pred)
        path = _make_temp_image()
        regions = [
            {"id": "r1", "type": "ImageRegion",
             "bbox": {"x": 0, "y": 0, "width": 400, "height": 300}},
        ]
        try:
            result = det.detect(path, regions=regions)
        finally:
            os.unlink(path)
        assert result["regions"][0]["line_count"] == 0
        det.predictor.assert_not_called()

    def test_tiny_region_skipped(self):
        mock_pred = [{"bboxes": []}]
        det = _mock_detector(mock_pred)
        path = _make_temp_image()
        regions = [
            {"id": "r1", "type": "TextRegion",
             "bbox": {"x": 0, "y": 0, "width": 5, "height": 5}},
        ]
        try:
            result = det.detect(path, regions=regions)
        finally:
            os.unlink(path)
        assert result["regions"][0]["line_count"] == 0
        det.predictor.assert_not_called()

    def test_surya_not_available(self):
        det = LineDetector()
        det._initialized = True
        det.predictor = None
        path = _make_temp_image()
        try:
            result = det.detect(path, regions=None)
        finally:
            os.unlink(path)
        assert result["status"] == "error"

    def test_lines_sorted_by_y(self):
        mock_pred = [{
            "bboxes": [
                {"polygon": [[5, 50], [100, 50], [100, 60], [5, 60]], "confidence": 0.8},
                {"polygon": [[5, 10], [100, 10], [100, 20], [5, 20]], "confidence": 0.9},
            ]
        }]
        det = _mock_detector(mock_pred)
        path = _make_temp_image()
        try:
            result = det.detect(path, regions=None)
        finally:
            os.unlink(path)
        lines = result["regions"][0]["lines"]
        assert lines[0]["bbox"]["y"] < lines[1]["bbox"]["y"]
