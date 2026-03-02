"""Tests for line detector normalization (dict vs object Surya responses)."""

import sys
from pathlib import Path

from unittest.mock import MagicMock

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_doc.detection.line_detector import (
    LineDetector,
    _get_bboxes_from_page,
    _get_confidence_from_bbox,
    _get_polygon_from_bbox,
)


def test_get_bboxes_from_page_dict():
    page = {"bboxes": [{"polygon": [[0, 0], [10, 0], [10, 5], [0, 5]], "confidence": 0.9}]}
    assert len(_get_bboxes_from_page(page)) == 1
    assert _get_bboxes_from_page({}) == []
    assert _get_bboxes_from_page({"bboxes": []}) == []


def test_get_bboxes_from_page_object():
    obj = MagicMock()
    obj.bboxes = [MagicMock(polygon=[[0, 0], [10, 0], [10, 5], [0, 5]], confidence=0.9)]
    assert len(_get_bboxes_from_page(obj)) == 1


def test_get_polygon_from_bbox_dict():
    bbox = {"polygon": [[1, 2], [11, 2], [11, 7], [1, 7]]}
    assert _get_polygon_from_bbox(bbox) == [[1, 2], [11, 2], [11, 7], [1, 7]]
    assert _get_polygon_from_bbox({}) == []
    assert _get_polygon_from_bbox({"polygon": []}) == []


def test_get_polygon_from_bbox_object():
    bbox = MagicMock()
    bbox.polygon = [(0, 0), (10, 0), (10, 5), (0, 5)]
    assert _get_polygon_from_bbox(bbox) == [[0, 0], [10, 0], [10, 5], [0, 5]]


def test_get_confidence_from_bbox_dict():
    assert _get_confidence_from_bbox({"confidence": 0.85}) == 0.85
    assert _get_confidence_from_bbox({}) == 0.0


def test_get_confidence_from_bbox_object():
    bbox = MagicMock()
    bbox.confidence = 0.77
    assert _get_confidence_from_bbox(bbox) == 0.77


def test_normalize_predictions_to_lines_dict_response():
    """Simulate Surya returning list of dicts (e.g. newer API)."""
    det = LineDetector(use_layout_fallback=False)
    predictions = [{
        "bboxes": [
            {"polygon": [[0, 0], [100, 0], [100, 12], [0, 12]], "confidence": 0.9},
            {"polygon": [[0, 20], [80, 20], [80, 32], [0, 32]], "confidence": 0.85},
        ]
    }]
    lines = det._normalize_predictions_to_lines(
        predictions, img_width=200, img_height=100, region_bbox=None, region_id="full_page"
    )
    assert len(lines) == 2
    assert lines[0]["id"] == "line_001"
    assert lines[0]["bbox"]["y"] <= lines[1]["bbox"]["y"]
    assert lines[0]["confidence"] == 0.9
    assert len(lines[0]["polygon"]) == 4


def test_normalize_predictions_to_lines_object_response():
    """Simulate Surya returning list of objects with .bboxes, .polygon, .confidence."""
    det = LineDetector(use_layout_fallback=False)
    bbox1 = MagicMock()
    bbox1.polygon = [[0, 0], [100, 0], [100, 12], [0, 12]]
    bbox1.confidence = 0.9
    page = MagicMock()
    page.bboxes = [bbox1]
    predictions = [page]
    lines = det._normalize_predictions_to_lines(
        predictions, img_width=200, img_height=100, region_bbox=None, region_id="full_page"
    )
    assert len(lines) == 1
    assert lines[0]["confidence"] == 0.9
    assert lines[0]["bbox"]["width"] == 100 and lines[0]["bbox"]["height"] == 12


def test_detect_full_image_mocked():
    """Test detect() with mocked Surya (dict response) to avoid loading model."""
    det = LineDetector(use_layout_fallback=False)
    det._initialized = True
    mock_pred = [{
        "bboxes": [
            {"polygon": [[5, 5], [50, 5], [50, 15], [5, 15]], "confidence": 0.88},
        ]
    }]
    det.predictor = MagicMock(return_value=mock_pred)

    from PIL import Image
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.new("RGB", (100, 50), color="white").save(f.name)
        try:
            result = det.detect(f.name, regions=None)
        finally:
            os.unlink(f.name)

    assert result["status"] == "success"
    assert result["total_lines"] == 1
    assert len(result["regions"]) == 1
    assert result["regions"][0]["line_count"] == 1
    assert result["regions"][0]["lines"][0]["bbox"]["x"] >= 0
