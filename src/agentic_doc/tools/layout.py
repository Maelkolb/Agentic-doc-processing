"""Layout tools: detect_regions, detect_lines, visualize_layout."""

import json
import os
import traceback
from typing import Optional

from langchain_core.tools import tool


def get_layout_tools(state, logger, region_detector, line_detector, visualizer):
    @tool
    def detect_regions(image_path: str, document_context: Optional[str] = None) -> str:
        """STEP 3: Detect and classify document regions using Gemini vision.
        Detects regions (paragraphs, tables, marginalia, etc.), types, reading order.
        Use the enhanced image path if enhance_image was called.
        Returns JSON with regions, bboxes, reading_order.
        Can also be called directly if assess_document failed (uses PIL, not cv2)."""
        logger.info(f"Detecting regions in: {image_path}")
        # Update state if this is the first tool to set the image path
        if not state.image_path:
            state.image_path = image_path
            state.current_image_path = image_path
        context_dict = None
        if document_context:
            try:
                context_dict = json.loads(document_context)
            except Exception:
                if state.assessment_result:
                    context_dict = state.assessment_result.get("content_analysis", {})
        elif state.assessment_result:
            context_dict = state.assessment_result.get("content_analysis", {})
        try:
            result = region_detector.detect_and_classify(image_path, context_dict)
            if result.get("status") == "success":
                state.region_result = result
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e), "traceback": traceback.format_exc()})

    @tool
    def detect_lines(image_path: str, regions_json: Optional[str] = None) -> str:
        """STEP 4: Detect text lines within each region using Surya. Must be called AFTER detect_regions.
        Returns JSON with regions updated to include line polygons and line counts."""
        logger.info(f"Detecting lines in: {image_path}")
        regions = None
        if regions_json:
            try:
                data = json.loads(regions_json)
                regions = data.get("regions", data)
            except json.JSONDecodeError:
                pass
        if not regions and state.region_result:
            regions = state.region_result.get("regions", [])
        if not regions:
            return json.dumps({"status": "error", "error": "No regions available. Call detect_regions first."})
        try:
            result = line_detector.detect(image_path, regions)
            if result.get("status") == "success":
                state.line_result = result
                logger.info(f"   Detected {result['total_lines']} lines across {len(result['regions'])} regions")
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e), "traceback": traceback.format_exc()})

    @tool
    def visualize_layout(image_path: str, output_path: Optional[str] = None) -> str:
        """Visualize detected regions and lines on the document image. Call after detect_lines for full view."""
        regions = None
        if state.line_result:
            regions = state.line_result.get("regions", [])
        elif state.region_result:
            regions = state.region_result.get("regions", [])
        if not regions:
            return json.dumps({"status": "error", "error": "No regions. Call detect_regions and detect_lines first."})
        try:
            if not output_path:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_layout_visualization.png"
            fig = visualizer.visualize(image_path, regions)
            fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
            import matplotlib.pyplot as plt
            plt.close(fig)
            state.output_files["visualization"] = output_path
            total_lines = sum(len(r.get("lines", [])) for r in regions)
            return json.dumps({
                "status": "success",
                "output_path": output_path,
                "total_regions": len(regions),
                "total_lines": total_lines,
                "region_types": list(set(r.get("type", "Unknown") for r in regions)),
            }, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    return [detect_regions, detect_lines, visualize_layout]
