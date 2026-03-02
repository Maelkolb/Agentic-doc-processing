"""Analysis tools: assess_document, enhance_image."""

import json
import os
import traceback
from typing import List

from langchain_core.tools import tool


def get_analysis_tools(state, logger, document_assessor, image_enhancer):
    @tool
    def assess_document(image_path: str) -> str:
        """STEP 1: Assess a document image to determine characteristics and optimal processing pipeline.
        Performs CV analysis (quality metrics) and LLM content analysis (script, language, layout).
        Returns JSON with quality_metrics, content_analysis, recommendations.
        Check recommendations.needs_preprocessing to decide if enhance_image is needed.
        DO NOT retry this tool with the same path if it returns an error — the error is permanent."""
        logger.info(f"Assessing document: {image_path}")
        # Validate the file exists before any processing
        if not os.path.isfile(image_path):
            dir_path = os.path.dirname(image_path) or "."
            try:
                contents = os.listdir(dir_path)[:20]
            except OSError:
                contents = ["<directory not accessible>"]
            return json.dumps({
                "status": "error",
                "error": f"File not found: {image_path}",
                "directory_exists": os.path.isdir(dir_path),
                "directory_contents_sample": contents,
                "hint": "Check the image path. The file must exist on disk before processing. "
                        "Do NOT retry with the same path — it will fail again.",
            })
        state.image_path = image_path
        state.current_image_path = image_path
        try:
            result = document_assessor.assess(image_path)
            state.assessment_result = result
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "hint": "If the image cannot be read, check the file format and integrity. "
                        "Do NOT retry with the same path — try detect_regions directly, "
                        "which uses a different image reader.",
            })

    @tool
    def enhance_image(image_path: str, operations: List[str]) -> str:
        """STEP 2 (OPTIONAL): Preprocess the image. Only call if assess_document recommended preprocessing.
        Operations: deskew, denoise, enhance_contrast, remove_bleedthrough, enhance_faded.
        Returns JSON with original_path and enhanced_path. Use enhanced_path for subsequent steps."""
        logger.info(f"Enhancing image with operations: {operations}")
        if not os.path.isfile(image_path):
            return json.dumps({"status": "error", "error": f"File not found: {image_path}"})
        try:
            result = image_enhancer.enhance(image_path, operations)
            if result.get("status") == "success":
                state.enhanced_path = result["enhanced_path"]
                state.current_image_path = result["enhanced_path"]
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    return [assess_document, enhance_image]
