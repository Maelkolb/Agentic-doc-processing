"""Transcription tools: get_transcription_plan, transcribe_with_*, compile_transcription."""

import json
from typing import Optional

import numpy as np
from langchain_core.tools import tool
from PIL import Image


def _get_regions_data(state):
    if state.line_result and "regions" in state.line_result:
        return state.line_result, True
    if state.region_result and "regions" in state.region_result:
        return state.region_result, False
    return None, False


def _find_region(regions_data, region_id):
    for r in regions_data.get("regions", []):
        if r.get("id") == region_id:
            return r
    return None


def get_transcription_tools(state, logger, tesseract_ocr, trocr_htr, llm_transcriber):
    @tool
    def get_transcription_plan() -> str:
        """Get region details and transcription recommendations. Shows which tools are available (tesseract, trocr, llm) based on line detection."""
        regions_data, has_lines = _get_regions_data(state)
        if not regions_data:
            return json.dumps({"status": "error", "error": "No regions. Run detect_regions first."})
        assessment = state.assessment_result or {}
        content_analysis = assessment.get("content_analysis", {})
        regions_info = []
        for r in regions_data.get("regions", []):
            region_type = r.get("type", "TextRegion")
            script_type = (content_analysis.get("script_type") or "unknown").lower()
            line_count = r.get("line_count", len(r.get("lines", [])))
            if region_type in ("ImageRegion", "DiagramRegion"):
                recommendation = "llm (output_format='description')"
                available_tools = ["llm"]
            elif region_type == "TableRegion":
                recommendation = "llm (output_format='markdown')"
                available_tools = ["llm"]
            elif region_type == "MarginaliaRegion":
                recommendation = "SKIP (marginalia not transcribed)"
                available_tools = []
            elif not has_lines or line_count == 0:
                recommendation = "llm (line detection failed/unavailable)"
                available_tools = ["llm"]
            else:
                available_tools = ["tesseract", "trocr", "llm"]
                recommendation = "llm" if "kurrent" in script_type else "choose tesseract/trocr/llm"
            regions_info.append({
                "region_id": r["id"],
                "type": region_type,
                "description": r.get("description", ""),
                "line_count": line_count,
                "bbox": r.get("bbox"),
                "available_tools": available_tools,
                "recommendation": recommendation,
            })
        return json.dumps({
            "status": "success",
            "line_detection_available": has_lines,
            "document_info": {
                "script_type": content_analysis.get("script_type", "unknown"),
                "is_printed": content_analysis.get("is_printed", False),
                "language": content_analysis.get("primary_language", "unknown"),
                "period": content_analysis.get("estimated_period", "unknown"),
            },
            "total_regions": len(regions_info),
            "regions": regions_info,
        }, indent=2)

    @tool
    def transcribe_with_tesseract(region_id: str, languages: Optional[str] = "german,latin") -> str:
        """Transcribe a text region using Tesseract OCR (line-by-line). Requires line detection. Best for printed text."""
        regions_data, has_lines = _get_regions_data(state)
        if not regions_data:
            return json.dumps({"status": "error", "error": "No regions. Run detect_regions first."})
        if not has_lines:
            return json.dumps({"status": "error", "error": "Line detection required for Tesseract. Use transcribe_with_llm instead.", "suggestion": "transcribe_with_llm"})
        region = _find_region(regions_data, region_id)
        if not region:
            return json.dumps({"status": "error", "error": f"Region {region_id} not found"})
        image_path = state.current_image_path
        if not image_path:
            return json.dumps({"status": "error", "error": "No image path in state."})
        image = Image.open(image_path).convert("RGB")
        lines = region.get("lines", [])
        lang_list = [s.strip() for s in (languages or "german,latin").split(",")]
        result = tesseract_ocr.transcribe_lines(image, lines, lang_list)
        result["tool"] = "tesseract"
        result["region_id"] = region_id
        if not hasattr(state, "transcription_results"):
            state.transcription_results = {}
        state.transcription_results[region_id] = result
        return json.dumps(result, indent=2)

    @tool
    def transcribe_with_trocr(region_id: str, model: Optional[str] = "handwritten") -> str:
        """Transcribe a text region using TrOCR (line-by-line HTR). Requires line detection. Best for handwriting/Kurrent."""
        regions_data, has_lines = _get_regions_data(state)
        if not regions_data:
            return json.dumps({"status": "error", "error": "No regions. Run detect_regions first."})
        if not has_lines:
            return json.dumps({"status": "error", "error": "Line detection required for TrOCR. Use transcribe_with_llm instead.", "suggestion": "transcribe_with_llm"})
        region = _find_region(regions_data, region_id)
        if not region:
            return json.dumps({"status": "error", "error": f"Region {region_id} not found"})
        image_path = state.current_image_path
        if not image_path:
            return json.dumps({"status": "error", "error": "No image path."})
        image = Image.open(image_path).convert("RGB")
        lines = region.get("lines", [])
        result = trocr_htr.transcribe_lines(image, lines, model or "handwritten")
        result["tool"] = "trocr"
        result["region_id"] = region_id
        if not hasattr(state, "transcription_results"):
            state.transcription_results = {}
        state.transcription_results[region_id] = result
        return json.dumps(result, indent=2)

    @tool
    def transcribe_with_llm(
        region_id: str,
        output_format: Optional[str] = "text",
    ) -> str:
        """Transcribe a region using Gemini vision. Works with or without line detection. Use for tables (markdown), images (description), or complex/handwritten text."""
        regions_data, _ = _get_regions_data(state)
        if not regions_data:
            return json.dumps({"status": "error", "error": "No regions. Run detect_regions first."})
        region = _find_region(regions_data, region_id)
        if not region:
            return json.dumps({"status": "error", "error": f"Region {region_id} not found"})
        image_path = state.current_image_path
        if not image_path:
            return json.dumps({"status": "error", "error": "No image path."})
        image = Image.open(image_path).convert("RGB")
        bbox = region.get("bbox", {})
        x, y = bbox.get("x", 0), bbox.get("y", 0)
        w, h = bbox.get("width", 0), bbox.get("height", 0)
        crop = image.crop((x, y, x + w, y + h))
        context = ""
        if state.assessment_result:
            ca = state.assessment_result.get("content_analysis", {})
            context = f"Script: {ca.get('script_type', '')}. Language: {ca.get('primary_language', '')}."
        result = llm_transcriber.transcribe_region(
            crop,
            region_type=region.get("type", "TextRegion"),
            context=context,
            output_format=output_format or "text",
            lines=region.get("lines"),
        )
        result["tool"] = "llm"
        result["region_id"] = region_id
        if not hasattr(state, "transcription_results"):
            state.transcription_results = {}
        state.transcription_results[region_id] = result
        return json.dumps(result, indent=2)

    @tool
    def compile_transcription() -> str:
        """Compile all transcription results into a final document. Merge line-level and region-level text. Call after transcribing all regions."""
        if not getattr(state, "transcription_results", None) or not state.transcription_results:
            return json.dumps({"status": "error", "error": "No transcriptions. Transcribe regions first."})
        regions_source = None
        if state.line_result:
            regions_source = state.line_result.get("regions", [])
        elif state.region_result:
            regions_source = state.region_result.get("regions", [])
        if not regions_source:
            return json.dumps({"status": "error", "error": "No region data."})
        detected_lines_by_region = {}
        if state.line_result:
            for r in state.line_result.get("regions", []):
                detected_lines_by_region[r.get("id", "")] = r.get("lines", [])
        ordered_results = []
        for region in regions_source:
            region_id = region.get("id", "")
            region_type = region.get("type", "TextRegion")
            region_bbox = region.get("bbox", {})
            reading_order = region.get("reading_order", 0)
            trans_result = state.transcription_results.get(region_id, {})
            if not trans_result:
                continue
            trans_text = trans_result.get("text", "")
            trans_lines = trans_result.get("lines", [])
            trans_confidence = trans_result.get("confidence", 0)
            trans_status = trans_result.get("status", "unknown")
            trans_tool = trans_result.get("tool", "unknown")
            detected_lines = detected_lines_by_region.get(region_id, [])
            final_lines = []
            if trans_lines:
                for i, trans_line in enumerate(trans_lines):
                    line_text = trans_line.get("text", "")
                    line_id = trans_line.get("line_id", f"{region_id}_line_{i:03d}")
                    line_bbox = trans_line.get("bbox", {})
                    line_polygon = trans_line.get("polygon", [])
                    if not line_bbox and i < len(detected_lines):
                        det = detected_lines[i]
                        line_bbox = det.get("bbox", {})
                        line_polygon = det.get("polygon", line_polygon)
                        line_id = det.get("id", line_id)
                    final_lines.append({"line_id": line_id, "text": line_text, "bbox": line_bbox, "polygon": line_polygon, "confidence": trans_line.get("confidence", trans_confidence)})
            elif detected_lines and trans_text:
                text_parts = [p.strip() for p in trans_text.split("\n") if p.strip()]
                for i, detected in enumerate(detected_lines):
                    text_part = text_parts[i] if i < len(text_parts) else ""
                    if i == len(detected_lines) - 1 and len(text_parts) > len(detected_lines):
                        text_part = " ".join(text_parts[i:])
                    final_lines.append({
                        "line_id": detected.get("id", f"{region_id}_line_{i:03d}"),
                        "text": text_part,
                        "bbox": detected.get("bbox", {}),
                        "polygon": detected.get("polygon", []),
                        "confidence": trans_confidence,
                    })
            ordered_results.append({
                "region_id": region_id,
                "region_type": region_type,
                "reading_order": reading_order,
                "bbox": region_bbox,
                "tool_used": trans_tool,
                "text": trans_text,
                "lines": final_lines,
                "confidence": trans_confidence,
                "status": trans_status,
                "has_line_bboxes": len(final_lines) > 0,
            })
        ordered_results.sort(key=lambda x: x["reading_order"])
        full_parts = []
        for r in ordered_results:
            if r.get("status") == "success" and r.get("text"):
                if r["region_type"] in ("TitleRegion", "HeadingRegion"):
                    full_parts.append(f"\n## {r['text']}\n")
                elif r["region_type"] == "TableRegion":
                    full_parts.append(f"\n{r['text']}\n")
                elif r["region_type"] in ("ImageRegion", "DiagramRegion", "FigureRegion"):
                    full_parts.append(f"\n[Image/Diagram: {r['text']}]\n")
                else:
                    full_parts.append(r["text"])
        full_text = "\n\n".join(full_parts)
        confidences = [r["confidence"] for r in ordered_results if r.get("confidence", 0) > 0]
        state.final_transcription = {
            "status": "success",
            "regions": ordered_results,
            "full_text": full_text,
            "total_regions": len(ordered_results),
            "total_regions_transcribed": len([r for r in ordered_results if r.get("status") == "success"]),
            "regions_with_line_bboxes": len([r for r in ordered_results if r.get("has_line_bboxes")]),
            "total_lines": sum(len(r.get("lines", [])) for r in ordered_results),
            "average_confidence": float(np.mean(confidences)) if confidences else 0,
        }
        summary = {
            "status": "success",
            "total_regions": len(ordered_results),
            "total_lines": state.final_transcription["total_lines"],
            "regions_with_lines": state.final_transcription["regions_with_line_bboxes"],
            "average_confidence": round(state.final_transcription["average_confidence"], 3),
            "preview": (full_text[:500] + "...") if len(full_text) > 500 else full_text,
        }
        return json.dumps(summary, indent=2)

    return [
        get_transcription_plan,
        transcribe_with_tesseract,
        transcribe_with_trocr,
        transcribe_with_llm,
        compile_transcription,
    ]
