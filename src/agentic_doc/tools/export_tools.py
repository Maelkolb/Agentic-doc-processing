"""Export tools: export_to_pagexml, export_to_markdown, export_to_html."""

import json
import os
from typing import Optional

from langchain_core.tools import tool

from ..export import write_html, write_markdown, write_pagexml


def get_export_tools(state, logger):
    def _get_base_name():
        source = state.current_image_path or "document"
        return os.path.splitext(os.path.basename(source))[0], source

    def _get_image_dimensions():
        if state.region_result:
            dims = state.region_result.get("image_dimensions", {})
            return dims.get("width", 0), dims.get("height", 0)
        return 0, 0

    @tool
    def export_to_pagexml(output_filename: Optional[str] = None) -> str:
        """Export the document to PAGE XML format (standard for layout + transcription). Run compile_transcription first."""
        logger.info("Exporting to PAGE XML")
        if not state.final_transcription:
            return json.dumps({"status": "error", "error": "No transcription. Run compile_transcription first."})
        base_name, source_image = _get_base_name()
        if not output_filename:
            output_filename = f"{base_name}_pagexml.xml"
        img_width, img_height = _get_image_dimensions()
        write_pagexml(
            state.final_transcription,
            output_filename,
            os.path.basename(source_image),
            img_width,
            img_height,
        )
        state.output_files["pagexml"] = output_filename
        regions = state.final_transcription.get("regions", [])
        return json.dumps({
            "status": "success",
            "format": "PAGE XML 2019",
            "output_path": output_filename,
            "regions_exported": len(regions),
            "total_lines": sum(len(r.get("lines", [])) for r in regions),
        }, indent=2)

    @tool
    def export_to_markdown(output_filename: Optional[str] = None, include_metadata: bool = True) -> str:
        """Export the document as a Markdown digital edition. Run compile_transcription first."""
        logger.info("Exporting to Markdown")
        if not state.final_transcription:
            return json.dumps({"status": "error", "error": "No transcription. Run compile_transcription first."})
        base_name, source_image = _get_base_name()
        if not output_filename:
            output_filename = f"{base_name}_edition.md"
        write_markdown(
            state.final_transcription,
            output_filename,
            source_image,
            state.assessment_result,
            include_metadata,
        )
        state.output_files["markdown"] = output_filename
        return json.dumps({
            "status": "success",
            "format": "Markdown Digital Edition",
            "output_path": output_filename,
            "sections": ["metadata", "layout", "transcription", "summary"] if include_metadata else ["transcription"],
        }, indent=2)

    @tool
    def export_to_html(output_filename: Optional[str] = None) -> str:
        """Export as interactive HTML with image overlay and region/line highlighting. Run compile_transcription first."""
        logger.info("Exporting to HTML")
        if not state.final_transcription:
            return json.dumps({"status": "error", "error": "No transcription. Run compile_transcription first."})
        base_name, source_image = _get_base_name()
        if not output_filename:
            output_filename = f"{base_name}_edition.html"
        img_width, img_height = _get_image_dimensions()
        write_html(
            state.final_transcription,
            output_filename,
            source_image,
            img_width,
            img_height,
        )
        state.output_files["html"] = output_filename
        regions = state.final_transcription.get("regions", [])
        return json.dumps({
            "status": "success",
            "format": "Interactive HTML Edition",
            "output_path": output_filename,
            "statistics": {"regions": len(regions), "lines": sum(len(r.get("lines", [])) for r in regions)},
        }, indent=2)

    return [export_to_pagexml, export_to_markdown, export_to_html]
