"""LangChain tools for the document processing agent. Built with state and logger injection."""

from .analysis import get_analysis_tools
from .layout import get_layout_tools
from .transcription_tools import get_transcription_tools
from .export_tools import get_export_tools


def get_tools(
    state,
    logger,
    document_assessor,
    image_enhancer,
    region_detector,
    line_detector,
    visualizer,
    tesseract_ocr,
    trocr_htr,
    llm_transcriber,
):
    """Build all tools with injected state, logger, and services."""
    tools = []
    tools.extend(
        get_analysis_tools(state, logger, document_assessor, image_enhancer)
    )
    tools.extend(
        get_layout_tools(state, logger, region_detector, line_detector, visualizer)
    )
    tools.extend(
        get_transcription_tools(
            state, logger, tesseract_ocr, trocr_htr, llm_transcriber
        )
    )
    tools.extend(get_export_tools(state, logger))
    return tools
