"""Build the document processing agent: LLM, tools, prompt, runnable."""

from typing import Any, Dict, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import load_config
from ..detection import (
    DocumentAssessor,
    ImageEnhancer,
    LayoutVisualizer,
    LineDetector,
    RegionDetector,
)
from ..logging_utils import RichAgentLogger
from ..state import ProcessingState
from ..tools import get_tools
from ..transcription import LLMTranscriber, TesseractOCR, TrOCRHTR

from .callbacks import StreamingAgentCallback
from .prompt import SYSTEM_PROMPT


def build_agent(
    config: Optional[Dict[str, Any]] = None,
    state: Optional[ProcessingState] = None,
    logger: Optional[RichAgentLogger] = None,
) -> Tuple[Any, ProcessingState, RichAgentLogger]:
    """
    Build the document processing agent runnable with injected state and logger.
    Returns (agent_runnable, state, logger).
    """
    cfg = config or load_config()
    api_key = cfg.get("GOOGLE_API_KEY") or ""
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Set it in env or pass config with GOOGLE_API_KEY.")
    agent_model = cfg.get("AGENT_MODEL", "gemini-2.0-flash")
    vision_model = cfg.get("VISION_MODEL", "gemini-2.0-flash-preview-05-20")
    use_layout_fallback = cfg.get("USE_LAYOUT_FALLBACK", False)

    if state is None:
        state = ProcessingState()
    if logger is None:
        logger = RichAgentLogger(verbose=True)

    # Google GenAI client for vision (assessor, region detector, LLM transcriber)
    from google import genai
    gemini_client = genai.Client(api_key=api_key)

    document_assessor = DocumentAssessor(gemini_client, vision_model)
    image_enhancer = ImageEnhancer()
    region_detector = RegionDetector(gemini_client, vision_model)
    line_detector = LineDetector(use_layout_fallback=use_layout_fallback)
    visualizer = LayoutVisualizer()
    tesseract_ocr = TesseractOCR()
    trocr_htr = TrOCRHTR()
    llm_transcriber = LLMTranscriber(gemini_client, vision_model)

    tools = get_tools(
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
    )

    llm = ChatGoogleGenerativeAI(
        model=agent_model,
        temperature=0.1,
        google_api_key=api_key,
    )

    # Prefer create_agent from langchain.agents (system_prompt=), fallback to langgraph create_react_agent (prompt=)
    try:
        from langchain.agents import create_agent
        agent_executor = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    except (ImportError, TypeError):
        from langgraph.prebuilt import create_react_agent
        agent_executor = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    return agent_executor, state, logger
