"""Processing state shared across tools."""

from typing import Any, Dict, Optional


class ProcessingState:
    """Holds state across processing steps."""

    def __init__(self) -> None:
        self.image_path: Optional[str] = None
        self.enhanced_path: Optional[str] = None
        self.current_image_path: Optional[str] = None
        self.assessment_result: Optional[Dict[str, Any]] = None
        self.region_result: Optional[Dict[str, Any]] = None
        self.line_result: Optional[Dict[str, Any]] = None
        self.output_files: Dict[str, str] = {}
        self.transcription_results: Dict[str, Dict[str, Any]] = {}
        self.final_transcription: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self.image_path = None
        self.enhanced_path = None
        self.current_image_path = None
        self.assessment_result = None
        self.region_result = None
        self.line_result = None
        self.output_files = {}
        self.transcription_results = {}
        self.final_transcription = None
