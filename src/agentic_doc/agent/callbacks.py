"""Callback handler for agent logging (thought, tool start/end, phase)."""

import json
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler

from ..logging_utils import RichAgentLogger


class StreamingAgentCallback(BaseCallbackHandler):
    """Callback that logs agent reasoning, tool calls, and phases to RichAgentLogger."""

    def __init__(self, logger: RichAgentLogger):
        self.logger = logger
        self.tool_start_times = {}
        self._in_tool_call = False
        self._last_content = ""

    def on_llm_start(self, serialized=None, prompts=None, **kwargs):
        if not self._in_tool_call:
            self.logger.info("Agent reasoning...")

    def on_llm_end(self, response, **kwargs):
        try:
            if not getattr(response, "generations", None) or not response.generations:
                return
            for gen in response.generations:
                for g in gen:
                    thought_text = None
                    if hasattr(g, "text") and g.text:
                        thought_text = g.text
                    if not thought_text and hasattr(g, "message"):
                        msg = g.message
                        if hasattr(msg, "content"):
                            if isinstance(msg.content, str):
                                thought_text = msg.content
                            elif isinstance(msg.content, list):
                                texts = []
                                for item in msg.content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        texts.append(item.get("text", ""))
                                    elif isinstance(item, str):
                                        texts.append(item)
                                thought_text = "\n".join(texts)
                    if thought_text and len(thought_text.strip()) > 15:
                        stripped = thought_text.strip()
                        if stripped.startswith("{") and stripped.endswith("}"):
                            return
                        if thought_text == self._last_content:
                            return
                        self._last_content = thought_text
                        lower_text = thought_text.lower()
                        if any(kw in lower_text for kw in ["i will", "let me", "first", "next", "now i", "thinking", "step", "because", "since"]):
                            self.logger.reasoning(thought_text[:1500])
                        else:
                            self.logger.thought(thought_text[:1500])
        except Exception:
            pass

    def on_tool_start(self, serialized=None, input_str=None, **kwargs):
        self._in_tool_call = True
        tool_name = serialized.get("name", "unknown") if serialized else "unknown"
        phase_map = {
            "assess_document": "assessment",
            "enhance_image": "enhancement",
            "detect_regions": "region_detection",
            "detect_lines": "line_detection",
            "visualize_layout": "visualization",
            "get_transcription_plan": "planning",
            "transcribe_with_tesseract": "transcription",
            "transcribe_with_trocr": "transcription",
            "transcribe_with_llm": "transcription",
            "compile_transcription": "compilation",
            "export_to_pagexml": "export",
            "export_to_markdown": "export",
            "export_to_html": "export",
        }
        new_phase = phase_map.get(tool_name)
        if new_phase and new_phase != self.logger.current_phase:
            self.logger.phase_end()
            self.logger.phase_start(new_phase)
        self.tool_start_times[tool_name] = self.logger.tool_start(tool_name)
        try:
            args = json.loads(input_str) if isinstance(input_str, str) else input_str
            self.logger.action(tool_name, args)
        except Exception:
            self.logger.action(tool_name, {"raw": str(input_str)[:200] if input_str else ""})

    def on_tool_end(self, output, **kwargs):
        self._in_tool_call = False
        if self.tool_start_times:
            tool_name = list(self.tool_start_times.keys())[-1]
            start_time = self.tool_start_times.pop(tool_name, datetime.now())
            self.logger.tool_end(tool_name, start_time)

    def on_tool_error(self, error, **kwargs):
        self._in_tool_call = False
        self.logger.error(f"Tool error: {error}")
