"""
GUI panel v12: document viewer with region/line overlay and agent log.
Designed for Jupyter/Colab: display(HTML(...)) then push state via GUIInterface.
Use run_with_gui() to run the agent with live GUI updates.
"""

import base64
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load full panel HTML from package resource
_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def _get_panel_html() -> str:
    path = _RESOURCES_DIR / "panel_v12.html"
    if not path.exists():
        return _fallback_minimal_html()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _fallback_minimal_html() -> str:
    """Minimal panel when template is missing."""
    return """
<style>
#docgui-panel { font-family: system-ui, sans-serif; padding: 16px; max-width: 900px; margin: 0 auto; }
#docgui-titlebar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; }
#docgui-log { background: #f8fafc; border-radius: 8px; padding: 12px; min-height: 120px; font-size: 13px; }
</style>
<div id="docgui-panel">
  <div id="docgui-titlebar">Document Processor — Agentic pipeline</div>
  <div id="docgui-log">Agent log will appear here when running in Jupyter/Colab with full GUI.</div>
</div>
<script>if (typeof window !== 'undefined') window.GUI = window.GUI || {};</script>
"""


def create_gui_panel_v12(display: bool = True) -> "GUIInterface":
    """
    Create and optionally display the GUI panel. Returns a GUIInterface instance
    that can push updates (load_document_image, set_assessment, set_regions, add_log, etc.).
    In Jupyter/Colab, set display=True to show the panel in the notebook.
    """
    html = _get_panel_html()
    if display:
        try:
            from IPython.display import HTML, display
            display(HTML(html))
            print("✅ GUI v12.0 panel created (with built-in region/line visualization)")
        except ImportError:
            pass
    gui = GUIInterface(html=html)
    gui._ready = display
    return gui


class GUIInterface:
    """Python interface to control the GUI v12.0 (image, regions, lines, log, outputs)."""

    def __init__(self, html: Optional[str] = None):
        self._ready = False
        self._html = html or _get_panel_html()

    @property
    def html(self) -> str:
        return self._html

    def _js(self, code: str) -> None:
        """Execute JavaScript in the notebook output (Jupyter/Colab only)."""
        if not self._ready:
            return
        try:
            from IPython.display import HTML, display
            # Escape </script> in code so it doesn't close the script tag
            safe_code = code.replace("</script>", r"<\/script>")
            display(HTML(f"<script>{safe_code}</script>"))
        except ImportError:
            pass

    def load_document_image(self, path: str) -> bool:
        """Load document image into the GUI viewer."""
        if not path or not os.path.exists(path):
            if path:
                print(f"   ⚠️ Image not found: {path}")
            return False
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            ext = (path.lower().split(".")[-1] if "." in path else "png")
            mime = "image/png" if ext == "png" else "image/jpeg"
            data_url = f"data:{mime};base64,{data}"
            # Pass data URL to JS; avoid breaking script by not embedding in quoted string if huge
            self._js(f"GUI.loadImage('{data_url}');")
            print(f"   ✅ Image loaded: {os.path.basename(path)}")
            return True
        except Exception as e:
            print(f"   ❌ Failed to load image: {e}")
            return False

    def set_regions(self, regions: List[Dict]) -> None:
        """Draw regions on the image (called when region_detection completes)."""
        if regions:
            js_data = json.dumps(regions)
            self._js(f"GUI.setRegions({js_data});")
            print(f"   📦 Regions drawn: {len(regions)} regions")

    def set_regions_with_lines(self, regions_with_lines: List[Dict]) -> None:
        """Draw regions with lines (alias for set_lines for compatibility)."""
        self.set_lines(regions_with_lines)

    def set_lines(self, regions_with_lines: List[Dict]) -> None:
        """Draw lines on the image (called when line_detection completes)."""
        if regions_with_lines:
            js_data = json.dumps(regions_with_lines)
            self._js(f"GUI.setLines({js_data});")
            total = sum(len(r.get("lines", [])) for r in regions_with_lines)
            print(f"   📏 Lines drawn: {total} lines")

    def set_phase(self, name: str, status: str = "active") -> None:
        self._js(f'GUI.setPhase("{name}", "{status}");')

    def complete_phase(self, name: str) -> None:
        self._js(f'GUI.completePhase("{name}");')

    def set_complete(self) -> None:
        self._js("GUI.setComplete();")

    def set_assessment(self, data: Optional[Dict[str, Any]]) -> None:
        if data:
            self._js(f"GUI.setAssessment({json.dumps(data)});")

    def add_log(
        self,
        level: str,
        content: str,
        timestamp: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> None:
        ts = timestamp or datetime.now().strftime("%H:%M:%S.%f")[:-3]
        safe = (
            str(content)[:1500]
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        tool_arg = f', "{tool_name}"' if tool_name else ", null"
        self._js(f'GUI.addLog("{level}", "{safe}", "{ts}"{tool_arg});')

    def clear_logs(self) -> None:
        self._js("GUI.clearLogs();")

    def set_output(self, output_type: str, content: str) -> None:
        if content:
            b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
            self._js(f'GUI.setOutput("{output_type}", decodeURIComponent(escape(atob("{b64}"))));')

    def set_output_html(self, content: str) -> None:
        self.set_output("html", content)

    def set_output_markdown(self, content: str) -> None:
        self.set_output("markdown", content)

    def set_output_pagexml(self, content: str) -> None:
        self.set_output("pagexml", content)

    def show_outputs(self) -> None:
        """Show the outputs modal (HTML/Markdown/PageXML)."""
        self._js("GUI.showOutputs();")


class GUILoggerAdapter:
    """Connects RichAgentLogger to the GUI so every log entry is also pushed to the panel."""

    def __init__(self, gui: GUIInterface, logger: Any):
        self.gui = gui
        self.logger = logger
        self._original_log = logger.log
        self._seen: set = set()
        self._last_tool = ""
        self._hook()

    def _hook(self) -> None:
        adapter = self

        def hooked_log(level: Any, content: Any, **kwargs: Any) -> Any:
            result = adapter._original_log(level, content, **kwargs)
            level_str = level.value if hasattr(level, "value") else str(level)
            h = hashlib.md5(f"{level_str}:{str(content)[:150]}".encode()).hexdigest()[:12]
            if h in adapter._seen:
                return result
            adapter._seen.add(h)
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if level_str in ("tool_start", "tool_end", "action"):
                match = re.search(r"(\w+)", str(content))
                if match:
                    adapter._last_tool = match.group(1)
            if level_str == "phase_start":
                phase = kwargs.get("metadata", {}).get("phase_name", content)
                adapter.gui.set_phase(phase, "active")
                adapter.gui.add_log("phase_start", f"Starting: {phase}", ts)
            elif level_str == "phase_end":
                adapter.gui.complete_phase(content)
                adapter.gui.add_log("phase_end", f"Done: {content}", ts)
            else:
                adapter.gui.add_log(level_str, str(content)[:1200], ts, adapter._last_tool)
            return result

        self.logger.log = hooked_log


def run_with_gui(
    agent_executor: Any,
    state: Any,
    logger: Any,
    image_path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    task_prompt: Optional[str] = None,
    callbacks: Optional[List] = None,
) -> Any:
    """
    Run the agent with the GUI: display panel, hook logger to GUI, stream agent events,
    and push state updates (image, assessment, regions, lines, outputs) to the panel.
    Use this in Jupyter/Colab for live visualization.
    """
    from langchain_core.messages import HumanMessage

    config = dict(config or {})
    if "configurable" not in config:
        config["configurable"] = {}
    if callbacks is not None:
        config["configurable"]["callbacks"] = callbacks
    elif not config["configurable"].get("callbacks") and hasattr(logger, "log"):
        try:
            from ..agent.callbacks import StreamingAgentCallback
            config["configurable"]["callbacks"] = [StreamingAgentCallback(logger)]
        except Exception:
            pass

    state.reset()
    logger.clear()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    gui = create_gui_panel_v12(display=True)
    time.sleep(0.5)

    GUILoggerAdapter(gui, logger)

    gui.load_document_image(image_path)
    gui.clear_logs()

    prompt = task_prompt or (
        f"Process the historical document at: {image_path}\n\n"
        "Execute the complete layout analysis pipeline:\n"
        "1. Assess the document\n2. Enhance if needed\n3. Detect and classify regions\n"
        "4. Detect lines within regions\n5. Transcribe all text\n"
        "6. Generate outputs (PageXML, Markdown, HTML)\n\n"
        "Provide detailed reasoning at each step."
    )

    logger.phase_start("initialization")
    final_result = None
    _enhanced_sent = False
    _assessment_sent = False
    _regions_sent = False
    _lines_sent = False

    try:
        # Prefer stream() for incremental GUI updates; fallback to invoke()
        if hasattr(agent_executor, "stream"):
            for event in agent_executor.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            ):
                final_result = event
                if not _enhanced_sent and getattr(state, "enhanced_path", None):
                    if os.path.exists(state.enhanced_path):
                        gui.load_document_image(state.enhanced_path)
                        _enhanced_sent = True
                if not _assessment_sent and getattr(state, "assessment_result", None):
                    gui.set_assessment(state.assessment_result)
                    _assessment_sent = True
                if not _regions_sent and getattr(state, "region_result", None):
                    regions = state.region_result.get("regions", [])
                    if regions:
                        gui.set_regions(regions)
                        _regions_sent = True
                if not _lines_sent and getattr(state, "line_result", None):
                    regions_with_lines = state.line_result.get("regions", [])
                    if regions_with_lines:
                        gui.set_lines(regions_with_lines)
                        _lines_sent = True
        else:
            final_result = agent_executor.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )
            if getattr(state, "enhanced_path", None) and os.path.exists(state.enhanced_path):
                gui.load_document_image(state.enhanced_path)
            if getattr(state, "assessment_result", None):
                gui.set_assessment(state.assessment_result)
            regions = getattr(state, "region_result", None) or {}
            if regions.get("regions"):
                gui.set_regions(regions["regions"])
            line_result = getattr(state, "line_result", None) or {}
            if line_result.get("regions"):
                gui.set_lines(line_result["regions"])
    finally:
        logger.phase_end()
        gui.set_complete()

    output_files = getattr(state, "output_files", None) or {}
    for key, method in [
        ("html", gui.set_output_html),
        ("markdown", gui.set_output_markdown),
        ("pagexml", gui.set_output_pagexml),
    ]:
        path = output_files.get(key)
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    method(f.read())
                print(f"   📄 {key.upper()} output loaded into GUI")
            except Exception as e:
                print(f"   ⚠️ Could not load {key} output: {e}")

    return final_result
