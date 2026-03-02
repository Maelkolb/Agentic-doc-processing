"""
GUI panel v12: document viewer with region/line overlay and agent log.
Designed for Jupyter: display(HTML(create_gui_panel_v12().html)) then push state via GUI.*
"""

import base64
import json
from typing import Any, Dict, List, Optional


def create_gui_panel_v12():
    """
    Create the GUI panel object. In Jupyter, display its .html then use the global GUI
    object to push updates: load_document_image, set_assessment, set_regions, set_regions_with_lines, addLog, showOutputs.
    """
    html = _build_panel_html()
    try:
        from IPython.display import display, HTML
        display(HTML(html))
    except ImportError:
        pass

    class GuiPanel:
        def __init__(self):
            self._html = html

        @property
        def html(self) -> str:
            return self._html

        def load_document_image(self, path: str) -> None:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                ext = path.lower().split(".")[-1] if "." in path else "png"
                mime = "image/png" if ext == "png" else "image/jpeg"
                self.addLog("info", f"Loaded image: {path}", "gui")
                # In full GUI, would inject into img src and redraw
            except Exception as e:
                self.addLog("error", str(e), "gui")

        def set_assessment(self, result: Dict[str, Any]) -> None:
            self.addLog("info", "Assessment updated", "gui")

        def set_regions(self, regions: List[Dict]) -> None:
            self.addLog("info", f"Regions updated: {len(regions)}", "gui")

        def set_regions_with_lines(self, regions: List[Dict]) -> None:
            self.addLog("info", f"Regions with lines: {len(regions)}", "gui")

        def addLog(self, level: str, content: str, source: str = "") -> None:
            # In Jupyter with full GUI, this would append to #docgui-log via JS
            pass

        def showOutputs(self) -> None:
            pass

    return GuiPanel()


def _build_panel_html() -> str:
    """Minimal panel HTML structure; can be replaced with full v12 HTML from notebook."""
    return """
<style>
#docgui-panel { font-family: system-ui, sans-serif; padding: 16px; max-width: 900px; margin: 0 auto; }
#docgui-titlebar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; }
#docgui-phases { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.phase-dot { width: 24px; height: 24px; border-radius: 50%; background: #e2e8f0; }
#docgui-log { background: #f8fafc; border-radius: 8px; padding: 12px; min-height: 120px; font-size: 13px; }
</style>
<div id="docgui-panel">
  <div id="docgui-titlebar">Document Processor — Agentic pipeline</div>
  <div id="docgui-phases">
    <span class="phase-dot" title="Init"></span>
    <span class="phase-dot" title="Assess"></span>
    <span class="phase-dot" title="Enhance"></span>
    <span class="phase-dot" title="Regions"></span>
    <span class="phase-dot" title="Lines"></span>
    <span class="phase-dot" title="Transcribe"></span>
    <span class="phase-dot" title="Export"></span>
  </div>
  <div id="docgui-log">Agent log will appear here when running in Jupyter with full GUI.</div>
</div>
<script>
if (typeof window !== 'undefined') window.GUI = window.GUI || {};
</script>
"""
