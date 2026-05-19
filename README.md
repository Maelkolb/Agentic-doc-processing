# Agentic Document Processing

A LangChain / LangGraph-powered pipeline that turns a document image into structured output (PageXML, Markdown, HTML). An AI agent drives every step — assessment, layout analysis, OCR/HTR, and export — choosing the right tools automatically.

## What it does

1. **Assess** the document (quality metrics via CV, content analysis via Gemini vision).
2. **Enhance** the image if needed (deskew, denoise, contrast — OpenCV).
3. **Detect regions** (paragraphs, headings, tables, marginalia — Gemini vision).
4. **Detect text lines** inside each region (Surya DetectionPredictor).
5. **Transcribe** every region with the best tool (Tesseract for print, TrOCR for handwriting, Gemini vision for complex layouts / Kurrent / tables).
6. **Export** to PAGE XML 2019, Markdown digital edition, and interactive HTML.

## Quick start — Google Colab

### 1. Install

```python
# Clone the repo (or upload the zip and unzip it)
!git clone https://github.com/Maelkolb/Agentic-doc-processing.git
%cd Agentic-doc-processing

# Install the package
!pip install -e ".[tesseract,trocr]"

# Ensure importable
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
```

> **Important:** The package pins `transformers>=4.56.1,<5` because Surya 0.17.x is incompatible with `transformers 5.x`. If Colab pre-installs `transformers 5.x`, the `pip install -e .` will automatically downgrade it. If you see Surya returning 0 lines, run `pip install 'transformers>=4.56.1,<5'` and **restart the runtime**.

### 2. Set your Gemini API key

Add a Colab Secret named `GEMINI_API_KEY` (sidebar 🔑), or:

```python
os.environ["GOOGLE_API_KEY"] = "your-key-here"
```

### 3. Upload an image and run

```python
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
```

**With GUI:**
```python
from agentic_doc.agent import build_agent
from agentic_doc.gui import run_with_gui

agent, state, logger = build_agent()
run_with_gui(agent, state, logger, image_path)
```

**Headless:**
```python
from langchain_core.messages import HumanMessage
from agentic_doc.agent import build_agent
from agentic_doc.agent.callbacks import StreamingAgentCallback

agent, state, logger = build_agent()
result = agent.invoke(
    {"messages": [HumanMessage(content=(
        f"Process this document image completely: {image_path}\n\n"
        "Follow the full pipeline: assess, enhance if recommended, detect regions, "
        "detect lines, get transcription plan, transcribe every text region "
        "(use transcribe_with_llm for tables and images), compile transcription, "
        "then export to PageXML, Markdown, and HTML."
    ))]},
    config={"configurable": {"callbacks": [StreamingAgentCallback(logger)]}},
)
```

## Quick start — Local (VS Code / CLI)

```bash
git clone https://github.com/YOUR_USER/Agentic-doc-processing.git
cd Agentic-doc-processing
pip install -e ".[tesseract,trocr]"
export GOOGLE_API_KEY="your-gemini-key"

# CLI
python main.py path/to/document.png
python main.py path/to/document.png --no-gui
```

## Surya line detection

Surya's `DetectionPredictor` detects text line polygons within each region. For each region the detector crops from the full-page image, runs inference on the crop, and maps coordinates back to full-image space.

### Critical: transformers version

Surya 0.17.x requires `transformers>=4.56.1,<5`. With `transformers 5.x` the model loads but outputs garbage (zero or one bbox for the entire image). This is pinned correctly in `pyproject.toml`. If your environment has `transformers 5.x` already installed, run:

```bash
pip install 'transformers>=4.56.1,<5'
```

and **restart your Python runtime**.

### Tuning

| Variable | Default | Effect |
|----------|---------|--------|
| `DETECTOR_BATCH_SIZE` | `36` (GPU) | GPU VRAM vs speed |
| `DETECTOR_BLANK_THRESHOLD` | `0.35` | Lower → more sensitive to gaps between lines |
| `DETECTOR_TEXT_THRESHOLD` | `0.6` | Higher → lines merge less |

## Configuration

| Env variable | Default | Purpose |
|-------------|---------|---------|
| `GOOGLE_API_KEY` | — | **Required.** Gemini API key |
| `AGENT_MODEL` | `gemini-2.0-flash` | LLM for agent tool calling |
| `VISION_MODEL` | `gemini-3-flash-preview` | LLM for vision tasks |

## Project layout

```
src/agentic_doc/
  config.py             # API key, model names
  state.py              # ProcessingState
  logging_utils.py      # RichAgentLogger
  utils.py              # MIME mapping, JSON cleaning, skew detection
  detection/
    assessor.py          # Document quality + content analysis
    image_enhancer.py    # Deskew, denoise, contrast (OpenCV)
    region_detector.py   # Gemini vision → region bboxes + types
    line_detector.py     # Surya DetectionPredictor
    visualizer.py        # Matplotlib region/line overlay
  transcription/
    tesseract_ocr.py     # Tesseract OCR (printed text)
    trocr.py             # TrOCR HTR (handwriting)
    llm_transcriber.py   # Gemini vision transcription
  export/
    pagexml.py           # PAGE XML 2019 writer
    markdown.py          # Markdown digital edition
    html_export.py       # Interactive HTML with overlays
  tools/                 # LangChain tools (analysis, layout, transcription, export)
  agent/                 # System prompt, callbacks, build_agent
  gui/                   # Panel (Jupyter/Colab)
main.py                  # CLI entry point
tests/test_line_detector.py
```

## Tests

```bash
pytest tests/ -v
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Surya returns 0 or 1 line | `pip install 'transformers>=4.56.1,<5'` then restart runtime |
| `ModuleNotFoundError: agentic_doc` | `pip install -e .` or add `sys.path.insert(0, "src")` |
| `GOOGLE_API_KEY not set` | Set via env var or Colab Secrets |
| GUI doesn't show | Run `run_with_gui()` in a notebook cell, not a `.py` script |
| TrOCR out of memory | Agent falls back to `transcribe_with_llm` automatically |

## License

See individual model licenses (Surya, TrOCR) for model weight usage terms.
