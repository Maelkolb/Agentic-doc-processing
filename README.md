# Agentic Document Processing

LangChain/LangGraph-powered document processing pipeline: assessment, region detection, **line detection (Surya)**, OCR/HTR (Tesseract, TrOCR, Gemini vision), and export (PageXML, Markdown, HTML).

## Features

- **Assessment**: CV quality metrics + Gemini vision content analysis (script, language, layout).
- **Preprocessing**: Deskew, denoise, contrast enhancement (OpenCV).
- **Layout**: Region detection (Gemini vision), **line detection (Surya)** with dict/object API compatibility and optional Layout fallback.
- **Transcription**: Tesseract (printed), TrOCR (handwriting/Kurrent), Gemini vision (tables, complex layout).
- **Export**: PAGE XML 2019, Markdown digital edition, interactive HTML.

## Setup

1. **Clone and install**

   ```bash
   cd Agentic-doc-processing
   pip install -e .
   # Optional: Tesseract/TrOCR
   pip install -e ".[tesseract,trocr]"
   ```

2. **Environment**

   Copy `.env.example` to `.env` and set:

   - `GOOGLE_API_KEY` (required for Gemini)

   Optional: `USE_LAYOUT_FALLBACK=true` to use Surya Layout when line detection returns no bboxes; `DETECTOR_BATCH_SIZE`, `DETECTOR_BLANK_THRESHOLD`, `DETECTOR_TEXT_THRESHOLD` for Surya tuning.

3. **Run**

   ```bash
   python main.py path/to/document.png
   python main.py path/to/document.png --no-gui
   ```

   With GUI (default): the panel is shown and updated live in **Jupyter/Colab**; from a plain CLI the pipeline still runs (GUI updates no-op unless in an IPython context).

   **Colab / Jupyter with full GUI**

   ```python
   from agentic_doc.agent import build_agent
   from agentic_doc.gui import run_with_gui

   agent, state, logger = build_agent()
   run_with_gui(agent, state, logger, "/path/to/document.png")
   ```

   This displays the v12 panel (document preview, region/line overlay, phase indicators, agent log) and streams agent events so the GUI updates as assessment, regions, lines, and outputs become available. Use **View Outputs** in the panel to open HTML/Markdown/PageXML.

   Or use as a package (headless / custom UI):

   ```python
   from agentic_doc.agent import build_agent
   from agentic_doc.agent.callbacks import StreamingAgentCallback
   from langchain_core.messages import HumanMessage

   agent, state, logger = build_agent()
   result = agent.invoke(
       {"messages": [HumanMessage(content="Process the document at: /path/to/image.png")]},
       config={"configurable": {"callbacks": [StreamingAgentCallback(logger)]}},
   )
   ```

## Line detection (Surya)

Line detection uses Surya’s `DetectionPredictor`. The code **normalizes both dict and object** API responses (e.g. `surya-ocr` 0.17.x and newer), so you don’t need to pin an old version. If detection returns no or very few bboxes, set `USE_LAYOUT_FALLBACK=true` to derive line-level bboxes from Surya’s Layout model.

- **Env vars**: `DETECTOR_BATCH_SIZE`, `DETECTOR_BLANK_THRESHOLD`, `DETECTOR_TEXT_THRESHOLD` (see Surya README).
- **Tests**: `pytest tests/test_line_detector.py`

## Project layout

```
src/agentic_doc/
  config.py          # API key, model names
  state.py           # ProcessingState
  logging_utils.py   # RichAgentLogger
  detection/         # assessor, region_detector, line_detector, image_enhancer, visualizer
  transcription/     # tesseract_ocr, trocr, llm_transcriber
  export/            # pagexml, markdown, html_export
  tools/             # LangChain tools (analysis, layout, transcription, export)
  agent/             # prompt, callbacks, build_agent
  gui/               # panel (create_gui_panel_v12, GUIInterface, GUILoggerAdapter, run_with_gui)
  gui/resources/    # panel_v12.html (full HTML/JS/CSS)
main.py              # CLI entry
tests/test_line_detector.py
```

## Original notebook

The pipeline was extracted from the Jupyter notebook `Agent_Doc_Processing_2026_02_V1 (1).ipynb`. You can keep a thin runner notebook that imports `agentic_doc` and runs the agent + GUI in Colab.
