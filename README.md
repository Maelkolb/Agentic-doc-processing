# Agentic Document Processing

A LangChain / LangGraph pipeline that converts a document image into PageXML, Markdown, and HTML. The agent picks the right OCR / HTR tool per region (Tesseract, TrOCR, or Gemini vision) based on a layout pass with Surya plus Gemini.

## Pipeline

1. Assess image quality (OpenCV) and content (Gemini vision).
2. Optionally enhance the image — deskew, denoise, contrast.
3. Detect regions: paragraphs, headings, tables, marginalia (Gemini vision).
4. Detect text-line polygons inside each region (Surya `DetectionPredictor`).
5. Transcribe each region with Tesseract, TrOCR, or Gemini vision.
6. Export to PAGE XML 2019, Markdown, and an interactive HTML viewer.

## Install

```bash
git clone https://github.com/Maelkolb/Agentic-doc-processing.git
cd Agentic-doc-processing
pip install -e ".[tesseract,trocr]"
```

Surya 0.17.x requires `transformers >= 4.56.1, < 5`. The pin is in `pyproject.toml`, but if Surya returns zero or one line you are probably on `transformers 5.x`. Fix it and restart the runtime:

```bash
pip install 'transformers>=4.56.1,<5'
```

## API key

A Gemini API key is required. The config layer accepts it from any of the sources below, in order:

| Source | How to set |
| --- | --- |
| `GEMINI_API_KEY` env var | `os.environ["GEMINI_API_KEY"] = "..."` |
| Colab secret `GEMINI_API_KEY` or `GOOGLE_API_KEY` | left sidebar 🔑 |
| Explicit argument | `build_agent(api_key="...")` |


## CLI

```bash
python main.py path/to/document.png
python main.py path/to/document.png --no-gui
```

| Flag | Default | Description |
| --- | --- | --- |
| `image_path` | required | Path to the document image |
| `--no-gui` | off | Run headless — no Jupyter / Colab panel |
| `--no-callbacks` | off | Disable streaming log callbacks |

## Colab / Jupyter

```python
import os
os.environ["GEMINI_API_KEY"] = "..."   # or use a Colab secret

from agentic_doc.agent import build_agent
from agentic_doc.gui import run_with_gui

agent, state, logger = build_agent()
run_with_gui(agent, state, logger, "path/to/document.png")
```

Headless inside a notebook:

```python
from langchain_core.messages import HumanMessage
from agentic_doc.agent import build_agent
from agentic_doc.agent.callbacks import StreamingAgentCallback

agent, state, logger = build_agent()
result = agent.invoke(
    {"messages": [HumanMessage(content=f"Process this document image completely: {image_path}")]},
    config={"configurable": {"callbacks": [StreamingAgentCallback(logger)]}},
)
```

## Configuration

| Env variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key (required) |
| `COLAB_GEMINI_SECRET` | — | Name of the Colab secret holding the key |
| `AGENT_MODEL` | `gemini-2.5-flash` | Model for agent tool calling |
| `VISION_MODEL` | `gemini-3-flash-preview` | Model for vision tasks |
| `USE_LAYOUT_FALLBACK` | `false` | Use the fallback layout detector if Surya fails |

## Project layout

```
src/agentic_doc/
  config.py          API key + model selection
  state.py           Pipeline state container
  logging_utils.py   Rich-based logger
  detection/         Quality assessment, image enhancement, region + line detection
  transcription/     Tesseract, TrOCR, Gemini-vision transcribers
  export/            PageXML 2019, Markdown, HTML writers
  tools/             LangChain tool wrappers
  agent/             System prompt, callbacks, build_agent
  gui/               Panel-based Jupyter / Colab viewer
main.py              CLI entry point
tests/               pytest suite
```

## Tests

```bash
pytest tests/ -v
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Surya returns 0 or 1 line | `pip install 'transformers>=4.56.1,<5'`, then restart the runtime |
| `ModuleNotFoundError: agentic_doc` | `pip install -e .` or `sys.path.insert(0, "src")` |
| `GOOGLE_API_KEY not set` | See the **API key** section above |
| `Both GOOGLE_API_KEY and GEMINI_API_KEY are set` warning | Update to the current `config.py` — only one env var is left set after `load_config()` runs |
| TrOCR runs out of memory | The agent automatically falls back to `transcribe_with_llm` |

## License

Each bundled model has its own license (Surya, TrOCR). Check those before deploying.
