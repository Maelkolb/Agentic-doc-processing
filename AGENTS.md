# AGENTS.md

## Project Overview

This repository contains a single Jupyter notebook (`Agent_Doc_Processing_2026_02_V1 (1).ipynb`) implementing an **Agentic Historical Document Processing System**. It uses a LangGraph ReAct agent with Google Gemini models for historical manuscript image analysis, layout detection, OCR/HTR transcription, and structured output generation.

## Cursor Cloud specific instructions

### Required Secret

- **`GOOGLE_API_KEY`**: A valid Google Gemini API key is required for all processing. The notebook checks `GEMINI_API_KEY` (Colab), `gemini-api`, and `GOOGLE_API_KEY` environment variables. Without this key, only the import and class-definition cells can execute; the agent pipeline will fail.

### Running the Notebook

```bash
export PATH="$HOME/.local/bin:$PATH"
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''
```

Or run cells programmatically:

```bash
jupyter nbconvert --to notebook --execute "Agent_Doc_Processing_2026_02_V1 (1).ipynb"
```

### Key Gotchas

- **Colab-specific cells**: Cell 28 uses `google.colab.files.upload()` which only works in Colab. When running locally, set `image_path` manually to a local image file path before running cell 29.
- **Surya version pinning**: The notebook requires `surya-ocr==0.17.0` exactly. Newer versions have breaking API changes (`DetectionPredictor` interface).
- **Tesseract is optional**: The system package `tesseract-ocr` enables the Tesseract transcription tool, but the LLM transcriber (`transcribe_with_llm`) is an alternative for all text types.
- **No GPU required**: All models run on CPU (slower but functional). Surya and TrOCR will auto-use GPU if CUDA is available.
- **First run downloads models**: HuggingFace models (TrOCR, Surya) download on first use (~500MB+ total). Subsequent runs use the cache.
- **matplotlib backend**: In headless environments, set `matplotlib.use('Agg')` before importing `pyplot`. The notebook's visualization tools generate images saved to files, not displayed inline.
- **`~/.local/bin` must be on PATH**: Jupyter and other pip-installed scripts install to `~/.local/bin`.

### Linting

No linter is configured for this notebook. Standard Python linting can be applied to extracted cells:

```bash
pip install ruff
jupyter nbconvert --to script "Agent_Doc_Processing_2026_02_V1 (1).ipynb" && ruff check "Agent_Doc_Processing_2026_02_V1 (1).py"
```

### Testing

There are no automated tests. Verification is done by:
1. Running all import/class-definition cells (cells 2–25) — should complete without errors.
2. With a valid `GOOGLE_API_KEY`, running the full pipeline on a sample historical document image.
