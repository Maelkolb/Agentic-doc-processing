"""LLM-based transcription using Gemini vision; processes full regions."""

import io
from typing import Any, Dict, List

from PIL import Image


class LLMTranscriber:
    """Region-level transcription using Gemini vision."""

    def __init__(self, client: Any, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def transcribe_region(
        self,
        image: Image.Image,
        region_type: str = "TextRegion",
        context: str = "",
        output_format: str = "text",
        lines: List[Dict] = None,
    ) -> Dict[str, Any]:
        from google.genai import types

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        if output_format == "description" or region_type in ("ImageRegion", "DiagramRegion", "FigureRegion"):
            prompt = f"""Describe this image/diagram.
{f'Context: {context}' if context else ''}
Provide a clear description of what is shown."""
        elif output_format == "markdown" or region_type == "TableRegion":
            prompt = f"""Transcribe this table from a historical document into Markdown format.
{f'Context: {context}' if context else ''}
Use proper Markdown table syntax with | and -.
If text is unclear, use [unclear] or your best interpretation with [?].
Output ONLY the Markdown table."""
        else:
            prompt = f"""Transcribe the text in this historical document image exactly as written.
Region type: {region_type}
{f'Context: {context}' if context else ''}

Rules:
- Transcribe exactly what you see, character by character
- Preserve line breaks as they appear in the original
- Use [?] for uncertain characters
- Use [illegible] for completely unreadable words
- Do NOT add interpretations or translations

Output ONLY the transcription, nothing else."""

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            text = response.text.strip()
            return {"status": "success", "text": text, "confidence": 0.8}
        except Exception as e:
            return {"status": "error", "error": str(e), "text": "", "confidence": 0}
