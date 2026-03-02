"""Export final transcription to interactive HTML with image overlay."""

import base64
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from PIL import Image


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_html(
    final_transcription: Dict[str, Any],
    output_filename: str,
    source_image_path: str,
    image_width: int = 0,
    image_height: int = 0,
) -> str:
    """
    Write interactive HTML with side-by-side image (with SVG overlays) and transcription.
    Returns output_filename.
    """
    base_name = os.path.basename(source_image_path)
    base_name = os.path.splitext(base_name)[0] if "." in base_name else base_name

    # Encode image for inline display
    try:
        img = Image.open(source_image_path).convert("RGB")
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        img_w, img_h = img.size
    except Exception:
        b64 = ""
        img_w = image_width or 800
        img_h = image_height or 600

    regions: List[Dict] = final_transcription.get("regions", [])
    scale = 1.0

    # SVG overlay for regions and lines
    svg_parts = []
    for region in regions:
        bbox = region.get("bbox", {})
        x, y = bbox.get("x", 0), bbox.get("y", 0)
        w, h = bbox.get("width", 0), bbox.get("height", 0)
        rid = region.get("region_id", "")
        rtype = region.get("region_type", "TextRegion")
        color = "#3498db" if "Table" in rtype else "#2980b9"
        svg_parts.append(
            f'<rect class="region-g" data-rid="{rid}" x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for line in region.get("lines", []):
            lid = line.get("line_id", "")
            poly = line.get("polygon", [])
            if poly:
                pts = " ".join(f"{p[0]},{p[1]}" for p in poly)
                svg_parts.append(f'<polygon class="line-poly" data-lid="{lid}" data-rid="{rid}" points="{pts}" fill="none" stroke="#00ff00" stroke-width="1" stroke-dasharray="4"/>')
            else:
                lb = line.get("bbox", {})
                lx, ly = lb.get("x", 0), lb.get("y", 0)
                lw, lh = lb.get("width", 0), lb.get("height", 0)
                svg_parts.append(
                    f'<rect class="line-poly" data-lid="{lid}" data-rid="{rid}" x="{lx}" y="{ly}" width="{lw}" height="{lh}" '
                    'fill="none" stroke="#00ff00" stroke-width="1" stroke-dasharray="4"/>'
                )

    svg_content = "\n".join(svg_parts)

    # Transcription HTML
    trans_parts = []
    for region in regions:
        rid = region.get("region_id", "")
        rtype = region.get("region_type", "TextRegion")
        trans_parts.append(f'<div class="read-region" data-rid="{rid}">')
        trans_parts.append(f'<div class="region-type">{_escape_html(rtype)}</div>')
        if rtype == "TableRegion" and region.get("text", "").strip().startswith("|"):
            trans_parts.append("<table>")
            lines = [l.strip() for l in region["text"].strip().split("\n") if l.strip()]
            for i, row in enumerate(lines):
                cells = [c.strip() for c in row.split("|") if c.strip()]
                tag = "th" if i == 0 and row.startswith("|") else "td"
                trans_parts.append("<tr>")
                for c in cells:
                    trans_parts.append(f"<{tag}>{_escape_html(c)}</{tag}>")
                trans_parts.append("</tr>")
            trans_parts.append("</table>")
        else:
            for line in region.get("lines", []):
                lid = line.get("line_id", "")
                text = line.get("text", "")
                trans_parts.append(f'<span class="line-span" data-lid="{lid}" data-rid="{rid}">{_escape_html(text)}</span>')
            if not region.get("lines") and region.get("text"):
                trans_parts.append(f'<p>{_escape_html(region["text"])}</p>')
        trans_parts.append("</div>")

    trans_html = "\n".join(trans_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Digital Edition — {_escape_html(base_name)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #f8fafc; }}
h1 {{ font-size: 1.25rem; color: #334155; }}
.layout {{ display: flex; gap: 20px; flex-wrap: wrap; }}
.img-panel {{ flex: 1; min-width: 300px; position: relative; }}
.img-panel img {{ max-width: 100%; height: auto; display: block; }}
.img-panel svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
.img-panel svg line, .img-panel svg rect, .img-panel svg polygon {{ pointer-events: auto; cursor: pointer; }}
.trans-panel {{ flex: 1; min-width: 300px; padding: 16px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.read-region {{ margin-bottom: 1em; }}
.region-type {{ font-size: 0.75rem; color: #64748b; margin-bottom: 4px; }}
.line-span {{ display: block; padding: 2px 0; }}
.line-span.sel, .read-region.sel {{ background: rgba(59, 130, 246, 0.15); }}
.line-poly.sel {{ stroke: #2563eb; stroke-width: 2; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; }}
th {{ background: #f1f5f9; }}
</style>
</head>
<body>
<h1>Digital Edition: {_escape_html(base_name)}</h1>
<p><em>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — Agentic Document Processor</em></p>
<div class="layout">
  <div class="img-panel">
    <img id="docImg" src="data:image/png;base64,{b64}" width="{img_w}" height="{img_h}" alt="Document"/>
    <svg id="overlay" width="{img_w}" height="{img_h}" viewBox="0 0 {img_w} {img_h}">
      {svg_content}
    </svg>
  </div>
  <div id="transView" class="trans-panel">
    {trans_html}
  </div>
</div>
<script>
(function() {{
  const svg = document.getElementById('overlay');
  function clearSel() {{
    document.querySelectorAll('.line-span.sel, .line-poly.sel, .read-region.sel').forEach(el => el.classList.remove('sel'));
  }}
  function selLine(lid, rid) {{
    clearSel();
    document.querySelectorAll('.line-poly[data-lid="' + lid + '"]').forEach(el => el.classList.add('sel'));
    document.querySelectorAll('.line-span[data-lid="' + lid + '"]').forEach(el => el.classList.add('sel'));
    document.querySelectorAll('.read-region[data-rid="' + rid + '"]').forEach(el => el.classList.add('sel'));
  }}
  function selRegion(rid) {{
    clearSel();
    document.querySelectorAll('.region-g[data-rid="' + rid + '"]').forEach(el => el.classList.add('sel'));
    document.querySelectorAll('.read-region[data-rid="' + rid + '"]').forEach(el => el.classList.add('sel'));
  }}
  svg.addEventListener('click', function(e) {{
    const line = e.target.closest('.line-poly');
    if (line) {{ e.stopPropagation(); selLine(line.dataset.lid, line.dataset.rid); return; }}
    const reg = e.target.closest('.region-g');
    if (reg) selRegion(reg.dataset.rid);
  }});
  document.getElementById('transView').addEventListener('click', function(e) {{
    const line = e.target.closest('.line-span');
    if (line) {{ e.stopPropagation(); selLine(line.dataset.lid, line.dataset.rid); return; }}
    const reg = e.target.closest('.read-region');
    if (reg) selRegion(reg.dataset.rid);
  }});
}})();
</script>
</body>
</html>"""

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html)
    return output_filename
