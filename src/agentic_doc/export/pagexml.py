"""Export final transcription to PAGE XML 2019."""

import os
from datetime import datetime
from typing import Any, Dict, Optional


def write_pagexml(
    final_transcription: Dict[str, Any],
    output_filename: str,
    source_image_name: str,
    image_width: int = 0,
    image_height: int = 0,
) -> str:
    """Write PAGE XML file. Returns output_filename."""
    type_mapping = {
        "ParagraphRegion": "TextRegion",
        "TitleRegion": "TextRegion",
        "HeadingRegion": "TextRegion",
        "SubheadingRegion": "TextRegion",
        "PageNumberRegion": "TextRegion",
        "MarginaliaRegion": "TextRegion",
        "TableRegion": "TableRegion",
        "ImageRegion": "ImageRegion",
        "DiagramRegion": "GraphicRegion",
        "FootnoteRegion": "TextRegion",
    }
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"'
        ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'
        ' http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd">',
        "  <Metadata>",
        "    <Creator>Agentic Document Processor (LangGraph + Gemini)</Creator>",
        f"    <Created>{timestamp}</Created>",
        f"    <LastChange>{timestamp}</LastChange>",
        "  </Metadata>",
        f'  <Page imageFilename="{os.path.basename(source_image_name)}" imageWidth="{image_width}" imageHeight="{image_height}">',
    ]
    xml_lines.append("    <ReadingOrder>")
    xml_lines.append('      <OrderedGroup id="reading_order">')
    for i, region in enumerate(final_transcription.get("regions", [])):
        rid = region["region_id"]
        xml_lines.append(f'        <RegionRefIndexed regionRef="{rid}" index="{i}"/>')
    xml_lines.append("      </OrderedGroup>")
    xml_lines.append("    </ReadingOrder>")

    for region in final_transcription.get("regions", []):
        rid = region["region_id"]
        rtype = region["region_type"]
        page_type = type_mapping.get(rtype, "TextRegion")
        bbox = region.get("bbox", {})
        x, y = bbox.get("x", 0), bbox.get("y", 0)
        w, h = bbox.get("width", 0), bbox.get("height", 0)
        coords = f"{x},{y} {x+w},{y} {x+w},{y+h} {x},{y+h}"

        if page_type in ("ImageRegion", "GraphicRegion"):
            xml_lines.append(f'    <{page_type} id="{rid}">')
            xml_lines.append(f'      <Coords points="{coords}"/>')
            xml_lines.append(f"    </{page_type}>")
        else:
            xml_lines.append(f'    <TextRegion id="{rid}" type="{rtype}">')
            xml_lines.append(f'      <Coords points="{coords}"/>')
            for line in region.get("lines", []):
                line_id = line.get("line_id", "")
                line_text = line.get("text", "")
                line_polygon = line.get("polygon", [])
                if line_polygon:
                    points = " ".join(f"{p[0]},{p[1]}" for p in line_polygon)
                else:
                    lb = line.get("bbox", {})
                    lx, ly = lb.get("x", 0), lb.get("y", 0)
                    lw, lh = lb.get("width", 0), lb.get("height", 0)
                    points = f"{lx},{ly} {lx+lw},{ly} {lx+lw},{ly+lh} {lx},{ly+lh}"
                xml_lines.append(f'      <TextLine id="{line_id}">')
                xml_lines.append(f'        <Coords points="{points}"/>')
                if line_text:
                    escaped = line_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                    xml_lines.append("        <TextEquiv>")
                    xml_lines.append(f"          <Unicode>{escaped}</Unicode>")
                    xml_lines.append("        </TextEquiv>")
                xml_lines.append("      </TextLine>")
            region_text = region.get("text", "")
            if region_text:
                escaped = region_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                xml_lines.append("      <TextEquiv>")
                xml_lines.append(f"        <Unicode>{escaped}</Unicode>")
                xml_lines.append("      </TextEquiv>")
            xml_lines.append("    </TextRegion>")
    xml_lines.append("  </Page>")
    xml_lines.append("</PcGts>")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_lines))
    return output_filename
