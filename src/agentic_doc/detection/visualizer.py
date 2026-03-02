"""Layout visualization: regions and lines on document image."""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image


class LayoutVisualizer:
    """Visualize detected regions and lines on the document image."""

    REGION_COLORS = {
        "TitleRegion": "#e74c3c",
        "SubtitleRegion": "#c0392b",
        "HeadingRegion": "#e67e22",
        "SubheadingRegion": "#d35400",
        "ParagraphRegion": "#3498db",
        "TextRegion": "#2980b9",
        "TableRegion": "#9b59b6",
        "MarginaliaRegion": "#1abc9c",
        "FootnoteRegion": "#f39c12",
        "ImageRegion": "#2ecc71",
        "HeaderRegion": "#95a5a6",
        "FooterRegion": "#7f8c8d",
        "CaptionRegion": "#34495e",
        "NumberedListRegion": "#16a085",
        "BulletListRegion": "#27ae60",
        "default": "#64748b",
    }
    LINE_COLOR = "#00ff00"

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def visualize(
        self,
        image_path: str,
        regions: List[Dict],
        show_lines: bool = True,
        show_reading_order: bool = True,
        figsize: Tuple[int, int] = (15, 20),
    ) -> plt.Figure:
        image = Image.open(image_path)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(image)

        for region in regions:
            bbox = region["bbox"]
            region_type = region.get("type", "TextRegion")
            color = self.REGION_COLORS.get(region_type, self.REGION_COLORS["default"])
            rgb = self.hex_to_rgb(color)
            rect = patches.FancyBboxPatch(
                (bbox["x"], bbox["y"]), bbox["width"], bbox["height"],
                linewidth=2.5, edgecolor=color, facecolor=(*rgb, 0.15),
                boxstyle="round,pad=0.01",
            )
            ax.add_patch(rect)
            label_text = region_type.replace("Region", "")
            if show_reading_order:
                label_text = f"{region.get('reading_order', '?')}. {label_text}"
            ax.text(
                bbox["x"] + 5, bbox["y"] + 20,
                label_text,
                fontsize=9, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9, edgecolor="none"),
            )
            if show_lines and "lines" in region:
                for line in region["lines"]:
                    if line.get("polygon"):
                        poly = MplPolygon(
                            line["polygon"],
                            fill=False, edgecolor=self.LINE_COLOR,
                            linewidth=1, alpha=0.7, linestyle="--",
                        )
                        ax.add_patch(poly)
                    elif line.get("bbox"):
                        lb = line["bbox"]
                        line_rect = patches.Rectangle(
                            (lb["x"], lb["y"]), lb["width"], lb["height"],
                            linewidth=1, edgecolor=self.LINE_COLOR,
                            facecolor="none", alpha=0.7, linestyle="--",
                        )
                        ax.add_patch(line_rect)

        ax.axis("off")
        ax.set_title("Layout Analysis: Regions (colored boxes) + Lines (green dashed)", fontsize=14)
        unique_types = list(set(r.get("type", "TextRegion") for r in regions))
        legend_patches = [patches.Patch(color=self.REGION_COLORS.get(t, self.REGION_COLORS["default"]), label=t.replace("Region", "")) for t in sorted(unique_types)]
        if show_lines:
            legend_patches.append(patches.Patch(facecolor="none", edgecolor=self.LINE_COLOR, linestyle="--", label="Text Lines"))
        ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
        plt.tight_layout()
        return fig

    def save_visualization(
        self, image_path: str, regions: List[Dict], output_path: str, **kwargs
    ) -> str:
        fig = self.visualize(image_path, regions, **kwargs)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path
