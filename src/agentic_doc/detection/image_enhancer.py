"""Image preprocessing for historical documents."""

import os
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image


class ImageEnhancer:
    """Image preprocessing: deskew, denoise, contrast, bleedthrough, faded."""

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        """Load image with cv2, falling back to PIL conversion if cv2 fails."""
        img = cv2.imread(image_path)
        if img is not None:
            return img
        # Fallback: load with PIL and convert to cv2 BGR format
        try:
            pil_img = Image.open(image_path).convert("RGB")
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Cannot read image: {image_path}. Error: {e}")

    def deskew(self, image: np.ndarray, angle: float = None) -> np.ndarray:
        if angle is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 30, 100, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, 80,
                minLineLength=gray.shape[1] // 8, maxLineGap=20,
            )
            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 - x1 != 0:
                        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        if abs(ang) < 15:
                            angles.append(ang)
                angle = float(np.median(angles)) if angles else 0.0
            else:
                angle = 0.0
        if abs(angle) < 0.1:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def denoise(self, image: np.ndarray, strength: int = 6) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
        return clahe.apply(image)

    def remove_bleedthrough(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kernel_size = max(gray.shape) // 30
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 51)
        background = cv2.morphologyEx(
            gray, cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )
        result = cv2.divide(gray, background, scale=255)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            _, a, b = cv2.split(lab)
            return cv2.cvtColor(cv2.merge([result, a, b]), cv2.COLOR_LAB2BGR)
        return result

    def enhance_faded(self, image: np.ndarray) -> np.ndarray:
        enhanced = self.enhance_contrast(image)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        return cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)

    def enhance(self, image_path: str, operations: List[str] = None) -> Dict[str, Any]:
        if not os.path.isfile(image_path):
            return {"status": "error", "error": f"File not found: {image_path}"}
        try:
            image = self._load_image(image_path)
        except ValueError as e:
            return {"status": "error", "error": str(e)}
        if operations is None:
            operations = ["deskew", "denoise", "enhance_contrast"]
        applied = []
        for op in operations:
            try:
                if op == "deskew":
                    image = self.deskew(image)
                elif op == "denoise":
                    image = self.denoise(image)
                elif op == "enhance_contrast":
                    image = self.enhance_contrast(image)
                elif op == "remove_bleedthrough":
                    image = self.remove_bleedthrough(image)
                elif op == "enhance_faded":
                    image = self.enhance_faded(image)
                applied.append(op)
            except Exception as e:
                print(f"Operation {op} failed: {e}")
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_enhanced{ext}"
        cv2.imwrite(output_path, image)
        return {
            "status": "success",
            "original_path": image_path,
            "enhanced_path": output_path,
            "operations_applied": applied,
        }
