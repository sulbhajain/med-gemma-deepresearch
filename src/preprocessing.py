"""
Medical-grade image preprocessing for fetal ultrasound images.

Pipeline
────────
1. Load as RGB (MedGemma vision encoder expects 3-channel input)
2. Convert to grayscale for US-specific processing
3. Gaussian despeckling  — reduces speckle noise common in US
4. CLAHE-approximation   — PIL contrast + sharpness enhancement
   (true CLAHE requires OpenCV; PIL version is Kaggle-compatible)
5. Back to RGB → resize to TARGET_SIZE (default 448 × 448)
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from config import (
    TARGET_SIZE, ENHANCE_CONTRAST, REDUCE_NOISE,
    CONTRAST_FACTOR, SHARPNESS_FACTOR, GAUSSIAN_RADIUS,
)


class FetalUltrasoundPreprocessor:
    """
    Callable preprocessor: FetalUltrasoundPreprocessor()(path) → PIL.Image

    Parameters
    ----------
    enhance_contrast : apply contrast + sharpness boost (CLAHE approximation)
    reduce_noise     : apply Gaussian blur for speckle reduction
    target_size      : (W, H) passed to MedGemma vision encoder
    """

    def __init__(
        self,
        enhance_contrast: bool = ENHANCE_CONTRAST,
        reduce_noise:     bool = REDUCE_NOISE,
        target_size:      tuple = TARGET_SIZE,
    ):
        self.enhance_contrast = enhance_contrast
        self.reduce_noise     = reduce_noise
        self.target_size      = target_size

    def __call__(self, image_path: str) -> Image.Image:
        return self.preprocess(image_path)

    def preprocess(self, image_path: str) -> Image.Image:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"   ⚠️  Could not open {image_path}: {e}")
            return Image.new("RGB", self.target_size, (64, 64, 64))

        gray = img.convert("L")

        if self.reduce_noise:
            gray = gray.filter(ImageFilter.GaussianBlur(radius=GAUSSIAN_RADIUS))

        if self.enhance_contrast:
            gray = ImageEnhance.Contrast(gray).enhance(CONTRAST_FACTOR)
            gray = ImageEnhance.Sharpness(gray).enhance(SHARPNESS_FACTOR)

        out = gray.convert("RGB").resize(self.target_size, Image.Resampling.LANCZOS)
        return out

    def preprocess_batch(
        self,
        paths: List[str],
        save_dir: Optional[Path] = None,
    ) -> List[Image.Image]:
        results = []
        for p in tqdm(paths, desc="Preprocessing images"):
            img = self.preprocess(p)
            if save_dir is not None:
                out = Path(save_dir) / f"proc_{Path(p).stem}.png"
                img.save(out)
            results.append(img)
        return results

    @staticmethod
    def visualise(image_path: str, save_path: Optional[str] = None):
        """Side-by-side comparison: raw vs preprocessed."""
        pre = FetalUltrasoundPreprocessor()
        raw  = Image.open(image_path).convert("RGB").resize(TARGET_SIZE)
        proc = pre.preprocess(image_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(np.array(raw), cmap="gray")
        ax1.set_title("Raw Ultrasound", fontsize=12)
        ax1.axis("off")
        ax2.imshow(np.array(proc), cmap="gray")
        ax2.set_title("After CLAHE + Despeckle", fontsize=12)
        ax2.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
