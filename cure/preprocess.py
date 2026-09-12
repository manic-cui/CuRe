"""Deterministic image preprocessing for CuRe."""

import io
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image


def preprocess_image(path):
    """Read an image and return a normalized float tensor of shape [3, 336, 336]."""
    path = Path(path)
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.load()
    if path.suffix.lower() == ".png":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=96)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            image = decoded.convert("RGB")
            image.load()
    pipeline = A.Compose([
        A.SmallestMaxSize(max_size=390, interpolation=cv2.INTER_CUBIC, p=1.0),
        A.CenterCrop(height=390, width=390, p=1.0),
        A.Resize(height=336, width=336, interpolation=cv2.INTER_CUBIC, p=1.0),
        A.ImageCompression(quality_range=(70, 70), p=1.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    return pipeline(image=np.asarray(image))["image"]
