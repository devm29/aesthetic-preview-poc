import numpy as np
from typing import Optional, Dict

"""
Lightweight face region segmentation helpers.

This module intentionally avoids heavy dependencies.
It uses our existing landmark-driven utilities from `src.face_utils.py` and
acts as a seam where a proper face-parsing model (e.g., BiSeNet trained on CelebAMask-HQ)
can be plugged later.

API
----
- segment_regions(image_rgb, landmarks_px) -> Dict[str, np.ndarray]
  Returns binary masks (uint8 0/255):
    - 'lip_ring' : outer minus inner lips
    - 'under_eye': union of left/right under-eye regions

If `landmarks_px` is None, returns empty dict.
"""

from .face_utils import (
    lip_ring_mask_from_landmarks,
    under_eye_mask,
)


def segment_regions(image_rgb: np.ndarray, landmarks_px: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if landmarks_px is None:
        return out

    # Lips: prefer ring (outer - inner) to avoid teeth
    lip_ring = lip_ring_mask_from_landmarks(image_rgb, landmarks_px)
    if lip_ring is not None:
        out["lip_ring"] = lip_ring

    # Under-eye: left + right
    m_l = under_eye_mask(image_rgb, landmarks_px, side="left")
    m_r = under_eye_mask(image_rgb, landmarks_px, side="right")
    if m_l is not None or m_r is not None:
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if m_l is not None:
            mask = np.maximum(mask, m_l)
        if m_r is not None:
            mask = np.maximum(mask, m_r)
        out["under_eye"] = mask

    return out
