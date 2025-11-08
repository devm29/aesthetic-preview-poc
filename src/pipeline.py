from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .qc import run_qc
from .face_utils import (
    skin_mask_from_bbox,
    skin_mask_from_landmarks,
    crop_with_bbox,
)
from .effects import apply_effects, lip_filler, under_eye_reduction
from .segmentation import segment_regions
from .diffusion import sdxl_inpaint_edit


def run_qc_pipeline(image_rgb: np.ndarray) -> Dict[str, Any]:
    """Thin wrapper around QC to keep app.py UI-focused."""
    return run_qc(image_rgb)


def compute_skin_mask(image_rgb: np.ndarray, face: Dict[str, Any]) -> np.ndarray:
    """Compute a skin mask from landmarks when available, else fall back to bbox."""
    if face is None:
        raise ValueError("Face metadata is required to compute a skin mask.")

    landmarks = face.get("landmarks_px")
    if landmarks is not None:
        return skin_mask_from_landmarks(image_rgb, landmarks)

    bbox = face.get("bbox")
    if bbox is None:
        raise ValueError("Face bbox is missing; cannot compute fallback skin mask.")
    return skin_mask_from_bbox(image_rgb, bbox)


def apply_adjustments_pipeline(
    image_rgb: np.ndarray,
    skin_mask: np.ndarray,
    wrinkle: int,
    tone: int,
    pigment: int,
) -> np.ndarray:
    """Apply classical, non-generative adjustments to the skin region."""
    return apply_effects(image_rgb, skin_mask, wrinkle=wrinkle, tone=tone, pigment=pigment)


def apply_aesthetic_procedures_pipeline(
    base_rgb: np.ndarray,
    image_for_regions: np.ndarray,
    face: Dict[str, Any],
    use_gen: bool,
    gen_strength: float,
    gen_steps: int,
    lip_int: int,
    ue_int: int,
    lip_prompt: str,
    lip_negative_prompt: str,
    eye_prompt: str,
    eye_negative_prompt: str,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply lip filler and under-eye reduction using generative or classical paths.

    Returns:
        result_rgb: The edited image.
        info: Dict with metadata, e.g.:
            - used_generative: bool
            - warnings: List[str]
    """
    result_rgb = base_rgb.copy()
    info: Dict[str, Any] = {"used_generative": False, "warnings": []}

    landmarks = None if face is None else face.get("landmarks_px")
    if landmarks is None:
        return result_rgb, info

    # If generative editing is enabled, try SDXL inpainting first.
    if use_gen:
        try:
            regions = segment_regions(image_for_regions, landmarks)
            bbox = face.get("bbox")

            if lip_int and lip_int > 0 and "lip_ring" in regions and bbox is not None:
                lip_mask = regions["lip_ring"]
                strength_lip = float(
                    np.clip(0.35 + 0.5 * (lip_int / 100.0), 0.1, 0.9)
                )
                # Allow geometry change: dilate lip mask according to intensity and face scale.
                bw = bbox[2]
                k = int(np.clip(bw * 0.015 * (lip_int / 100.0) * 1.5, 2, 24))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                lip_mask_edit = cv2.dilate(lip_mask, kernel, iterations=1)
                result_rgb = sdxl_inpaint_edit(
                    result_rgb,
                    lip_mask_edit,
                    prompt=lip_prompt,
                    negative_prompt=lip_negative_prompt,
                    strength=gen_strength if gen_strength is not None else strength_lip,
                    steps=gen_steps,
                    seed=int(seed) if seed is not None else None,
                )

            if ue_int and ue_int > 0 and "under_eye" in regions and bbox is not None:
                under_eye_mask = regions["under_eye"]
                strength_eye = float(
                    np.clip(0.30 + 0.45 * (ue_int / 100.0), 0.1, 0.9)
                )
                # Slight dilation to give model room, but smaller than lips.
                bw = bbox[2]
                k2 = int(np.clip(bw * 0.010 * (ue_int / 100.0), 1, 16))
                kernel2 = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * k2 + 1, 2 * k2 + 1)
                )
                under_eye_mask_edit = cv2.dilate(
                    under_eye_mask, kernel2, iterations=1
                )
                result_rgb = sdxl_inpaint_edit(
                    result_rgb,
                    under_eye_mask_edit,
                    prompt=eye_prompt,
                    negative_prompt=eye_negative_prompt,
                    strength=gen_strength if gen_strength is not None else strength_eye,
                    steps=gen_steps,
                    seed=int(seed) + 1 if seed is not None else None,
                )

            info["used_generative"] = True
            return result_rgb, info
        except ImportError:
            info["warnings"].append(
                "Generative editing dependencies not installed; using fast non-generative effects instead."
            )
            # Fall through to classical path below.

    # Classical (non-generative) fallback or explicit choice.
    if lip_int and lip_int > 0:
        result_rgb = lip_filler(result_rgb, landmarks, lip_int)
    if ue_int and ue_int > 0:
        result_rgb = under_eye_reduction(result_rgb, landmarks, ue_int)

    return result_rgb, info


def compute_identity_similarity(
    original_rgb: np.ndarray,
    edited_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.1,
) -> float:
    """Compute SSIM over the cropped face region as an identity proxy."""
    orig_crop = crop_with_bbox(original_rgb, bbox, margin=margin)
    res_crop = crop_with_bbox(edited_rgb, bbox, margin=margin)

    orig_g = cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY)
    res_g = cv2.cvtColor(res_crop, cv2.COLOR_RGB2GRAY)

    min_h = min(orig_g.shape[0], res_g.shape[0])
    min_w = min(orig_g.shape[1], res_g.shape[1])
    orig_g_resized = cv2.resize(orig_g, (min_w, min_h))
    res_g_resized = cv2.resize(res_g, (min_w, min_h))

    try:
        return float(ssim(orig_g_resized, res_g_resized))
    except Exception:
        return 0.0

