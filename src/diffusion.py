from typing import Optional, Tuple
import logging

import cv2
import numpy as np
from PIL import Image

from .config import SDXL_INPAINT_MODEL_ID

"""
SDXL inpainting-based editor with lazy imports.

We avoid importing heavy libs unless actually used.
Provides a single entrypoint `sdxl_inpaint_edit` that edits only inside the
provided mask (uint8 0/255). Internally, it crops to the mask ROI for speed
and pastes back.

If diffusers/torch are missing, raises ImportError with a clear message so the
caller can inform the user.
"""

logger = logging.getLogger(__name__)

_PIPE = None
_DEVICE = None
_DTYPE = None


def _get_device_and_dtype(torch):
    # Prefer MPS on Apple Silicon if available, else CUDA, else CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32  # fp32 on MPS is safer
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def _ensure_pipeline(model_id: str = SDXL_INPAINT_MODEL_ID):
    global _PIPE, _DEVICE, _DTYPE
    if _PIPE is not None:
        return _PIPE, _DEVICE, _DTYPE

    try:
        import torch  # noqa: F401
        from diffusers import StableDiffusionXLInpaintPipeline
    except Exception as e:
        logger.exception("Failed to import generative editing dependencies.")
        raise ImportError(
            "Generative editing dependencies not installed. Install from requirements-ml.txt"
        ) from e

    import torch

    _DEVICE, _DTYPE = _get_device_and_dtype(torch)

    logger.info("Loading SDXL inpaint pipeline '%s' on device %s", model_id, _DEVICE)
    _PIPE = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=_DTYPE,
        variant="fp16" if _DTYPE == torch.float16 else None,
    )
    _PIPE.to(_DEVICE)
    _PIPE.enable_attention_slicing()
    return _PIPE, _DEVICE, _DTYPE


def _numpy_to_pil(img: np.ndarray) -> Image.Image:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def _roi_from_mask(mask: np.ndarray, pad: int = 16) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = mask.shape[:2]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return int(x0), int(y0), int(x1), int(y1)


def sdxl_inpaint_edit(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    negative_prompt: str = "over-smooth, plastic skin, cartoon, drawing, lipstick, makeup, heavy makeup",
    strength: float = 0.55,
    steps: int = 14,
    seed: Optional[int] = None,
    model_id: str = SDXL_INPAINT_MODEL_ID,
) -> np.ndarray:
    """Run SDXL inpainting within a tight ROI defined by mask.

    image_rgb: HxWx3 uint8
    mask: HxW uint8 where 255 indicates region to modify
    """
    pipe, device, _ = _ensure_pipeline(model_id)

    # Compute ROI around the mask to speed up
    roi = _roi_from_mask(mask)
    if roi is None:
        return image_rgb
    x0, y0, x1, y1 = roi

    img_roi = image_rgb[y0:y1, x0:x1]
    mask_roi = mask[y0:y1, x0:x1]

    # Resize ROI to manageable size (<= 768 longest side)
    h, w = img_roi.shape[:2]
    max_side = 768
    scale = min(1.0, max_side / float(max(h, w)))
    target_w, target_h = int(w * scale), int(h * scale)
    if scale < 1.0:
        img_in = np.ascontiguousarray(np.round(
            cv2.resize(img_roi, (target_w, target_h), interpolation=cv2.INTER_AREA)
        ).astype(np.uint8))
        mask_in = np.ascontiguousarray(cv2.resize(mask_roi, (target_w, target_h), interpolation=cv2.INTER_NEAREST))
    else:
        img_in, mask_in = img_roi, mask_roi

    # Convert to PIL for diffusers
    init_pil = _numpy_to_pil(img_in)
    mask_pil = _numpy_to_pil(mask_in)

    import torch

    generator = torch.Generator(device=device) if seed is not None else None
    if generator is not None:
        generator = generator.manual_seed(int(seed))

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_pil,
        mask_image=mask_pil,
        guidance_scale=5.0,
        strength=float(np.clip(strength, 0.1, 0.95)),
        num_inference_steps=int(np.clip(steps, 4, 50)),
        generator=generator,
    ).images[0]

    out_np = np.array(out)

    # Resize back to ROI size, paste into original using mask as alpha
    if (out_np.shape[1], out_np.shape[0]) != (w, h):
        out_np = cv2.resize(out_np, (w, h), interpolation=cv2.INTER_CUBIC)
        mask_back = cv2.resize(mask_in, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_back = mask_roi

    result = image_rgb.copy()
    m = (mask_back.astype(np.float32) / 255.0)[..., None]
    result[y0:y1, x0:x1] = (out_np.astype(np.float32) * m + img_roi.astype(np.float32) * (1 - m)).astype(np.uint8)
    return result
