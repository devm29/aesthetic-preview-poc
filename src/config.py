from typing import Tuple

"""
Central configuration for QC thresholds, diffusion model selection,
and default generative prompts. Centralizing these values makes it
easier to tune behavior without touching core logic.
"""

# QC configuration
QC_BLUR_TARGET_WIDTH: int = 512
QC_BLUR_VARIANCE_THRESHOLD: float = 100.0

QC_LIGHTING_MEAN_RANGE: Tuple[float, float] = (90.0, 200.0)
QC_LIGHTING_STD_MIN: float = 25.0
QC_LIGHTING_OVEREXPOSED_MAX_FRAC: float = 0.15
QC_LIGHTING_UNDEREXPOSED_MAX_FRAC: float = 0.15

QC_ORIENTATION_MAX_ABS_DEG: float = 20.0

# Diffusion / generative model configuration
SDXL_INPAINT_MODEL_ID: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# Default prompts for generative edits.
DEFAULT_LIP_PROMPT: str = (
    "natural lip augmentation, fuller lips, realistic shading and skin texture, "
    "neutral tone, photographic, high detail"
)
DEFAULT_LIP_NEGATIVE_PROMPT: str = (
    "over-smooth, plastic skin, lipstick, heavy makeup, cartoon, deformed, disfigured"
)
DEFAULT_EYE_PROMPT: str = (
    "reduce under-eye bags, smooth under-eye skin, even tone, maintain pore detail, "
    "realistic portrait, photographic"
)
DEFAULT_EYE_NEGATIVE_PROMPT: str = (
    "over-smooth, plastic, blur, cartoon, airbrushed"
)

