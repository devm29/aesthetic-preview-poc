import logging
from typing import Dict

import cv2
import numpy as np

from .face_utils import detect_face_landmarks
from .config import (
    QC_BLUR_TARGET_WIDTH,
    QC_BLUR_VARIANCE_THRESHOLD,
    QC_LIGHTING_MEAN_RANGE,
    QC_LIGHTING_STD_MIN,
    QC_LIGHTING_OVEREXPOSED_MAX_FRAC,
    QC_LIGHTING_UNDEREXPOSED_MAX_FRAC,
    QC_ORIENTATION_MAX_ABS_DEG,
)

logger = logging.getLogger(__name__)


def _variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_blur(
    image_rgb: np.ndarray,
    target_w: int = QC_BLUR_TARGET_WIDTH,
    threshold: float = QC_BLUR_VARIANCE_THRESHOLD,
) -> Dict:
    h, w = image_rgb.shape[:2]
    scale = target_w / float(w)
    resized = cv2.resize(image_rgb, (target_w, int(h * scale))) if w > target_w else image_rgb
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    var = _variance_of_laplacian(gray)
    passed = var >= threshold
    return {"pass": passed, "variance": var}


def check_lighting(image_rgb: np.ndarray) -> Dict:
    # Use HSV V channel statistics
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    mean = float(np.mean(v))
    std = float(np.std(v))
    over = float(np.mean(v > 240))  # fraction overexposed
    under = float(np.mean(v < 15))  # fraction underexposed

    mean_low, mean_high = QC_LIGHTING_MEAN_RANGE
    passed = (
        (mean_low <= mean <= mean_high)
        and (std >= QC_LIGHTING_STD_MIN)
        and (over < QC_LIGHTING_OVEREXPOSED_MAX_FRAC)
        and (under < QC_LIGHTING_UNDEREXPOSED_MAX_FRAC)
    )
    return {"pass": passed, "mean": mean, "std": std, "over_frac": over, "under_frac": under}


def check_orientation(
    roll_deg: float,
    max_abs_deg: float = QC_ORIENTATION_MAX_ABS_DEG,
) -> Dict:
    passed = abs(roll_deg) <= max_abs_deg
    return {"pass": passed, "roll_deg": float(roll_deg)}


def run_qc(image_rgb: np.ndarray) -> Dict:
    out: Dict = {"overall_pass": False, "messages": []}

    face = detect_face_landmarks(image_rgb)
    if face is None:
        out.update({"face_present": False})
        out["messages"].append("No face detected. Please center your face and try again.")
        logger.info("QC failed: no face detected.")
        return out
    out["face_present"] = True

    blur = check_blur(image_rgb)
    light = check_lighting(image_rgb)
    orient = check_orientation(face["roll_deg"]) if face else {"pass": False, "roll_deg": 999}

    out["blur"] = blur
    out["lighting"] = light
    out["orientation"] = orient

    msgs = []
    if not blur["pass"]:
        msgs.append("Image appears blurry. Hold steady or improve focus.")
    if not light["pass"]:
        msgs.append("Lighting not ideal. Use even, indirect light; avoid backlighting.")
    if not orient["pass"]:
        msgs.append("Rotate your head to be upright (~0°).")

    out["overall_pass"] = len(msgs) == 0
    out["messages"] = msgs
    out["face"] = face

    logger.debug(
        "QC results: overall_pass=%s, blur=%s, lighting=%s, orientation=%s",
        out["overall_pass"],
        blur,
        light,
        orient,
    )
    return out
