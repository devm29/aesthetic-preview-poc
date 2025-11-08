import cv2
import numpy as np
from typing import Tuple
from skimage.transform import PiecewiseAffineTransform, warp

from .face_utils import (
    lip_points_from_landmarks,
    lip_mask_from_landmarks,
    lip_ring_mask_from_landmarks,
    under_eye_mask,
)


def _blend_masked(src: np.ndarray, dst: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend dst into src within mask using alpha (0..1)."""
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    out = src.astype(np.float32) * (1 - mask_f * alpha) + dst.astype(np.float32) * (mask_f * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def wrinkle_smoothing(image_rgb: np.ndarray, skin_mask: np.ndarray, intensity: int) -> np.ndarray:
    if intensity <= 0:
        return image_rgb
    alpha = np.clip(intensity / 100.0, 0.0, 1.0)
    # Edge-preserving smoothing via bilateral filter
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    sm_bgr = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    sm_rgb = cv2.cvtColor(sm_bgr, cv2.COLOR_BGR2RGB)
    return _blend_masked(image_rgb, sm_rgb, skin_mask, alpha)


def tone_evening(image_rgb: np.ndarray, skin_mask: np.ndarray, intensity: int) -> np.ndarray:
    if intensity <= 0:
        return image_rgb
    alpha = np.clip(intensity / 100.0, 0.0, 1.0)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return _blend_masked(image_rgb, rgb_eq, skin_mask, alpha)


def pigmentation_reduction(image_rgb: np.ndarray, skin_mask: np.ndarray, intensity: int) -> np.ndarray:
    if intensity <= 0:
        return image_rgb
    alpha = np.clip(intensity / 100.0, 0.0, 1.0)
    # Gentle local brightening for pixels darker than local mean
    blur = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=7, sigmaY=7)
    diff = blur.astype(np.float32) - image_rgb.astype(np.float32)
    lift = np.maximum(diff, 0)  # only lighten darker-than-local
    k = 0.6  # strength factor
    brightened = np.clip(image_rgb.astype(np.float32) + k * lift, 0, 255).astype(np.uint8)
    return _blend_masked(image_rgb, brightened, skin_mask, alpha)


def apply_effects(image_rgb: np.ndarray, skin_mask: np.ndarray, wrinkle: int, tone: int, pigment: int) -> np.ndarray:
    out = image_rgb
    out = wrinkle_smoothing(out, skin_mask, wrinkle)
    out = tone_evening(out, skin_mask, tone)
    out = pigmentation_reduction(out, skin_mask, pigment)
    return out


# ---------- New effects ----------
def _feather(mask: np.ndarray, ksize: int = 11) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    m = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return m


def lip_filler(image_rgb: np.ndarray, landmarks_px: np.ndarray, intensity: int) -> np.ndarray:
    """Increase lip volume by radial expansion + piecewise affine warp. Requires landmarks.

    intensity: 0..100 -> expansion factor k in [0.0, 0.35]
    """
    if intensity <= 0 or landmarks_px is None:
        return image_rgb

    lip_pts = lip_points_from_landmarks(landmarks_px)
    # Prefer ring mask (outer - inner) to avoid teeth/mouth interior warping artifacts
    lip_mask = lip_ring_mask_from_landmarks(image_rgb, landmarks_px)
    if lip_mask is None:
        lip_mask = lip_mask_from_landmarks(image_rgb, landmarks_px)
    if lip_pts is None or lip_mask is None:
        return image_rgb

    s = float(np.clip(intensity, 0, 100)) / 100.0
    # Anisotropic scaling around lip centroid using PCA: increase vertical fullness more than width
    src = lip_pts.astype(np.float32)
    c = np.mean(src, axis=0, keepdims=True)
    X = src - c
    # PCA basis
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort by descending variance: v0 ~ horizontal mouth axis, v1 ~ vertical
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order]  # 2x2
    # Scale: small on horizontal (keep width), larger on vertical (fullness)
    kx = 0.05 * s
    ky = 0.45 * s

    # Corner-aware attenuation: keep corners more stable to avoid fish-mouth effect
    left_corner = src[np.argmin(src[:, 0])]
    right_corner = src[np.argmax(src[:, 0])]
    lip_width = max(8.0, float(right_corner[0] - left_corner[0]))
    corner_scale = []
    X_ev = (V.T @ X.T).T  # points in eigen-basis
    for i, p in enumerate(src):
        d = min(np.linalg.norm(p - left_corner), np.linalg.norm(p - right_corner))
        t = np.clip(d / (0.45 * lip_width), 0.0, 1.0)
        w = 0.25 + 0.75 * t  # reduce near corners
        corner_scale.append(w)
    corner_scale = np.asarray(corner_scale, dtype=np.float32)

    dst_list = []
    for i in range(src.shape[0]):
        S_i = np.diag([1.0 + kx * 0.5 * corner_scale[i], 1.0 + ky * corner_scale[i]]).astype(np.float32)
        vec = X_ev[i]
        vec_scaled = (V @ (S_i @ vec)).T  # back to image basis
        dst_list.append((vec_scaled + c.squeeze()).astype(np.float32))
    dst = np.vstack(dst_list)

    # Add local anchors (ring around lip hull) to keep warp local
    x0, y0 = np.min(src, axis=0)
    x1, y1 = np.max(src, axis=0)
    pad_x = max(6.0, (x1 - x0) * 0.25)
    pad_y = max(6.0, (y1 - y0) * 0.25)
    # 8 anchors around ROI (corners + mid-edges)
    anchors = np.array([
        [x0 - pad_x, y0 - pad_y],
        [(x0 + x1) / 2, y0 - pad_y],
        [x1 + pad_x, y0 - pad_y],
        [x0 - pad_x, (y0 + y1) / 2],
        [x1 + pad_x, (y0 + y1) / 2],
        [x0 - pad_x, y1 + pad_y],
        [(x0 + x1) / 2, y1 + pad_y],
        [x1 + pad_x, y1 + pad_y],
    ], dtype=np.float32)
    src_all = np.vstack([src, anchors])
    dst_all = np.vstack([dst, anchors])

    tform = PiecewiseAffineTransform()
    ok = tform.estimate(src_all, dst_all)
    if not ok:
        return image_rgb

    warped = warp(
        image_rgb,
        tform,
        output_shape=image_rgb.shape,
        mode="edge",
        preserve_range=True,
        order=1,
    ).astype(np.uint8)

    # Subtle color/saturation enhancement within lips
    hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, sv, vv = cv2.split(hsv)
    sv = np.clip(sv * (1.0 + 0.25 * s), 0, 255)
    vv = np.clip(vv * (1.0 + 0.06 * s), 0, 255)
    hsv2 = cv2.merge([h, sv, vv]).astype(np.uint8)
    colored = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

    # Seamless (Poisson) clone for realistic boundaries, fallback to alpha blend
    try:
        moments = cv2.moments(lip_mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            src_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
            dst_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            clone_bgr = cv2.seamlessClone(src_bgr, dst_bgr, lip_mask, (cx, cy), cv2.NORMAL_CLONE)
            clone_rgb = cv2.cvtColor(clone_bgr, cv2.COLOR_BGR2RGB)
            return clone_rgb
    except Exception:
        pass
    lip_mask_f = _feather(lip_mask, 9)
    return _blend_masked(image_rgb, colored, lip_mask_f, alpha=1.0)


def under_eye_reduction(image_rgb: np.ndarray, landmarks_px: np.ndarray, intensity: int) -> np.ndarray:
    """Reduce under-eye darkness/bags by localized brightening + smoothing. Requires landmarks."""
    if intensity <= 0 or landmarks_px is None:
        return image_rgb

    mask_l = under_eye_mask(image_rgb, landmarks_px, side="left")
    mask_r = under_eye_mask(image_rgb, landmarks_px, side="right")
    if mask_l is None and mask_r is None:
        return image_rgb
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    if mask_l is not None:
        mask = cv2.bitwise_or(mask, mask_l)
    if mask_r is not None:
        mask = cv2.bitwise_or(mask, mask_r)

    # Refine mask with gentle erosion to avoid cheeks, then feather
    s = float(np.clip(intensity, 0, 100)) / 100.0
    ksz = max(3, int(5 + 6 * s))
    if ksz % 2 == 0:
        ksz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    mask_ref = cv2.erode(mask, kernel, iterations=1)
    mask_f = _feather(mask_ref, 11)

    # Convert to LAB and brighten L inside the mask
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32)
    # Lighten and reduce bluish/purplish tint
    deltaL = (255.0 - Lf) * (0.35 * s)
    A2 = (A.astype(np.float32) * (1.0 - 0.18 * s)).clip(0, 255).astype(np.uint8)
    B2 = np.clip(B.astype(np.float32) + 18.0 * s, 0, 255).astype(np.uint8)
    L_new = np.clip(Lf + deltaL, 0, 255).astype(np.uint8)
    lab_new = cv2.merge([L_new, A2, B2])
    rgb_bright = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)

    # Mild smoothing + detail suppression (frequency separation) within mask
    base = cv2.GaussianBlur(rgb_bright, (0, 0), sigmaX=2, sigmaY=2)
    detail = (rgb_bright.astype(np.float32) - base.astype(np.float32))
    detail_scale = 1.0 - 0.6 * s
    suppressed = np.clip(base.astype(np.float32) + detail_scale * detail, 0, 255).astype(np.uint8)
    # Optional guided filter edge-preserving refinement if available (opencv-contrib)
    try:
        gf = cv2.ximgproc.guidedFilter(image_rgb, suppressed, int(5 + 5 * s), 1e-3)
        suppressed = gf
    except Exception:
        pass
    bgr = cv2.cvtColor(suppressed, cv2.COLOR_RGB2BGR)
    sm_bgr = cv2.bilateralFilter(bgr, d=7, sigmaColor=40, sigmaSpace=40)
    sm_rgb = cv2.cvtColor(sm_bgr, cv2.COLOR_BGR2RGB)

    return _blend_masked(image_rgb, sm_rgb, mask_f, alpha=1.0)
