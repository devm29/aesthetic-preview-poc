import cv2
import numpy as np
from typing import Dict, Optional, Tuple

try:
    import mediapipe as mp
except Exception as e:
    mp = None


_FACE_MESH_IDXS = {
    "left_eye": 33,   # outer corner
    "right_eye": 263, # outer corner
}


def _haar_face_bbox(image_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Fallback face bbox detection using OpenCV Haar cascade. Returns largest bbox or None."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return None
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # Choose largest area face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)


def detect_face_landmarks(image_rgb: np.ndarray) -> Optional[Dict]:
    """
    Returns dict with keys: 'landmarks_px' (Nx2), 'bbox' (x, y, w, h), 'roll_deg' (float)
    or None if no face.
    """
    h, w = image_rgb.shape[:2]

    # Try MediaPipe FaceMesh first if available
    if mp is not None:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as mesh:
            results = mesh.process(image_rgb)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0]
                pts = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark], dtype=np.float32)
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
                bbox = (
                    int(max(0, x_min)),
                    int(max(0, y_min)),
                    int(min(w - 1, x_max) - max(0, x_min)),
                    int(min(h - 1, y_max) - max(0, y_min)),
                )
                roll_deg = compute_roll_degrees(pts)
                return {"landmarks_px": pts, "bbox": bbox, "roll_deg": roll_deg}

    # Fallback: Haar-based bbox only
    bbox = _haar_face_bbox(image_rgb)
    if bbox is None:
        return None
    # Without landmarks, we cannot estimate roll precisely; assume upright
    return {"landmarks_px": None, "bbox": bbox, "roll_deg": 0.0}


def compute_roll_degrees(landmarks_px: np.ndarray) -> float:
    """Estimate roll (tilt) from outer eye corners using face mesh indices."""
    li = _FACE_MESH_IDXS["left_eye"]
    ri = _FACE_MESH_IDXS["right_eye"]
    p_left = landmarks_px[li]
    p_right = landmarks_px[ri]
    dy = p_right[1] - p_left[1]
    dx = p_right[0] - p_left[0]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return float(angle_deg)


def skin_mask_from_landmarks(image_rgb: np.ndarray, landmarks_px: np.ndarray) -> np.ndarray:
    """Create a convex-hull skin mask from face landmarks."""
    h, w = image_rgb.shape[:2]
    pts = landmarks_px.astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def skin_mask_from_bbox(image_rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Approximate skin mask as an ellipse within the face bbox (fallback if landmarks missing)."""
    h, w = image_rgb.shape[:2]
    x, y, bw, bh = bbox
    mask = np.zeros((h, w), dtype=np.uint8)
    # Slightly shrink the ellipse to avoid hair/background
    cx = x + bw // 2
    cy = y + bh // 2
    axes = (int(bw * 0.45), int(bh * 0.55))
    angle = 0
    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, thickness=-1)
    return mask


def crop_with_bbox(image_rgb: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.1) -> np.ndarray:
    x, y, bw, bh = bbox
    h, w = image_rgb.shape[:2]
    mx = int(bw * margin)
    my = int(bh * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w, x + bw + mx)
    y1 = min(h, y + bh + my)
    return image_rgb[y0:y1, x0:x1]


# --- Extra regions: lips and eyes (MediaPipe path) ---
def _unique_idxs_from_connections(connections) -> Optional[list]:
    try:
        idxs = set()
        for a, b in connections:
            idxs.add(a)
            idxs.add(b)
        return sorted(list(idxs))
    except Exception:
        return None


def get_lip_indices() -> Optional[list]:
    if mp is None:
        return None
    try:
        fmc = mp.solutions.face_mesh_connections
        return _unique_idxs_from_connections(fmc.FACEMESH_LIPS)
    except Exception:
        return None


def get_eye_indices(side: str) -> Optional[list]:
    if mp is None:
        return None
    try:
        fmc = mp.solutions.face_mesh_connections
        if side.lower().startswith("l"):
            conns = fmc.FACEMESH_LEFT_EYE
        else:
            conns = fmc.FACEMESH_RIGHT_EYE
        return _unique_idxs_from_connections(conns)
    except Exception:
        return None


def lip_points_from_landmarks(landmarks_px: np.ndarray) -> Optional[np.ndarray]:
    idxs = get_lip_indices()
    if not idxs:
        return None
    try:
        return landmarks_px[idxs]
    except Exception:
        return None


def lip_mask_from_landmarks(image_rgb: np.ndarray, landmarks_px: np.ndarray) -> Optional[np.ndarray]:
    """Create a convex-hull mask over the lip region. Returns None if unavailable."""
    pts = lip_points_from_landmarks(landmarks_px)
    if pts is None or len(pts) < 3:
        return None
    h, w = image_rgb.shape[:2]
    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def eye_polygon_from_landmarks(landmarks_px: np.ndarray, side: str) -> Optional[np.ndarray]:
    idxs = get_eye_indices(side)
    if not idxs:
        return None
    try:
        return landmarks_px[idxs]
    except Exception:
        return None


def under_eye_mask(image_rgb: np.ndarray, landmarks_px: np.ndarray, side: str) -> Optional[np.ndarray]:
    """Approximate under-eye area with an ellipse below the eye polygon."""
    poly = eye_polygon_from_landmarks(landmarks_px, side)
    if poly is None or len(poly) < 3:
        return None
    h, w = image_rgb.shape[:2]
    min_x, min_y = np.min(poly, axis=0)
    max_x, max_y = np.max(poly, axis=0)
    cx = int((min_x + max_x) / 2)
    cy = int((min_y + max_y) / 2)
    width = max(8, int(max_x - min_x))
    height = max(6, int(max_y - min_y))
    # Place ellipse slightly below eye center
    cy_under = int(cy + height * 0.35)
    axes = (int(width * 0.55), int(height * 0.60))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy_under), axes, 0, 0, 360, 255, thickness=-1)
    return mask


# --- Lip outer/inner ring utilities (avoid teeth warping) ---
_LIP_OUTER_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
_LIP_INNER_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]


def lip_ring_mask_from_landmarks(image_rgb: np.ndarray, landmarks_px: np.ndarray) -> Optional[np.ndarray]:
    """Returns outer-minus-inner lip mask (the lip tissue), feather-ready."""
    h, w = image_rgb.shape[:2]
    if landmarks_px is None or landmarks_px.shape[0] < 400:  # sanity
        return None
    try:
        outer = landmarks_px[_LIP_OUTER_IDX].astype(np.int32)
        inner = landmarks_px[_LIP_INNER_IDX].astype(np.int32)
    except Exception:
        return None
    mask_outer = np.zeros((h, w), dtype=np.uint8)
    mask_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_outer, cv2.convexHull(outer), 255)
    cv2.fillConvexPoly(mask_inner, cv2.convexHull(inner), 255)
    ring = cv2.subtract(mask_outer, mask_inner)
    return ring
