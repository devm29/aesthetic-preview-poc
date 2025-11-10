import numpy as np

from src import face_utils


def test_skin_mask_from_bbox_covers_face_center():
    h, w = 200, 200
    image = np.zeros((h, w, 3), dtype=np.uint8)
    bbox = (50, 50, 100, 100)

    mask = face_utils.skin_mask_from_bbox(image, bbox)

    assert mask.shape[:2] == (h, w)
    # Center of the bbox should be inside the mask.
    cx = bbox[0] + bbox[2] // 2
    cy = bbox[1] + bbox[3] // 2
    assert mask[cy, cx] == 255


def test_under_eye_mask_returns_none_with_too_few_points():
    # Not enough landmarks for an eye polygon.
    landmarks = np.zeros((10, 2), dtype=np.float32)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    mask = face_utils.under_eye_mask(image, landmarks, side="left")
    assert mask is None


def test_lip_ring_mask_from_landmarks_produces_non_empty_mask():
    # Construct synthetic landmarks with valid indices for lip ring.
    num_points = 500
    landmarks = np.zeros((num_points, 2), dtype=np.float32)
    # Place all points roughly in the center region.
    landmarks[:, 0] = 100.0
    landmarks[:, 1] = 100.0

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = face_utils.lip_ring_mask_from_landmarks(image, landmarks)

    assert mask is not None
    assert mask.shape[:2] == (200, 200)
    assert np.any(mask > 0)

