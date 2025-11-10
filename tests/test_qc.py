import numpy as np

from src import qc


def test_check_blur_returns_expected_keys_and_types():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    result = qc.check_blur(image)

    assert set(result.keys()) == {"pass", "variance"}
    assert isinstance(result["pass"], bool)
    assert isinstance(result["variance"], float)


def test_check_lighting_returns_expected_keys_and_types():
    # Mid-gray image should produce reasonable stats.
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    result = qc.check_lighting(image)

    expected_keys = {"pass", "mean", "std", "over_frac", "under_frac"}
    assert expected_keys.issubset(result.keys())
    assert isinstance(result["pass"], bool)


def test_run_qc_handles_no_face_gracefully():
    # Completely blank image should yield "no face detected" path.
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    out = qc.run_qc(image)

    assert out.get("face_present") is False
    assert out.get("overall_pass") is False
    assert any("No face detected" in msg for msg in out.get("messages", []))

