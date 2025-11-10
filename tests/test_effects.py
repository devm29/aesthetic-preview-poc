import numpy as np

from src import effects


def _make_dummy_image_and_mask(h: int = 128, w: int = 128):
    image = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    mask = np.full((h, w), 255, dtype=np.uint8)
    return image, mask


def test_apply_effects_preserves_shape_and_dtype():
    image, mask = _make_dummy_image_and_mask()

    out = effects.apply_effects(image, mask, wrinkle=50, tone=30, pigment=40)

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert np.all((out >= 0) & (out <= 255))


def test_individual_effects_are_noops_when_intensity_zero():
    image, mask = _make_dummy_image_and_mask()

    out_wrinkle = effects.wrinkle_smoothing(image, mask, intensity=0)
    out_tone = effects.tone_evening(image, mask, intensity=0)
    out_pigment = effects.pigmentation_reduction(image, mask, intensity=0)

    assert np.array_equal(out_wrinkle, image)
    assert np.array_equal(out_tone, image)
    assert np.array_equal(out_pigment, image)

