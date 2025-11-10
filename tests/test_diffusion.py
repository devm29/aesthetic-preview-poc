import sys
import types

import numpy as np
from PIL import Image

from src import diffusion


class _FakePipeline:
    """Lightweight stand-in for StableDiffusionXLInpaintPipeline.

    Returns a solid-color image so we can verify ROI compositing without
    importing heavy ML dependencies.
    """

    def __call__(self, *args, **kwargs):
        init_image = kwargs["image"]
        width, height = init_image.size
        img = Image.new("RGB", (width, height), (128, 0, 0))
        return types.SimpleNamespace(images=[img])


class _FakeGenerator:
    def __init__(self, device=None):
        self.device = device

    def manual_seed(self, seed: int):
        return self


def test_sdxl_inpaint_edit_preserves_shape_and_edits_roi(monkeypatch):
    # Patch pipeline loader to avoid importing diffusers/torch.
    fake_pipe = _FakePipeline()
    device = "cpu"
    dtype = "float32"
    monkeypatch.setattr(
        diffusion,
        "_ensure_pipeline",
        lambda model_id=None: (fake_pipe, device, dtype),
    )

    # Patch torch module used inside sdxl_inpaint_edit.
    fake_torch = types.SimpleNamespace(Generator=_FakeGenerator)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255

    out = diffusion.sdxl_inpaint_edit(
        image,
        mask,
        prompt="test prompt",
        negative_prompt="",
        strength=0.5,
        steps=4,
        seed=0,
    )

    assert out.shape == image.shape
    assert out.dtype == np.uint8

    # Pixels outside the mask should remain unchanged (still black).
    assert np.all(out[:8, :8] == 0)
    # Inside the masked region we expect edits from the fake pipeline.
    assert np.any(out[20:44, 20:44] != 0)

