# Local Aesthetic Preview POC (Streamlit)

A local-only, open-source proof-of-concept to preview non-invasive aesthetic treatment effects on a selfie. Runs entirely on your machine with CPU using Streamlit.

## Features
- **Upload & QC**: Face presence, blur, orientation (roll), and lighting checks.
- **Adjustments**: Sliders for wrinkle smoothing, skin tone evening, and pigmentation reduction (parametric, non-generative).
- **Procedures (beta)**: Optional lip augmentation and under-eye reduction using either fast, non-generative warps or SDXL inpainting on precise masks when ML deps are installed.
- **Compare**: Side-by-side before/after preview.
- **Identity Safety**: SSIM similarity over face crop as a simple identity-preservation signal.
- **MediaPipe Optional**: If `mediapipe` is not installed, the app falls back to OpenCV Haar-based face detection.
- **Open Source**: MIT licensed; no cloud dependencies.

## Quickstart

1. Create and activate a Python 3.10+ environment.
2. Install deps (MediaPipe optional):
   ```bash
   pip install -r requirements.txt
   ```
   Optionally, for more accurate landmarks:
   ```bash
   pip install mediapipe==0.10.9
   ```
   And for generative SDXL inpainting (larger install, GPU/MPS recommended):
   ```bash
   pip install -r requirements-ml.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open the provided localhost URL in your browser.

## What this POC does (today)
- Runs locally on CPU by default (optional GPU/MPS acceleration for SDXL inpainting).
- Performs selfie QC and basic, medically conservative, non-generative retouch effects:
  - Wrinkle smoothing (skin-only, edge-preserving blending).
  - Tone evening (CLAHE on luminance in skin region).
  - Pigmentation reduction (gentle local contrast lift on darker-than-local skin areas).
- Optionally applies localized, generative edits (SDXL inpainting) to lips and under-eye regions when ML dependencies are installed.
- Ensures minimal identity drift via SSIM check on the face crop.

## Notes and Limitations
- This is a POC, not a clinical tool. Outputs are illustrative and not medical advice.
- Diffusion/generative models are optional and gated behind `requirements-ml.txt` and explicit UI toggles.
- Identity similarity uses SSIM (structure), not biometric embeddings.
- Face detection uses MediaPipe FaceMesh if available; otherwise it uses OpenCV's Haar cascade (bbox only) and a bbox-based skin mask.

## Roadmap (post-POC)
- Optional identity embeddings (InsightFace, ONNXRuntime) behind a feature flag.
- Faster/even more realistic effects, additional treatment types, hair/teeth segmentation, etc.
- Optional on-device optimizations and packaging.

## Repository Structure
```
Derma-Scan/
├─ app.py                # Streamlit app (UI only; delegates to src/pipeline.py)
├─ requirements.txt      # Core, non-generative dependencies
├─ requirements-ml.txt   # Optional heavy ML deps for SDXL inpainting
├─ pytest.ini
├─ LICENSE (MIT)
├─ README.md
├─ .gitignore
├─ src/
│  ├─ __init__.py
│  ├─ config.py          # Central configuration and prompts
│  ├─ pipeline.py        # Orchestration pipeline (QC, masks, effects, diffusion)
│  ├─ face_utils.py      # Mediapipe/OpenCV-based face detection/landmarks, masks
│  ├─ qc.py              # QC checks: face, blur, orientation, lighting
│  ├─ effects.py         # Parametric image effects (skin-mask aware)
│  ├─ segmentation.py    # Landmark-based lip/under-eye segmentation
│  └─ diffusion.py       # SDXL inpainting wrapper with lazy loading
└─ tests/
   ├─ test_qc.py
   ├─ test_face_utils.py
   ├─ test_effects.py
   └─ test_diffusion.py
```

## License
MIT © 2025

## Disclaimer
This software is for demonstration purposes only and is **not** a medical device. Use responsibly.

## Troubleshooting
- If you see "No face detected": ensure your face is centered, upright, and well-lit. The fallback Haar detector is more sensitive to lighting.
- If installs fail on macOS ARM: try `pip install --upgrade pip setuptools wheel` first, then install requirements.
- If MediaPipe install is slow or fails, skip it; the fallback will run without it.
