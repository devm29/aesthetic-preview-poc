import io
import hashlib

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from src.config import (
    DEFAULT_LIP_PROMPT,
    DEFAULT_LIP_NEGATIVE_PROMPT,
    DEFAULT_EYE_PROMPT,
    DEFAULT_EYE_NEGATIVE_PROMPT,
)
from src.pipeline import (
    run_qc_pipeline,
    compute_skin_mask,
    apply_adjustments_pipeline,
    apply_aesthetic_procedures_pipeline,
    compute_identity_similarity,
)


st.set_page_config(page_title="Local Aesthetic Preview POC", layout="wide")

st.title("Aesthetic Preview POC")


def _fix_orientation(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def _pil_to_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _resize_max(image_rgb: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        return cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image_rgb


def _bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:10]


with st.sidebar:
    st.header("Upload & QC")
    file = st.file_uploader("Upload a selfie (JPG/PNG)", type=["jpg", "jpeg", "png"]) 
    proceed_anyway = st.checkbox("Proceed even if QC fails", value=False)

if file is None:
    st.info("Upload a selfie to begin.")
    st.stop()

# Load image with PIL to respect EXIF orientation
pil = Image.open(file)
orig_pil = _fix_orientation(pil)
image_rgb = _pil_to_rgb(orig_pil)
image_rgb = _resize_max(image_rgb, max_side=1280)

# QC
qc = run_qc_pipeline(image_rgb)

qc_col, img_col = st.columns([1, 2])
with qc_col:
    st.subheader("Quality Checks")
    if not qc.get("face_present", False):
        st.error("No face detected.")
    else:
        st.success("Face detected.")
    if "blur" in qc:
        st.write(f"Blur variance: {qc['blur']['variance']:.1f}")
        st.write("Blur check: ", "✅ Pass" if qc['blur']['pass'] else "❌ Fail")
    if "orientation" in qc:
        st.write(f"Roll: {qc['orientation']['roll_deg']:.1f}°")
        st.write("Orientation: ", "✅ Pass" if qc['orientation']['pass'] else "❌ Fail")
    if "lighting" in qc:
        st.write(f"Lighting mean: {qc['lighting']['mean']:.1f}, std: {qc['lighting']['std']:.1f}")
        st.write("Lighting: ", "✅ Pass" if qc['lighting']['pass'] else "❌ Fail")
    for m in qc.get("messages", []):
        st.warning(m)

with img_col:
    st.image(image_rgb, caption="Original", use_column_width=True)

ok_to_process = qc.get("overall_pass", False) or proceed_anyway
if not ok_to_process:
    st.stop()

# Landmarks and skin mask
face = qc.get("face")
if face is None:
    st.error("Could not compute landmarks.")
    st.stop()

try:
    skin_mask = compute_skin_mask(image_rgb, face)  # uint8 0/255
except Exception:
    st.error("Could not compute skin mask for the detected face.")
    st.stop()

st.subheader("Adjustments")
col1, col2, col3 = st.columns(3)
with col1:
    wrinkle = st.slider("Wrinkle smoothing", 0, 100, 40, step=5)
with col2:
    tone = st.slider("Tone evening", 0, 100, 25, step=5)
with col3:
    pigment = st.slider("Pigmentation reduction", 0, 100, 20, step=5)

# Aesthetic procedures (beta)
st.subheader("Aesthetic procedures (beta)")

# Generative editing toggle (SDXL inpainting). Falls back to classical CV if off/missing deps.
gen_col1, gen_col2, gen_col3 = st.columns([1, 1, 2])
with gen_col1:
    use_gen = st.toggle("Use generative editing", value=True, help="Uses SDXL inpainting on precise masks for high realism. Requires optional ML deps.")
with gen_col2:
    gen_strength = st.slider("Gen strength", 0.10, 0.90, 0.55, 0.05, help="How strongly to apply generative change inside the mask.")
with gen_col3:
    gen_steps = st.slider("Gen steps", 4, 40, 14, 1, help="More steps = higher quality, slower.")

# Advanced controls for prompts and reproducibility
adv = st.expander("Advanced generative controls", expanded=False)
with adv:
    lip_prompt_user = st.text_input("Lip prompt", value=DEFAULT_LIP_PROMPT)
    lip_neg_user = st.text_input("Lip negative prompt", value=DEFAULT_LIP_NEGATIVE_PROMPT)
    eye_prompt_user = st.text_input("Under-eye prompt", value=DEFAULT_EYE_PROMPT)
    eye_neg_user = st.text_input("Under-eye negative prompt", value=DEFAULT_EYE_NEGATIVE_PROMPT)
    seed_user = st.number_input("Seed (optional)", value=42, step=1)

lp_col, ue_col = st.columns(2)
with lp_col:
    lip_enabled = face.get("landmarks_px") is not None
    lip_help = "Requires landmarks. If generative is on, uses SDXL inpaint for realism; otherwise uses fast warp." if lip_enabled else "Requires MediaPipe FaceMesh landmarks."
    lip_int = st.slider("Lip filler", 0, 100, 0, step=5, help=lip_help, disabled=not lip_enabled)
with ue_col:
    ue_enabled = face.get("landmarks_px") is not None
    ue_help = "Requires landmarks. If generative is on, uses SDXL inpaint; otherwise local smoothing/brightening." if ue_enabled else "Requires MediaPipe FaceMesh landmarks."
    ue_int = st.slider("Under-eye reduction", 0, 100, 0, step=5, help=ue_help, disabled=not ue_enabled)

# Process
result_rgb = apply_adjustments_pipeline(
    image_rgb,
    skin_mask,
    wrinkle=wrinkle,
    tone=tone,
    pigment=pigment,
)

# Apply procedures (require landmarks)
if face.get("landmarks_px") is not None:
    result_rgb, proc_info = apply_aesthetic_procedures_pipeline(
        base_rgb=result_rgb,
        image_for_regions=image_rgb,
        face=face,
        use_gen=use_gen,
        gen_strength=gen_strength,
        gen_steps=gen_steps,
        lip_int=lip_int,
        ue_int=ue_int,
        lip_prompt=lip_prompt_user,
        lip_negative_prompt=lip_neg_user,
        eye_prompt=eye_prompt_user,
        eye_negative_prompt=eye_neg_user,
        seed=int(seed_user),
    )
    for msg in proc_info.get("warnings", []):
        st.info(msg)

# Identity similarity via SSIM on face crop
bbox = face["bbox"]
ssim_val = compute_identity_similarity(image_rgb, result_rgb, bbox)

c1, c2 = st.columns(2)
with c1:
    st.image(image_rgb, caption="Before", use_column_width=True)
with c2:
    st.image(result_rgb, caption="After", use_column_width=True)

st.write(f"Identity similarity (SSIM on face crop): {ssim_val:.3f}")

# Download
res_pil = Image.fromarray(result_rgb)
buf = io.BytesIO()
res_pil.save(buf, format="PNG")
bytes_data = buf.getvalue()
hash_id = _bytes_hash(bytes_data)

st.download_button(
    label=f"Download After Image (after_{hash_id}.png)",
    data=bytes_data,
    file_name=f"after_{hash_id}.png",
    mime="image/png",
)


