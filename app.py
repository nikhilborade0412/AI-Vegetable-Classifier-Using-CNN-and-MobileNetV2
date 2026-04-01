import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import cv2

from veg_info import veg_info
from recipe_info import recipe_info


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Vegetable Classifier",
    page_icon="🥦",
    layout="wide"
)

st.markdown("""
<style>
.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🥦 AI Vegetable Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Capture a vegetable photo — the AI will detect, highlight, and identify it.</p>', unsafe_allow_html=True)
st.divider()


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_transfer_model.keras")

model = load_model()

class_names = [
    'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_entropy(prediction):
    """Shannon entropy — no scipy needed."""
    p = np.clip(prediction, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))


def is_vegetable(prediction, confidence_threshold=0.60, entropy_threshold=2.5):
    top_confidence = float(np.max(prediction))
    pred_entropy   = compute_entropy(prediction)
    if top_confidence < confidence_threshold:
        return False, f"Low confidence ({top_confidence*100:.1f}%). This may not be a vegetable."
    if pred_entropy > entropy_threshold:
        return False, f"High uncertainty (entropy={pred_entropy:.2f}). Image doesn't look like a vegetable."
    return True, ""


def detect_vegetable_bbox(pil_image):
    """
    Detects the main object (vegetable) in the image using:
    1. Convert to BGR for OpenCV
    2. GrabCut for foreground segmentation
    3. Find the largest contour → bounding box
    Returns (x, y, w, h) or None if nothing found.
    """
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]

    # ── GrabCut foreground segmentation ──────────────────────────────────────
    mask        = np.zeros((h, w), np.uint8)
    bgd_model   = np.zeros((1, 65), np.float64)
    fgd_model   = np.zeros((1, 65), np.float64)

    # Initial rect: 10% margin from each edge
    margin = int(min(h, w) * 0.10)
    rect   = (margin, margin, w - 2 * margin, h - 2 * margin)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    # Pixels marked as probable/definite foreground
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")

    # ── Morphological cleanup ─────────────────────────────────────────────────
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # ── Find largest contour ──────────────────────────────────────────────────
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: use edge detection
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        edges   = cv2.Canny(blurred, 30, 120)
        kernel2 = np.ones((15, 15), np.uint8)
        edges   = cv2.dilate(edges, kernel2, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    # Skip if contour is too small (< 2% of image area)
    if cv2.contourArea(largest) < 0.02 * h * w:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)

    # Add small padding (5%)
    pad  = int(min(bw, bh) * 0.05)
    x    = max(0, x - pad)
    y    = max(0, y - pad)
    bw   = min(w - x, bw + 2 * pad)
    bh   = min(h - y, bh + 2 * pad)

    return x, y, bw, bh


def draw_bounding_box(pil_image, bbox, label, confidence, color=(0, 220, 100)):
    """Draw a styled bounding box with label onto a PIL image."""
    img_draw  = pil_image.copy()
    draw      = ImageDraw.Draw(img_draw)
    x, y, bw, bh = bbox

    # Box border (thick, 4px)
    for t in range(4):
        draw.rectangle(
            [x - t, y - t, x + bw + t, y + bh + t],
            outline=color
        )

    # Corner accents (L-shaped ticks)
    tick = 20
    for (cx, cy, dx, dy) in [
        (x,        y,        1,  1),
        (x + bw,   y,       -1,  1),
        (x,        y + bh,   1, -1),
        (x + bw,   y + bh,  -1, -1),
    ]:
        draw.line([(cx, cy), (cx + dx * tick, cy)],           fill=color, width=4)
        draw.line([(cx, cy), (cx,             cy + dy * tick)], fill=color, width=4)

    # Label background + text
    label_text = f"  {label}  {confidence*100:.1f}%  "
    text_bbox  = draw.textbbox((0, 0), label_text)
    text_w     = text_bbox[2] - text_bbox[0]
    text_h     = text_bbox[3] - text_bbox[1]
    label_y    = max(0, y - text_h - 10)

    draw.rectangle(
        [x, label_y, x + text_w + 4, label_y + text_h + 6],
        fill=color
    )
    draw.text((x + 2, label_y + 2), label_text, fill=(0, 0, 0))

    return img_draw


def preprocess_crop(pil_image, bbox=None):
    """Crop to bbox if available, then preprocess for model."""
    if bbox:
        x, y, bw, bh = bbox
        pil_image = pil_image.crop((x, y, x + bw, y + bh))
    img = pil_image.resize((128, 128)).convert("RGB")
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📷 Input Options")

option = st.sidebar.radio(
    "Select Image Source",
    ("📁 Upload Image", "📸 Use Camera")
)

st.sidebar.divider()
st.sidebar.markdown("#### ⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.30, 0.90, 0.60, 0.05,
    help="Minimum confidence to accept a prediction as a vegetable.")
show_bbox = st.sidebar.checkbox("Show Bounding Box", value=True,
    help="Detect and highlight the vegetable region in the image.")

image = None

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload Vegetable Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

if option == "📸 Use Camera":
    st.info("📸 Point your camera at a vegetable and click **Take Photo**.")
    camera_image = st.camera_input("Capture Vegetable Image")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")


# ── Main prediction flow ──────────────────────────────────────────────────────
if image:

    col1, col2 = st.columns([1, 1])

    bbox = None

    # ── Bounding box detection ────────────────────────────────────────────────
    if show_bbox:
        with st.spinner("🔍 Detecting vegetable region..."):
            bbox = detect_vegetable_bbox(image)

    # ── Run prediction (on cropped region if bbox found) ──────────────────────
    img_array  = preprocess_crop(image, bbox)
    prediction = model.predict(img_array, verbose=0)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence      = float(prediction[predicted_index])

    valid, reason = is_vegetable(prediction,
                                  confidence_threshold=confidence_threshold)

    # ── Draw bounding box on image ────────────────────────────────────────────
    display_image = image
    if show_bbox and bbox and valid:
        display_image = draw_bounding_box(
            image, bbox,
            label=predicted_class,
            confidence=confidence,
            color=(0, 220, 100)
        )
    elif show_bbox and bbox and not valid:
        # Draw red box when not a vegetable
        display_image = draw_bounding_box(
            image, bbox,
            label="Unknown",
            confidence=confidence,
            color=(220, 50, 50)
        )

    with col1:
        st.markdown("#### 🖼️ Detected Image")
        st.image(display_image, use_container_width=True)

        if bbox and show_bbox:
            x, y, bw, bh = bbox
            st.caption(f"📦 Bounding box: x={x}, y={y}, w={bw}, h={bh}")
        elif show_bbox:
            st.caption("⚠️ No clear object region detected — predicting on full image.")

    # ── Results panel ─────────────────────────────────────────────────────────
    with col2:
        st.markdown("### 🌟 Prediction Result")

        if not valid:
            st.error("❌ This does not appear to be a vegetable image!")
            st.warning(f"**Reason:** {reason}")
            st.info(
                "💡 **Tips for best results:**\n\n"
                "- Upload a clear, well-lit photo of a **single vegetable**\n"
                "- Make sure the vegetable **fills most of the frame**\n"
                "- Avoid non-vegetable images (clothes, people, logos, etc.)\n\n"
                f"**Supported vegetables:** {', '.join(class_names)}"
            )

        else:
            st.success(f"**🥦 Vegetable:** {predicted_class}")
            st.info(f"**📊 Confidence:** {confidence * 100:.2f}%")
            st.progress(confidence, text=f"Model confidence: {confidence*100:.1f}%")

            st.divider()

            # Top 3
            top3 = np.argsort(prediction)[::-1][:3]
            st.markdown("#### 📊 Top 3 Predictions")
            for idx in top3:
                bar_val = float(prediction[idx])
                st.write(f"**{class_names[idx]}** — {bar_val*100:.2f}%")
                st.progress(bar_val)

            st.divider()

            # Vegetable info
            if predicted_class in veg_info:
                st.markdown("## 🥗 Vegetable Information")
                st.write(veg_info[predicted_class]["info"])
                st.divider()

                st.markdown("## 🍎 Nutritional Values (per 100g)")
                nutrition = veg_info[predicted_class]["nutrition"]
                df = pd.DataFrame(nutrition.items(), columns=["Nutrient", "Value"])
                st.table(df)

            # Recipes
            if predicted_class in recipe_info:
                st.divider()
                st.markdown("## 🍲 Recommended Recipes")
                for recipe in recipe_info[predicted_class]:
                    st.write("•", recipe)