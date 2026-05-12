import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import time
import os
import random
import zipfile
from huggingface_hub import hf_hub_download

# =========================================================
# 🚀 DOWNLOAD MODEL & DATASET
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------------------------------------
# Download Model
# ---------------------------------------------------------

if not os.path.exists("best_malaria_model.keras") and HF_TOKEN:

    try:

        hf_hub_download(
            repo_id="paulaman1/Malaria-classifier",
            filename="best_malaria_model.keras",
            token=HF_TOKEN,
            local_dir="."
        )

        print("✅ Model downloaded successfully.")

    except Exception as e:

        print("❌ Model download error:", e)

# ---------------------------------------------------------
# Download Dataset
# ---------------------------------------------------------

if not os.path.exists("extracted_data") and HF_TOKEN:

    try:

        zip_path = hf_hub_download(
            repo_id="paulaman1/Malaria-dataset",
            filename="Dataset.zip",
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="."
        )

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:

            zip_ref.extractall("extracted_data")

        print("✅ Dataset extracted successfully.")

    except Exception as e:

        print("❌ Dataset extraction error:", e)

# =========================================================
# 🧠 LOAD MODEL
# =========================================================

try:

    model = tf.keras.models.load_model(
        "best_malaria_model.keras"
    )

    print("✅ Model loaded successfully.")

except Exception as e:

    model = None

    print("❌ Failed to load model:", e)

# =========================================================
# 🔬 CLASSIFICATION FUNCTION
# =========================================================

def classify_cell(image):

    # -----------------------------------------------------
    # BASIC CHECKS
    # -----------------------------------------------------

    if image is None:

        return "Please upload an image.", "0.00 ms"

    if model is None:

        return "Model failed to load.", "0.00 ms"

    start_time = time.perf_counter()

    # -----------------------------------------------------
    # PREPROCESSING
    # IMPORTANT:
    # DO NOT divide by 255 here
    # because model already has:
    # Rescaling(1./255)
    # -----------------------------------------------------

    img = cv2.resize(
        image,
        (100, 100)
    )

    img = img.astype("float32")

    img = np.expand_dims(
        img,
        axis=0
    )

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    try:

        prediction = float(
            model.predict(
                img,
                verbose=0
            )[0][0]
        )

    except Exception as e:

        return (
            f"Prediction Error: {e}",
            "0.00 ms"
        )

    raw_output = prediction

    inference_time = (
        time.perf_counter() - start_time
    ) * 1000

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = max(
        prediction,
        1 - prediction
    ) * 100

    # -----------------------------------------------------
    # LOW CONFIDENCE / UNKNOWN IMAGE
    # -----------------------------------------------------

    if confidence < 85:

        return (
            "🚫 Invalid / Unknown Image\n\n"
            "The uploaded image does not appear to be "
            "a clear malaria single-cell microscopy sample.",
            f"{inference_time:.2f} ms"
        )

    # -----------------------------------------------------
    # UNCERTAIN ZONE
    # -----------------------------------------------------

    if 0.45 <= prediction <= 0.55:

        return (
            f"⚠️ Uncertain Prediction\n\n"
            f"Raw Output: {raw_output:.4f}\n"
            f"Confidence: {confidence:.2f}%",
            f"{inference_time:.2f} ms"
        )

    # -----------------------------------------------------
    # FINAL CLASSIFICATION
    # -----------------------------------------------------

    # ASSUMPTION:
    # 0 = Parasitized
    # 1 = Uninfected

    if prediction > 0.5:

        result = "✅ Uninfected (Healthy Cell)"

    else:

        result = "🔬 Infected (Parasite Detected)"

    return (
        f"{result}\n\n"
        f"Confidence: {confidence:.2f}%\n"
        f"Raw Output: {raw_output:.4f}",
        f"{inference_time:.2f} ms"
    )

# =========================================================
# 🖼️ TEST DATASET GALLERY ONLY
# =========================================================

def get_random_samples():

    infected = []
    uninfected = []

    # -----------------------------------------------------
    # TEST DATASET PATH
    # -----------------------------------------------------

    TEST_PATH = os.path.join(
        "extracted_data",
        "Dataset",
        "Test"
    )

    print("📂 TEST PATH:", TEST_PATH)

    if not os.path.exists(TEST_PATH):

        print("❌ Test path not found.")

        return infected, uninfected

    for root, _, files in os.walk(TEST_PATH):

        for file in files:

            if file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):

                path = os.path.join(root, file)

                folder = root.lower()

                if "uninfected" in folder:

                    uninfected.append(path)

                elif (
                    "infected" in folder
                    or "parasitized" in folder
                ):

                    infected.append(path)

    infected = random.sample(
        infected,
        min(5, len(infected))
    )

    uninfected = random.sample(
        uninfected,
        min(5, len(uninfected))
    )

    return infected, uninfected

# =========================================================
# 🖱️ GALLERY CLICK HANDLER
# =========================================================

def set_img(evt: gr.SelectData):

    if (
        isinstance(evt.value, dict)
        and 'image' in evt.value
    ):

        return evt.value['image']['path']

    return evt.value

# =========================================================
# 🎨 UI DESIGN
# =========================================================

with gr.Blocks(
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        "# 🩺 Malaria Cell Classification System"
    )

    gr.Markdown(
        """
### AI-Based Single-Cell Malaria Detection Prototype

Upload a malaria microscopy single-cell image to classify:

- 🔬 Infected (Parasitized)
- ✅ Uninfected (Healthy)

---
⚠️ DISCLAIMER

- This is a research and educational prototype.
- Only malaria microscopy single-cell images should be uploaded.
- Uploading random objects, natural images, or unrelated content may produce unreliable predictions.
- Please verify once from doctor or medical authorities
"""
    )

    with gr.Row():

        with gr.Column():

            img_input = gr.Image(
                type="numpy",
                label="Upload Malaria Cell Image",
                height=300
            )

            classify_btn = gr.Button(
                "Analyze Cell",
                variant="primary"
            )

        with gr.Column():

            output_text = gr.Textbox(
                label="Prediction Result",
                lines=8
            )

            time_text = gr.Textbox(
                label="Inference Time"
            )

    gr.Markdown("---")

    gr.Markdown(
        "## 🔀 Random Test Dataset Samples"
    )

    with gr.Row():

        inf_gallery = gr.Gallery(
            label="🔬 Infected Test Samples",
            columns=5,
            height=150
        )

        uninf_gallery = gr.Gallery(
            label="✅ Uninfected Test Samples",
            columns=5,
            height=150
        )

    refresh_btn = gr.Button(
        "🔄 Refresh Test Samples"
    )

    # -----------------------------------------------------
    # LOAD TEST SAMPLES
    # -----------------------------------------------------

    demo.load(
        fn=get_random_samples,
        outputs=[
            inf_gallery,
            uninf_gallery
        ]
    )

    refresh_btn.click(
        fn=get_random_samples,
        outputs=[
            inf_gallery,
            uninf_gallery
        ]
    )

    # -----------------------------------------------------
    # GALLERY CLICK EVENTS
    # -----------------------------------------------------

    inf_gallery.select(
        fn=set_img,
        outputs=img_input
    ).then(
        fn=classify_cell,
        inputs=img_input,
        outputs=[
            output_text,
            time_text
        ]
    )

    uninf_gallery.select(
        fn=set_img,
        outputs=img_input
    ).then(
        fn=classify_cell,
        inputs=img_input,
        outputs=[
            output_text,
            time_text
        ]
    )

    # -----------------------------------------------------
    # ANALYZE BUTTON
    # -----------------------------------------------------

    classify_btn.click(
        fn=classify_cell,
        inputs=img_input,
        outputs=[
            output_text,
            time_text
        ]
    )

# =========================================================
# 🚀 LAUNCH APPLICATION
# =========================================================

demo.launch()