"""Streamlit UI for the potato leaf disease classifier.

Mirrors what api/main.py does, minus the HTTP hop: the model is loaded once
per session and inference runs in-process.
"""
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# saved_models/1 and /2 can't be loaded by current Keras: their in-model
# RandomFlip layers were exported without a serialized call function.
MODEL_DIR = Path(__file__).resolve().parent / "saved_models" / "3"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
IMAGE_SIZE = 256

st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔")


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return tf.keras.models.load_model(MODEL_DIR)


def predict(model, image: Image.Image):
    # The model's input is a fixed 256x256x3; rescaling happens inside it.
    resized = tf.image.resize(np.array(image.convert("RGB")), [IMAGE_SIZE, IMAGE_SIZE])
    scores = model.predict(np.expand_dims(resized.numpy(), 0), verbose=0)[0]
    return scores


st.title("🥔 Potato Disease Classifier")
st.caption("Upload a potato leaf photo to check it for early blight or late blight.")

uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a leaf image to get a prediction.")
    st.stop()

image = Image.open(uploaded)
st.image(image, caption=uploaded.name, use_column_width=True)

scores = predict(load_model(), image)
top = int(np.argmax(scores))

st.subheader(f"{CLASS_NAMES[top]}")
st.metric("Confidence", f"{scores[top]:.1%}")

st.write("All classes")
for name, score in zip(CLASS_NAMES, scores):
    st.progress(float(score), text=f"{name} — {score:.1%}")

st.caption(
    "Trained on the PlantVillage dataset of clean lab images, so accuracy drops "
    "on ordinary photos taken in the field."
)
