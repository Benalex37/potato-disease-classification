"""Streamlit UI for the potato leaf disease classifier.

Runs the TFLite export of saved_models/3 rather than the Keras model, which
keeps TensorFlow out of the dependency list entirely: the interpreter is a
few MB and has wheels for every current Python, whereas the Keras format
these models were saved in (2021, Keras 2) pins you to tensorflow 2.15 and
Python <=3.11. Predictions are identical to api/main.py.
"""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # local dev machines that already have tensorflow
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

MODEL_PATH = Path(__file__).resolve().parent / "potato_model.tflite"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
IMAGE_SIZE = 256

st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔")


@st.cache_resource(show_spinner="Loading model…")
def load_interpreter():
    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


def resize_bilinear(image: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize matching tf.image.resize (half-pixel centers, no antialias).

    Pillow's own bilinear applies an antialiasing filter and gives visibly
    different confidences, so the model's training-time preprocessing is
    reproduced here instead.
    """
    def axis(n_in, n_out):
        pos = np.clip((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0, n_in - 1)
        low = np.floor(pos).astype(int)
        return low, np.minimum(low + 1, n_in - 1), (pos - low)[:, None]

    h, w = image.shape[:2]
    ly, hy, wy = axis(h, size)
    lx, hx, wx = axis(w, size)
    image = image.astype(np.float32)
    rows = image[ly] * (1 - wy[:, :, None]) + image[hy] * wy[:, :, None]
    return rows[:, lx] * (1 - wx.T[:, :, None]) + rows[:, hx] * wx.T[:, :, None]


def predict(interpreter, image: Image.Image) -> np.ndarray:
    # Rescaling to 0-1 happens inside the model, so feed raw 0-255 values.
    pixels = resize_bilinear(np.array(image.convert("RGB")), IMAGE_SIZE)
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], pixels[None].astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])[0]


st.title("🥔 Potato Disease Classifier")
st.caption("Upload a potato leaf photo to check it for early blight or late blight.")

uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a leaf image to get a prediction.")
    st.stop()

image = Image.open(uploaded)
# Streamlit renamed this argument in 1.41; support both so the pinned
# local install and the Cloud's latest release each work.
try:
    st.image(image, caption=uploaded.name, use_container_width=True)
except TypeError:
    st.image(image, caption=uploaded.name, use_column_width=True)

scores = predict(load_interpreter(), image)
top = int(np.argmax(scores))

st.subheader(CLASS_NAMES[top])
st.metric("Confidence", f"{scores[top]:.1%}")

st.write("All classes")
for name, score in zip(CLASS_NAMES, scores):
    st.progress(float(score), text=f"{name} — {score:.1%}")

st.caption(
    "Trained on the PlantVillage dataset of clean lab images, so accuracy drops "
    "on ordinary photos taken in the field."
)
