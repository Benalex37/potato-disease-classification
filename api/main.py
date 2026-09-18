from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# saved_models/1 and /2 fail to load on modern Keras: their in-model
# RandomFlip/RandomRotation layers were exported without a serialized call
# function. Version 3 has no augmentation layers, so it loads cleanly.
MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models" / "3"
MODEL = tf.keras.models.load_model(MODEL_DIR)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
IMAGE_SIZE = 256

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    # The model's input is a fixed (256, 256, 3), so uploads have to be
    # resized here; only 3-channel RGB is accepted (drops PNG alpha, etc).
    image = Image.open(BytesIO(data)).convert("RGB")
    return tf.image.resize(np.array(image), [IMAGE_SIZE, IMAGE_SIZE]).numpy()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
