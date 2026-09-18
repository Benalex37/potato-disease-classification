# Potato Disease Classification

A convolutional neural network that reads a photo of a potato leaf and reports whether the plant has **early blight**, **late blight**, or is **healthy**.

Both diseases are treatable, but they need different treatment and both spread fast. Telling them apart normally takes an agronomist. This project puts that call in the hands of anyone with a phone camera.

The repository carries the whole path from raw data to something a farmer could open: the training notebooks, the trained model in four formats, a REST API, a web UI, a mobile app, and several deployment targets.

## Demo

**[Try it live](https://potato-disease-classification-hzy52svuf8gsxqzcqtpgta.streamlit.app/)** — no install needed.

Upload a leaf photo and the app returns the predicted class with a confidence score for all three categories.

## The model

A small CNN trained from scratch — no transfer learning.

| | |
|---|---|
| Architecture | 6 × (Conv2D → MaxPooling2D) → Flatten → Dense → Dense |
| Parameters | 183,747 |
| Input | 256 × 256 RGB |
| Classes | Early Blight, Late Blight, Healthy |
| Training | 50 epochs, batch size 32 |
| Test accuracy | 100% |
| Dataset | [PlantVillage](https://www.kaggle.com/arjuntejaswi/plant-village) (potato subset), ~2,150 images |

Resizing and rescaling are layers *inside* the model, so it takes raw 0–255 pixel values rather than requiring callers to normalize first.

### On that 100% accuracy

It is real, and it is also less impressive than it sounds. PlantVillage images are single leaves shot against a plain background under even lighting. Field photos have soil, other leaves, shadows and glare, and the model has never seen any of that.

You can watch it fail, in both directions and without hesitation:

- `test_images_from_internet/late_blight_1.jpg` is a textbook late blight lesion. The model says Early Blight, 73.9%.
- A clear early blight leaf — the concentric target-shaped lesions the disease is named for — comes back as Late Blight at 100.0%.

That second one is the important one. The model is not uncertain and wrong; it is confident and wrong, which is worse, because the confidence score gives you no warning. Read it as a rough signal, never as a diagnosis.

Fixing this means retraining on field photography, not tuning the existing model. The ceiling here is the dataset, not the architecture.

## What's in the repository

| Path | What it is |
|---|---|
| `training/` | Jupyter notebooks — data loading, augmentation, training, and TFLite conversion (including quantization-aware training) |
| `saved_models/1,2,3` | TensorFlow SavedModel exports, one per training run |
| `potatoes.h5` | Keras H5 export, used by the Google Cloud Function |
| `potato_model.tflite` | TFLite build used by the Streamlit app (724 KB) |
| `tf-lite-models/` | Earlier TFLite exports for the mobile app |
| `streamlit_app.py` | Single-file web app — upload, classify, show confidence |
| `api/main.py` | FastAPI server that loads the model in-process |
| `api/main-tf-serving.py` | Same API, but delegating inference to TensorFlow Serving |
| `frontend/` | React web UI with drag-and-drop upload |
| `mobile-app/` | React Native app for Android and iOS |
| `gcp/` | Google Cloud Function handlers for serverless deployment |
| `test_images_from_internet/` | Two sample leaf photos for quick testing |

## Running it

### Streamlit app — quickest

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. No TensorFlow required: it runs the TFLite model through LiteRT, which installs in seconds instead of minutes.

### FastAPI + React

Two terminals.

```bash
# terminal 1 — API on :8000
cd api
pip install fastapi uvicorn python-multipart pillow "tensorflow==2.15.1" "numpy<2"
python main.py
```

```bash
# terminal 2 — web UI on :3000
cd frontend
npm install --legacy-peer-deps
cp .env.example .env          # set REACT_APP_API_URL=http://localhost:8000/predict
npm start
```

On Windows, `start-api.bat` and `start-frontend.bat` do both.

Test the API directly:

```bash
curl -X POST -F "file=@test_images_from_internet/early_blight_1.jpg" \
  http://localhost:8000/predict
# {"class":"Early Blight","confidence":1.0}
```

### Version requirements

These are tighter than they look, and getting them wrong is the most common way to waste an hour here.

- **TensorFlow 2.15** is the newest release that can load `saved_models/` and `potatoes.h5`. They were exported under Keras 2 in 2021; Keras 3 refuses them.
- **Python ≤ 3.11** follows from that — TensorFlow 2.15 publishes no wheels for 3.12+.
- **`saved_models/1` and `2` will not load at all.** Their in-model `RandomFlip` layers were exported without a serialized call function. Use `saved_models/3`.
- **The Streamlit app escapes all of this** by running TFLite instead, which is why it has no TensorFlow dependency and no version pins.

## Training

```bash
pip install -r training/requirements.txt
jupyter notebook
```

Open `training/potato-disease-classification-model.ipynb`, point cell 2 at your PlantVillage download, and run through. Save the result into `saved_models/` under the next version number.

`training/tf-lite-converter.ipynb` converts a trained model to TFLite; the quantization-aware variant trades a little accuracy for a smaller file.

## Deployment

| Target | Notes |
|---|---|
| Streamlit Community Cloud | Point it at this repo, main file `streamlit_app.py`. Nothing else to configure. |
| Google Cloud Functions | `cd gcp && gcloud functions deploy predict_lite --runtime python38 --trigger-http --memory 512`. Upload the model to a GCS bucket first and set `BUCKET_NAME` in `gcp/main.py`. |
| TensorFlow Serving | `docker run -p 8501:8501 -v $(pwd):/models tensorflow/serving --model_config_file=/models/models.config`, then run `api/main-tf-serving.py`. Copy `models.config.example` first. |
| Mobile | `cd mobile-app && npm install && npm run android` (or `ios`). Bundles the TFLite model for on-device inference. |

## Dataset

[PlantVillage](https://www.kaggle.com/arjuntejaswi/plant-village) on Kaggle. Download it, keep only the three potato folders, and point the training notebook at that directory.
