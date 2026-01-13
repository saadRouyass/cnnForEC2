import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprime les messages INFO et WARNING de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Désactive les messages oneDNN

from fastapi import FastAPI, UploadFile, File,HTTPException
from PIL import Image
import io
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Supprimer les warnings absl
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow import keras

from base64_to_img import base64_to_image 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# =========================
# 1. Initialisation FastAPI
# =========================
app = FastAPI(title="Model API")
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],        # allow all HTTP methods
    allow_headers=["*"],        # allow all headers
)
# =========================
# 2. Chargement du modèle
# =========================
# Le modèle est chargé UNE SEULE FOIS au démarrage
MODEL_PATH = "model/doodle_classifier_model.h5"



model = keras.models.load_model(MODEL_PATH)

# =========================
# 3. Fonction d'inférence
# =========================
def predict(image: Image.Image) -> int:
    image = image.convert("L")          # grayscale
    image = image.resize((64, 64))       # MUST match training size

    x = np.array(image, dtype=np.float32) / 255.0

    # (28, 28) → (28, 28, 1)
    x = np.expand_dims(x, axis=-1)

    # (28, 28, 1) → (1, 28, 28, 1)
    x = np.expand_dims(x, axis=0)

    y = model.predict(x, verbose=0)
    return int(np.argmax(y, axis=1)[0])

def predictBase64(image_path: str) -> int:
    """
    Load an image from disk and run model inference.

    :param image_path: Path to the image file
    :return: Predicted class index
    """

    image = Image.open(image_path).convert("L")  # grayscale
    image = image.resize((64, 64))                # MUST match training size

    x = np.array(image, dtype=np.float32) / 255.0

    # (64, 64) → (64, 64, 1)
    x = np.expand_dims(x, axis=-1)

    # (64, 64, 1) → (1, 64, 64, 1)
    x = np.expand_dims(x, axis=0)

    y = model.predict(x, verbose=0)
    return int(np.argmax(y, axis=1)[0])


# =========================
# 4. Endpoint API
# =========================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prediction = predict(image)

    return {
        "filename": file.filename,
        "prediction": prediction
    }


class Base64ImageRequest(BaseModel):
    image_base64: str

@app.post("/predict/base64")
async def predict_image_base64(payload: Base64ImageRequest):
    try:
        # Remove data URI header if present
        base64_str = payload.image_base64
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        base64_to_image(base64_str,"img.png")
        prediction = predictBase64("img.png")

        return {
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 image: {str(e)}")