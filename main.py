import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprime les messages INFO et WARNING de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Désactive les messages oneDNN

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Supprimer les warnings absl
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow import keras

# =========================
# 1. Initialisation FastAPI
# =========================
app = FastAPI(title="Model API")

# =========================
# 2. Chargement du modèle
# =========================
# Le modèle est chargé UNE SEULE FOIS au démarrage
MODEL_PATH = "doodle_classifier_model.h5"



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
