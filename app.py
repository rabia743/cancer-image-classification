import tempfile
from pathlib import Path

import joblib
from flask import Flask, render_template, request
from PIL import UnidentifiedImageError
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from SVM import (
    CALIBRATOR_PATH,
    ENCODER_PATH,
    MODEL_PATH,
    predict_image_with_confidence,
)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
if not ENCODER_PATH.is_file():
    raise FileNotFoundError(f"Label encoder not found: {ENCODER_PATH}")
if not CALIBRATOR_PATH.is_file():
    raise FileNotFoundError(
        f"Confidence calibrator not found: {CALIBRATOR_PATH}. "
        "Run 'python SVM.py' to train the model and calibrator."
    )

model: LinearSVC = joblib.load(MODEL_PATH)
encoder: LabelEncoder = joblib.load(ENCODER_PATH)
confidence_calibrator = joblib.load(CALIBRATOR_PATH)


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/predict")
def predict() -> tuple[str, int] | str:
    uploaded_image = request.files.get("image")
    if uploaded_image is None or not uploaded_image.filename:
        return render_template(
            "index.html", error="Please choose an image to upload."
        ), 400

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "customer-image"
            uploaded_image.save(image_path)
            prediction, class_probabilities = predict_image_with_confidence(
                image_path, model, encoder, confidence_calibrator
            )
    except UnidentifiedImageError:
        return render_template(
            "index.html",
            error="The uploaded file is not a valid image.",
            uploaded_filename=uploaded_image.filename,
        ), 400

    return render_template(
        "index.html",
        prediction=prediction,
        class_probabilities=class_probabilities,
        uploaded_filename=uploaded_image.filename,
    )


if __name__ == "__main__":
    app.run()
