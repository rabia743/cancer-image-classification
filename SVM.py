import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (128, 128)
MODEL_PATH = BASE_DIR / "svm_model.pkl"
ENCODER_PATH = BASE_DIR / "cancer_label_encoder.pkl"
CALIBRATOR_PATH = BASE_DIR / "cancer_confidence_calibrator.pkl"


def _image_features(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        return np.asarray(image, dtype=np.float32).reshape(1, -1) / 255.0


def predict_image(
    image_path: Path, model: LinearSVC, encoder: LabelEncoder
) -> str:
    features = _image_features(image_path)
    encoded_prediction = model.predict(features)
    return str(encoder.inverse_transform(encoded_prediction)[0])


def predict_image_with_confidence(
    image_path: Path,
    model: LinearSVC,
    encoder: LabelEncoder,
    calibrator: LogisticRegression,
) -> tuple[str, dict[str, float]]:
    features = _image_features(image_path)
    encoded_prediction = model.predict(features)
    decision_score = model.decision_function(features).reshape(-1, 1)
    probabilities = calibrator.predict_proba(decision_score)[0]
    labels = encoder.inverse_transform(calibrator.classes_.astype(int))
    class_probabilities = {
        str(label): float(probability)
        for label, probability in zip(labels, probabilities)
    }
    prediction = str(encoder.inverse_transform(encoded_prediction)[0])
    return prediction, class_probabilities


def fit_confidence_calibrator(
    model: LinearSVC, X_calibration: np.ndarray, y_calibration: np.ndarray
) -> LogisticRegression:
    calibration_scores = model.decision_function(X_calibration)
    calibrator = LogisticRegression()
    calibrator.fit(calibration_scores.reshape(-1, 1), y_calibration)
    return calibrator


def train_and_save_model(verbose: bool = False) -> tuple[LinearSVC, LabelEncoder]:
    X_train = np.load(BASE_DIR / "cancer_train_images.npy").reshape(-1, 128 * 128 * 3)
    y_train = np.load(BASE_DIR / "cancer_train_labels.npy")
    X_test = np.load(BASE_DIR / "cancer_test_images.npy").reshape(-1, 128 * 128 * 3)
    y_test = np.load(BASE_DIR / "cancer_test_labels.npy")
    encoder = joblib.load(ENCODER_PATH)

    model = LinearSVC(
        class_weight="balanced",
        max_iter=10000,
        random_state=42,
        verbose=1 if verbose else 0,
    )
    print("Training SVM...", flush=True)
    model.fit(X_train, y_train)
    print("SVM training complete. Evaluating...", flush=True)

    test_predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, test_predictions):.2%}")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    print("Calibrating model scores on held-out images...", flush=True)
    calibrator = fit_confidence_calibrator(model, X_test, y_test)

    print("Saving model...", flush=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Confidence calibrator saved: {CALIBRATOR_PATH}")
    return model, encoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and save an SVM, or predict an image with the saved model."
    )
    parser.add_argument(
        "--predict",
        type=Path,
        metavar="IMAGE",
        help="Predict this image using the previously saved model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show training progress output from scikit-learn (prints dots).",
    )
    args = parser.parse_args()

    if args.predict is not None:
        if not MODEL_PATH.is_file():
            print(
                f"Model not found at {MODEL_PATH}. Training a simple SVM and saving it first...",
                flush=True,
            )
            train_and_save_model(verbose=args.verbose)
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        image_path = args.predict.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        prediction = predict_image(image_path, model, encoder)
        print(f"Prediction for {image_path}: {prediction}")
        return

    train_and_save_model(verbose=args.verbose)


if __name__ == "__main__":
    main()
