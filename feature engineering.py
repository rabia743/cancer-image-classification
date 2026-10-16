import joblib
import numpy as np
import pandas as pd
from pathlib import Path, PurePosixPath
from PIL import Image
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
IMAGE_SIZE = (128, 128)

cancer = pd.read_csv(BASE_DIR / "cancer.csv")
required_columns = {"images", "labels"}
if not required_columns.issubset(cancer.columns):
    raise ValueError("cancer.csv must contain images and labels columns.")
if cancer[list(required_columns)].isnull().any().any():
    raise ValueError("cancer.csv contains empty image paths or labels.")
if cancer["images"].duplicated().any():
    duplicates = cancer.loc[cancer["images"].duplicated(), "images"].tolist()
    raise ValueError(f"cancer.csv contains duplicate image paths: {duplicates[:10]}")


def get_split(image_path: str) -> str:
    parts = PurePosixPath(image_path).parts
    if len(parts) < 4 or parts[0] != "data" or parts[1] not in SPLITS:
        raise ValueError(
            f"Image path must be under data/train, data/val, or data/test: {image_path}"
        )
    return parts[1]


cancer["split"] = cancer["images"].map(get_split)

# Fit on training labels and reuse the same mapping for validation and test.
label_encoder = LabelEncoder()
train_labels = cancer.loc[cancer["split"] == "train", "labels"]
if train_labels.empty:
    raise ValueError("cancer.csv does not contain any training images.")
label_encoder.fit(train_labels)
unknown_labels = set(cancer["labels"]) - set(label_encoder.classes_)
if unknown_labels:
    raise ValueError(f"Labels not found in training data: {sorted(unknown_labels)}")

print("Dataset loaded successfully.")
print("Encoded labels:", label_encoder.classes_)

arrays = {}
for split in SPLITS:
    split_rows = cancer.loc[cancer["split"] == split]
    if split_rows.empty:
        raise ValueError(f"cancer.csv does not contain any {split} images.")

    images = []
    for image_path in split_rows["images"]:
        relative_path = PurePosixPath(str(image_path))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe image path in cancer.csv: {image_path}")

        full_path = (BASE_DIR / Path(*relative_path.parts)).resolve()
        if BASE_DIR not in full_path.parents:
            raise ValueError(f"Image path is outside the project folder: {image_path}")
        if not full_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {full_path}\n"
                "Download the image listed in cancer.csv before running this script."
            )

        with Image.open(full_path) as image:
            image = image.convert("RGB").resize(IMAGE_SIZE)
            images.append(np.asarray(image, dtype=np.float32) / 255.0)

    X = np.stack(images).astype(np.float32, copy=False)
    y = label_encoder.transform(split_rows["labels"])
    arrays[split] = (X, y)

for split, (X, y) in arrays.items():
    np.save(BASE_DIR / f"cancer_{split}_images.npy", X)
    np.save(BASE_DIR / f"cancer_{split}_labels.npy", y)
    print(f"{split}: images {X.shape}, labels {y.shape}")

joblib.dump(label_encoder, BASE_DIR / "cancer_label_encoder.pkl")
print("Split image arrays, label arrays, and label encoder saved successfully.")
