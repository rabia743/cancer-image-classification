# Breast Cancer Image Classifier

A machine learning project that classifies breast cancer histopathology images as **Benign** or **Malignant** using a LinearSVC model and a Flask web application.

## Features
-  Image preprocessing (128 × 128 RGB)
-  LinearSVC machine learning model
-  Calibrated confidence scores (if implemented)
-  Flask web interface for image predictions
-  Saved model for reuse

## Technologies
Python, Pandas, NumPy, Scikit-learn, Pillow, Flask, Joblib, HTML, and CSS.

##  Run the Project

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

##  Dataset
The `cancer.csv` file contains image paths and labels:
- `images` — image file paths
- `labels` — benign or malignant

##  Disclaimer
This project is for educational purposes only. It is not a medical device and must not be used for clinical diagnosis.

## Author
**Rabia Zafar**  
Python Developer | Machine Learning Learner
