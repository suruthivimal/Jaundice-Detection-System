# Jaundice Detection in Newborns (Image-Based)

This project is a Flask-based web application that detects jaundice in newborn babies by analyzing facial images. It uses a Convolutional Neural Network (CNN) trained on labeled image data to classify between "normal" and "jaundice" conditions.

---

## 🧠 Model Overview

The model is trained on facial images of newborns with two labels:

* `normal` (label 0)
* `jaundice` (label 1)

### 🔧 Key Steps:

1. Images are resized to **128x128** using OpenCV.
2. Data is loaded and labeled using NumPy.
3. The model is a CNN (defined in the notebook `jaundice in new born baby.ipynb`).
4. Final model is saved for prediction in Flask app.

---

## 📂 Project Structure

```
├── app.py                            # Flask backend with image upload and prediction logic
├── templates/
│   └── index.html                    # HTML UI for uploading images (served via Flask)
├── static/
│   └── style.css                     # CSS for styling
├── jaundice in new born baby.ipynb   # Model training notebook (CNN using image dataset)
├── processed_dataset/                # Resized and labeled training images
├── jaundice_model.keras              # Trained CNN model used in app.py
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
```

---

## 🚀 Running the Application

### 1. Clone the Repository

```bash
git clone https://github.com/Suruthimeow13/jaundice-detection.git
cd jaundice-detection
```

### 2. Set Up Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate           # macOS/Linux
venv\Scripts\activate              # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask App

```bash
python app.py
```

### 5. Open in Browser

```bash
http://127.0.0.1:5000/
```

---

## 📸 Application Screenshots

### 🏠 Home Page

The homepage allows users to upload a newborn's forehead image for jaundice screening. It also provides image capture guidelines and information about neonatal jaundice.

Markdown:

![Home Page](screenshots/Screenshot%202026-07-03%20230921.png)

---

### 🏥 Nearby Hospitals

If the model predicts severe jaundice, the application helps users locate nearby pediatric hospitals based on their location.

![Nearby Hospitals](screenshots/Screenshot%202026-07-03%20231056.png)


---

### 📄 Diagnostic Report

After prediction, the application generates a downloadable PDF medical report containing:

- Patient information
- Classification result
- Confidence score
- Severity level
- Clinical recommendation

![Diagnostic Report](screenshots/Screenshot%202026-07-03%20231110.png)

## 📎 Requirements

Add these to `requirements.txt`:

```txt
Flask
numpy
opencv-python
matplotlib
tensorflow
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 📸 Dataset Notes

* Stored locally at: `jaundice_dataset/`
* Two folders:

  * `normal/` – images of non-jaundiced babies
  * `jaundice/` – images of babies with jaundice

Processed dataset saved to: `processed_dataset/`

---

## 👤 Author

**Suruthivimal**
GitHub: [suruthivimal](https://github.com/suruthivimal)

---
