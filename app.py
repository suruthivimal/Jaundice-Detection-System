from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

app = Flask(__name__)
model = load_model('unet_jaundice_classifier.keras')

UPLOAD_FOLDER = 'static/uploads'
REPORT_PATH = 'static/report.pdf'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    severity = None
    recommendation = None

    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(path)

            img = preprocess_image(path)
            pred = model.predict(img)[0][0]
            confidence = round(float(pred if pred >= 0.5 else 1 - pred), 2)

            if pred < 0.5:
                prediction = "Jaundice"
                if pred < 0.2:
                    severity = "Severe"
                    recommendation = "Immediate hospitalization is recommended."
                elif pred < 0.35:
                    severity = "Moderate"
                    recommendation = "Visit a pediatrician within 24–48 hours."
                else:
                    severity = "Mild"
                    recommendation = "Consult a doctor within the week."
            else:
                prediction = "Normal"
                severity = None
                recommendation = "No action needed. Continue monitoring."

            return render_template(
                'index.html',
                prediction=prediction,
                confidence=confidence,
                image_path=path,
                severity=severity,
                recommendation=recommendation
            )

    return render_template('index.html')

@app.route('/download_report')
def download_report():
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    severity = request.args.get('severity', 'N/A')
    recommendation = request.args.get('recommendation', 'N/A')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c = canvas.Canvas(REPORT_PATH, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 750, "🧾 Jaundice Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Prediction: {prediction}")
    c.drawString(100, 700, f"Confidence: {float(confidence) * 100:.2f}%")
    c.drawString(100, 680, f"Severity: {severity}")
    c.drawString(100, 660, f"Recommendation: {recommendation}")
    c.drawString(100, 640, f"Timestamp: {timestamp}")

    c.save()
    return send_file(REPORT_PATH, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
