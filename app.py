from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import io
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from datetime import datetime

app = Flask(__name__)
model = load_model('unet_jaundice_classifier.keras')

UPLOAD_FOLDER = 'static/uploads'
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
            filename = secure_filename(image_file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
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

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # 1. Header Bar & Title
    c.setFillColor(colors.HexColor('#4f46e5'))
    c.rect(50, 735, 512, 8, fill=True, stroke=False)
    
    c.setFillColor(colors.HexColor('#1e293b'))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 755, "JaundiceCare Diagnostics")
    
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor('#64748b'))
    c.drawString(50, 715, "CLINICAL ASSESSMENT REPORT - NEONATAL BILIRUBIN SCREENING")
    
    c.setStrokeColor(colors.HexColor('#cbd5e1'))
    c.setLineWidth(1)
    c.line(50, 705, 562, 705)

    # 2. Patient / Test Information Box
    c.setFillColor(colors.HexColor('#f8fafc'))
    c.setStrokeColor(colors.HexColor('#e2e8f0'))
    c.rect(50, 595, 512, 95, fill=True, stroke=True)
    
    c.setFillColor(colors.HexColor('#1e293b'))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(65, 670, "PATIENT INFORMATION")
    c.drawString(320, 670, "TEST INFORMATION")
    
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor('#334155'))
    c.drawString(65, 650, "Patient Name: Newborn / Infant")
    c.drawString(65, 630, "Patient ID: JNC-2026-0703")
    c.drawString(65, 610, "Age / Gender: Neonatal / Infant")
    
    c.drawString(320, 650, f"Assessment Date: {timestamp}")
    c.drawString(320, 630, "Methodology: CNN Image-Based Classification")
    c.drawString(320, 610, "Status: Screening Completed")

    # 3. Results Box (Color coded: Red for Jaundice, Green for Normal)
    is_jaundice = prediction.lower() == "jaundice"
    accent_color = colors.HexColor('#ef4444') if is_jaundice else colors.HexColor('#10b981')
    bg_accent = colors.HexColor('#fef2f2') if is_jaundice else colors.HexColor('#ecfdf5')
    border_accent = colors.HexColor('#fee2e2') if is_jaundice else colors.HexColor('#d1fae5')
    
    c.setFillColor(bg_accent)
    c.setStrokeColor(border_accent)
    c.rect(50, 465, 512, 110, fill=True, stroke=True)
    
    c.setFillColor(colors.HexColor('#1e293b'))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(65, 555, "SCREENING ASSESSMENT RESULTS")
    
    c.setFont("Helvetica", 11)
    c.drawString(65, 530, "Result / Classification:")
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(accent_color)
    c.drawString(200, 530, prediction.upper())
    
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.HexColor('#334155'))
    c.drawString(65, 505, "Confidence Probability:")
    try:
        conf_val = float(confidence) * 100
        conf_str = f"{conf_val:.2f}%"
    except (ValueError, TypeError):
        conf_str = f"{confidence}"
    c.setFont("Helvetica-Bold", 12)
    c.drawString(200, 505, conf_str)
    
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.HexColor('#334155'))
    c.drawString(65, 480, "Clinical Severity:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(200, 480, str(severity) if (severity and severity != 'None' and severity != 'N/A') else "N/A (Normal)")

    # 4. Pediatric Action & Recommendations Box
    c.setFillColor(colors.HexColor('#f8fafc'))
    c.setStrokeColor(colors.HexColor('#e2e8f0'))
    c.rect(50, 310, 512, 140, fill=True, stroke=True)
    
    c.setFillColor(colors.HexColor('#1e293b'))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(65, 430, "PEDIATRIC ACTION & RECOMMENDATION")
    
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor('#475569'))
    c.drawString(65, 405, "The classification is based on dermal color analysis of the forehead area.")
    c.drawString(65, 385, "Recommendation details:")
    
    c.setFont("Helvetica-Oblique", 11)
    c.setFillColor(colors.HexColor('#0f172a'))
    c.drawString(65, 360, f"\"{recommendation}\"")
    
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor('#64748b'))
    c.drawString(65, 330, "Note: Always confirm findings with a serum total bilirubin (TSB) test.")

    # 5. Footer & Clinical Disclaimer
    c.setStrokeColor(colors.HexColor('#cbd5e1'))
    c.setLineWidth(1)
    c.line(50, 120, 562, 120)
    
    c.setFillColor(colors.HexColor('#64748b'))
    c.setFont("Helvetica-Bold", 8)
    c.drawString(50, 105, "DISCLAIMER & LIMITATION OF LIABILITY")
    
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 90, "This report is generated automatically by a Convolutional Neural Network (CNN) trained on clinical forehead image datasets.")
    c.drawString(50, 78, "It is intended for preliminary screening assistance and does NOT constitute a clinical diagnosis, medical prescription,")
    c.drawString(50, 66, "or medical advice. Decisions regarding neonatal care should be made under professional pediatric supervision.")
    
    c.setFont("Helvetica", 8)
    c.drawString(50, 50, "JaundiceCare System v1.0.0 | Powered by TensorFlow & Keras")

    c.save()
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"jaundice_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    )

if __name__ == '__main__':
    app.run(debug=True)
