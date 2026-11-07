from flask import Flask, render_template, request
import requests
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
model = YOLO("model.pt")
class_labels = model.names
API= "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key=579b464db66ec23bdd000001cdc3b564546246a772a26393094f5645&offset=0&limit=all&format=json&limit=10000"
medicine = {
    'Tomato Early blight': 'Medicine: Chlorothalonil',
    'Tomato Leaf Mold': 'Medicine: Copper Based Fungicides',
    'Tomato healthy': 'Healthy Plant, Keep Monitoring :)'}

def fetch_live_data():
    try:
        response = requests.get(API)
        response.raise_for_status()
        return response.json().get('records', [])
    except:
        return []

def analyze_image(file):
    img = Image.open(io.BytesIO(file.read()))
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = model(image)
    if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
        cls_id = int(results[0].boxes.cls[0])
        predicted_class = class_labels[cls_id]
    else:
        predicted_class = "No disease detected"
        return predicted_class, "No treatment needed"

    treatment = medicine.get(predicted_class, "Medicine not assigned yet. Monitor the crop.")
    return predicted_class, treatment

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/market", methods=["GET", "POST"])
def market():
    data = fetch_live_data()
    filtered_data = data

    if request.method == "POST":
        commodity = request.form.get("commodity", "").lower()
        market = request.form.get("market", "").lower()
        if commodity:
            filtered_data = [d for d in filtered_data if commodity in d["commodity"].lower()]
        if market:
            filtered_data = [d for d in filtered_data if market in d["market"].lower()]
    return render_template("market.html", data=filtered_data)

@app.route("/advisor", methods=["GET", "POST"])
def advisor():
    advice = None
    crop = None
    if request.method == "POST":
        crop = request.form.get("crop", "").lower()
        if crop == "wheat":
            advice = "Water your wheat crop every 20–25 days and protect it from rust."
        elif crop == "rice":
            advice = "Keep the rice field flooded and use proper fertilizers."
        elif crop == "tomato":
            advice = "Water regularly, provide support to the plant, and monitor for early blight."
        else:
            advice = "Sorry, no advice available for this crop yet."
    return render_template("advisor.html", advice=advice, crop=crop)

@app.route("/legal")
def legal():
    return render_template("legal.html")
@app.route("/ai", methods=["GET", "POST"])
def ai():
    result = None
    if request.method == "POST":
        if "image" not in request.files:
            result = {"disease": "No file uploaded", "treatment": ""}
        else:
            file = request.files["image"]
            if file.filename == "":
                result = {"disease": "No file selected", "treatment": ""}
            else:
                disease, treatment = analyze_image(file)
                result = {"disease": disease, "treatment": treatment}
    return render_template("ai.html", result=result)
if __name__ == "__main__":
    app.run(debug=True)