import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, filedialog
import serial

model = YOLO("model.pt")
class_labels = model.names
medicine_map = {'Tomato Early blight': 'A','Tomato Late blight': 'A','Tomato Leaf Mold': 'B',}
def upload_image():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Image", filetypes=[("Images", "*.jpg *.png *.jpeg")]
    )
try:
    serialcomm = serial.Serial('COM6', 9600)
except Exception as e:
    print(f"Could not open serial port: {e}")
    exit()

file_path = upload_image()
if not file_path:
    print("No image selected.")
    exit()
image = cv2.imread(file_path)
results = model(image)
if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
    cls_id = int(results[0].boxes.cls[0])
    predicted_class = class_labels[cls_id]
else:
    predicted_class = "Unknown"
    print("No disease detected.")
    serialcomm.close()
    exit()

med_code = medicine_map.get(predicted_class, "N")
treatment = f"Medicine {med_code}" if med_code in ['A', 'B', 'C'] else "No medicine required"
if med_code in ['A', 'B', 'C']:
    serialcomm.write(med_code.encode())
    print(f"Sent to Arduino: {med_code} for {treatment}")
else:
    print("No treatment necessary.")
serialcomm.close()
