import cv2
from ultralytics import YOLO
import threading
model = YOLO('model.pt')
frame = None
annotated_frame = None
lock = threading.Lock()
running = True

def inference_thread():
    global frame, annotated_frame, running
    while running:
        with lock:
            current_frame = frame.copy() if frame is not None else None
        if current_frame is not None:
            results = model(current_frame)
            out_frame = results[0].plot()
            with lock:
                annotated_frame = out_frame
cap = cv2.VideoCapture(0)
thread = threading.Thread(target=inference_thread)
thread.start()

while True:
    ret, img = cap.read()
    if not ret:
        break
    with lock:
        frame = img.copy()
        display_frame = annotated_frame if annotated_frame is not None else img
    cv2.imshow("Live Disease Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break
cap.release()
thread.join()
cv2.destroyAllWindows()
