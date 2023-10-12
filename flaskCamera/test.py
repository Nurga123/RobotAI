import cv2
from ultralytics import YOLO
from flask import Flask, render_template, Response, request

app = Flask(__name__)
cap = cv2.VideoCapture(0)
model = YOLO("../models/best.pt")

def framesGenerator():
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        results = model(frame)
        frame_annotated = results[0].plot()
        
        _, buffer = cv2.imencode(".jpg", frame_annotated)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        
@app.route("/video_feed")
def video_feed():
    return Response(
        framesGenerator(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)