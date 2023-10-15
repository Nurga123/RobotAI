import cv2
from ultralytics import YOLO
from flask import Flask, render_template, Response, request

app = Flask(__name__)
cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")

def framesGenerator():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #frame_resized = cv2.resize(frame, (360,480), interpolation=cv2.INTER_AREA)
	

        results = model(frame)
        #frame_annotated = results[0].plot()
        
        for result in results:
            boxes=result.boxes.cpu().numpy()
            xyxys=boxes.xyxy

            for xyxy in xyxys:
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0))

        _, buffer = cv2.imencode(".jpg", frame)
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
