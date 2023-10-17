from flask import Flask, render_template, Response, request
import cv2
import serial
import threading
import time
import json
import argparse

app = Flask(__name__)
cap = cv2.VideoCapture(0)
model = YOLO("../models/best.pt")

controlX, controlY = 0.0, 0.0

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def framesGenerator():
    while True:
        global controlX, controlY
        iSee=False
        classes = ['Person', 'Car']
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[0:2]
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #frame_resized = cv2.resize(frame, (360,480), interpolation=cv2.INTER_AREA)
	

        results = model(frame)

        for result in results:
            for r in result.boxes.data.tolist():
                x1,y1,x2,y2,score,class_id=r
                
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                controlX = (
                    2*(x_center-width/2) / width
                )

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0))
                cv2.putText(frame, classes[int(class_id)], (int(x2), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if class_id == 1:
                    iSee = True
                    break

        if iSee==True:
            controlY = 0.5
        else:
            controlY = 0.0
            controlX = 0.0
        
        
        cv2.putText(frame, f"iSee: {iSee}", (width - 180, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"controlX: {round(controlX, 5)}", (width - 500, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1, cv2.LINE_AA)

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
    msg = {
        "speedA": 0,
        "speedB": 0  
    }
    # параметры робота
    speedScale = 0.60  # определяет скорость в процентах (0.60 = 60%) от максимальной абсолютной
    maxAbsSpeed = 100  # максимальное абсолютное отправляемое значение скорости
    sendFreq = 10  # слать 10 пакетов в секунду
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5000, help="Running port")
    parser.add_argument("-i", "--ip", type=str, default="0.0.0.0", help="Ip address")
    parser.add_argument(
        "-s", "--serial", type=str, default="/dev/ttyUSB0", help="Serial port"
    )
    args = parser.parse_args()
    
    serialPort = serial.Serial(args.serial, 9600)   # открываем uart
    def sender():
        global controlX, controlY
        while True:
            speedA = maxAbsSpeed * (controlY + controlX)
            speedB = maxAbsSpeed * (controlY - controlX)
            speedA = max(-maxAbsSpeed, min(speedA, maxAbsSpeed))
            speedB = max(-maxAbsSpeed, min(speedB, maxAbsSpeed))
    
            msg["speedA"], msg["speedB"] = speedScale * speedA, speedScale * speedB
    
            serialPort.write(json.dumps(msg, ensure_ascii=False).encode("utf8"))
            time.sleep(1 / sendFreq)
    
    threading.Thread(target=sender, daemon=True).start()

    app.run(debug=False, host=args.ip, port=5000)  # запускаем flask приложение

