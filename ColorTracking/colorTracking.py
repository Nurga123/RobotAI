from flask import Flask, render_template, Response, request
import cv2
import serial
import threading
import time
import json
import argparse

app = Flask(__name__)
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

controlX, controlY = 0.0, 0.0


def getFramesGenerator():
    global controlX, controlY
    while True:
        iSee = False
        success, frame = camera.read()
        if success:
            frame = cv2.resize(frame, (360, 240), interpolation=cv2.INTER_AREA)
            height, width = frame.shape[0:2]

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
            binary = cv2.inRange(
                hsv, (18, 60, 100), (32, 255, 255)
            )  

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if len(contours) != 0:
                maxc = max(contours, key=cv2.contourArea)
                moments = cv2.moments(maxc) 

                if (
                    moments["m00"] > 20
                ):  
                    cx = int(
                        moments["m10"] / moments["m00"]
                    )  
                    cy = int(
                        moments["m01"] / moments["m00"]
                    )  

                    iSee = True  

                    controlX = (
                        2 * (cx - width / 2) / width
                    )  
                        
                    if abs(controlX) < 0.2:
                        controlX = 0
                    cv2.drawContours(frame, maxc, -1, (0, 255, 0), 1) 
                    cv2.line(
                        frame, (cx, 0), (cx, height), (0, 255, 0), 1
                    ) 
                    cv2.line(frame, (0, cy), (width, cy), (0, 255, 0), 1) 

            if iSee: 
                controlY = 0.5  
            else:
                controlY = 0.0  
                controlX = 0.0  

            miniBin = cv2.resize(
                binary,
                (
                    int(binary.shape[1] / 4),
                    int(binary.shape[0] / 4),
                ),  
                interpolation=cv2.INTER_AREA,
            )  
            miniBin = cv2.cvtColor(miniBin, cv2.COLOR_GRAY2BGR) 
            frame[
                -2 - miniBin.shape[0] : -2, 2 : 2 + miniBin.shape[1]
            ] = miniBin 

            cv2.putText(
                frame,
                "iSee: {};".format(iSee),
                (width - 120, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )  
            cv2.putText(
                frame,
                "controlX: {:.2f}".format(controlX),
                (width - 70, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )  

            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )


@app.route("/video_feed")
def video_feed():
    return Response(
        getFramesGenerator(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
     # Arduino
    msg = {
        "speedA": 0,  
        "speedB": 0  
    }
    
    speedScale = 0.60 
    maxAbsSpeed = 100  
    sendFreq = 10  
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5000, help="Running port")
    parser.add_argument("-i", "--ip", type=str, default="0.0.0.0", help="Ip address")
    parser.add_argument(
        "-s", "--serial", type=str, default="/dev/ttyUSB0", help="Serial port"
    )
    args = parser.parse_args()
    
    serialPort = serial.Serial(args.serial, 9600)  
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

    app.run(debug=False, host=args.ip, port=5000)  
