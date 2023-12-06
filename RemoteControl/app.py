from flask import Flask, render_template, Response, request
import cv2
import serial
import threading
import time
import json
import argparse

app = Flask(__name__)
camera = cv2.VideoCapture(0)  

controlX, controlY = 0, 0  

mode = 2 #REMOTE CONTROL

def getFramesGenerator():
    while True:
        success, frame = camera.read()  
        if success:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)  
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            print("Failed to open camera")

@app.route('/video_feed')
def video_feed():
    return Response(getFramesGenerator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/control')
def control():
    global controlX, controlY
    controlX, controlY = float(request.args.get('x')) / 100.0, float(request.args.get('y')) / 100.0
    return '', 200, {'Content-Type': 'text/plain'}


if __name__ == '__main__':
    msg = {
        "speedA": 0,
        "speedB": 0,
        "mode": 2
    }
    
    speedScale = 1  
    maxAbsSpeed = 100  
    sendFreq = 10  

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=5000, help="Running port")
    parser.add_argument("-i", "--ip", type=str, default='127.0.0.1', help="Ip address")
    parser.add_argument('-s', '--serial', type=str, default='/dev/ttyUSB0', help="Serial port")
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
            msg["mode"] = mode
            serialPort.write(json.dumps(msg, ensure_ascii=False).encode("utf8"))
            time.sleep(1 / sendFreq)

    threading.Thread(target=sender, daemon=True).start()

    app.run(debug=False, host='0.0.0.0', port=5000) 

