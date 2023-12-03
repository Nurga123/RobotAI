import os
import argparse
import cv2
import numpy as np
import sys
import time
import threading
from threading import Thread
import importlib.util
from flask import Flask, render_template, Response, Request
import serial
import json


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


app = Flask(__name__)

object_class = "default"

MODEL_NAME = "models"
GRAPH_NAME = "detect02.12.tflite"
LABELMAP_NAME = "labelmap02.12.txt"
min_conf_threshold = float(0.5)
resW, resH = 640, 480
imW, imH = int(resW), int(resH)

pkg = importlib.util.find_spec("tflite_runtime")

if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter


CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, "r") as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]

floating_model = input_details[0]["dtype"] == np.float32

input_mean = 127.5
input_std = 127.5

outname = output_details[0]["name"]

if "StatefulPartitionedCall" in outname:  # TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = cv2.getTickFrequency()

controlX = 0.0
controlY = 0.0

videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)


@app.route("/detect/<new_object_class>")
def detect(new_object_class):
    global object_class
    object_class = new_object_class
    print(object_class)


def getFramesGenerator():
    while True:
        t1 = cv2.getTickCount()

        global controlX, controlY
        iSee = False

        frame1 = videostream.read()

        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (320, 320))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]["index"])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]["index"])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]["index"])[0]

        for i in range(len(scores)):
            if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                x_center = (xmin + xmax) / 2
                y_center = (ymax + ymin) / 2

                cv2.circle(frame, (int(x_center), int(y_center)), 7, (0, 0, 255), -1)

                controlX = 2 * (x_center - imW / 2) / imW
                controlX = round(controlX, 3)

                if abs(controlX) < 0.2:
                    controlX = 0

                object_name = labels[int(classes[i])]

                label = "%s: %d%%" % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(
                    frame,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

                if object_name == object_class:
                    iSee = True
                    break
                else:
                    iSee = False

        if iSee == True:
            if controlX == 0:
                controlY = 0.7
            else:
                controlY = 0
        else:
            controlX = 0.0
            controlY = 0.0

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        cv2.putText(
            frame,
            "FPS: {0:.2f}".format(frame_rate_calc),
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "iSee: {};".format(iSee),
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "controlX: {:.2f}".format(controlX),
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        # time.sleep(1)


@app.route("/video_feed")
def video_feed():
    """Генерируем и отправляем изображения с камеры"""
    return Response(
        getFramesGenerator(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/control")
def control():
    """Пришел запрос на управления роботом"""
    global controlX, controlY
    controlX, controlY = (
        float(request.args.get("x")) / 100.0,
        float(request.args.get("y")) / 100.0,
    )
    return "", 200, {"Content-Type": "text/plain"}


if __name__ == "__main__":
    msg = {"speedA": 0, "speedB": 0}
    speedScale = 0.5
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
        """функция цикличной отправки пакетов по uart"""
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

    app.run(debug=False, host="0.0.0.0", port=5000)

cv2.destroyAllWindows()
videostream.stop()
