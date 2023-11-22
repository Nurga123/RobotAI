from flask import Flask, render_template, Response, request
import cv2
import serial
import threading
from threading import Thread
import time
import json
import argparse
import os
import numpy as np
import sys
import importlib.util


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


MODEL_NAME = "models"
GRAPH_NAME = "detect1.tflite"
LABELMAP_NAME = "labelmap1.txt"
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


app = Flask(__name__)
camera = cv2.VideoCapture(0)  # веб камера

controlX, controlY = 0, 0  # глобальные переменные положения джойстика с web-страницы


def getFramesGenerator():
    """Генератор фреймов для вывода в веб-страницу, тут же можно поиграть с openCV"""
    while True:  # ограничение fps (если видео тупит, можно убрать)
        t1 = cv2.getTickCount()

        global controlX, controlY
        iSee = False

        frame1 = videostream.read()

        frame = frame1.copy()
        # frame = cv2.resize(frame, (480, 320))
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

                if object_name == "Human hand" or object_name == "Mobile phone":
                    iSee = True
                    break
                else:
                    iSee = False

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


@app.route("/video_feed")
def video_feed():
    """Генерируем и отправляем изображения с камеры"""
    return Response(
        getFramesGenerator(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    """Крутим html страницу"""
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
    # пакет, посылаемый на робота
    msg = {
        "speedA": 0,  # в пакете посылается скорость на левый и правый борт тележки
        "speedB": 0,  #
    }

    # параметры робота
    speedScale = (
        1  # определяет скорость в процентах (0.50 = 50%) от максимальной абсолютной
    )
    maxAbsSpeed = 100  # максимальное абсолютное отправляемое значение скорости
    sendFreq = 10  # слать 10 пакетов в секунду

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5000, help="Running port")
    parser.add_argument("-i", "--ip", type=str, default="127.0.0.1", help="Ip address")
    parser.add_argument(
        "-s", "--serial", type=str, default="/dev/ttyUSB0", help="Serial port"
    )
    args = parser.parse_args()

    serialPort = serial.Serial(args.serial, 9600)  # открываем uart

    def sender():
        """функция цикличной отправки пакетов по uart"""
        global controlX, controlY
        while True:
            speedA = maxAbsSpeed * (controlY + controlX)  # преобразуем скорость робота,
            speedB = maxAbsSpeed * (
                controlY - controlX
            )  # в зависимости от положения джойстика

            speedA = max(
                -maxAbsSpeed, min(speedA, maxAbsSpeed)
            )  # функция аналогичная constrain в arduino
            speedB = max(
                -maxAbsSpeed, min(speedB, maxAbsSpeed)
            )  # функция аналогичная constrain в arduino

            msg["speedA"], msg["speedB"] = (
                speedScale * speedA,
                speedScale * speedB,
            )  # урезаем скорость и упаковываем

            serialPort.write(
                json.dumps(msg, ensure_ascii=False).encode("utf8")
            )  # отправляем пакет в виде json файла
            time.sleep(1 / sendFreq)

    threading.Thread(
        target=sender, daemon=True
    ).start()  # запускаем тред отправки пакетов по uart с демоном

    app.run(debug=False, host="0.0.0.0", port=5000)  # запускаем flask приложение
