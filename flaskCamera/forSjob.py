from flask import Flask, render_template, Response, request
from ultralytics import YOLO
import cv2
import numpy
import serial
import threading
import time
import json
import argparse


app = Flask(__name__)


model = YOLO("RobotAI/models/best.pt")
camera = cv2.VideoCapture(0)  # веб камера

controlX, controlY = (
    0.0,
    0.0,
)  # глобальные переменные вектора движения робота. Диапазоны: [-1, 1]


def getFramesGenerator():
    """Генератор фреймов для вывода в веб-страницу, тут же можно поиграть с openCV"""
    global controlX, controlY
    while True:
        iSee = False  # флаг: был ли найден контур

        success, frame = camera.read()  # Получаем фрейм с камеры

        if success:
            frame = cv2.resize(
                frame, (320, 240), interpolation=cv2.INTER_AREA
            )  # уменьшаем разрешение кадров (если
            # видео тупит, можно уменьшить еще больше)
            height, width = frame.shape[0:2]  # получаем разрешение кадра
            results = model(frame)
            annotated_frame = results[0].plot()

            if iSee:  # если был найден объект
                controlY = 0.5  # начинаем ехать вперед с 50% мощностью
            else:
                controlY = 0.0  # останавливаемся
                controlX = 0.0  # сбрасываем меру поворота

            cv2.putText(
                frame,
                "iSee: {};".format(iSee),
                (width - 120, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )  # добавляем поверх кадра текст
            cv2.putText(
                frame,
                "controlX: {:.2f}".format(controlX),
                (width - 70, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )  # добавляем поверх кадра текст

            _, buffer = cv2.imencode(".jpg", annotated_frame)
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
        "speedA": 0,    #пакетте бірінші және екінші 
        "speedB": 0     #қозғалтқышқа жылдамдық жіберіледі 
          
    }
    
    # робот параметрлері
    speedScale = 0.60  # жылдамдықты максималды абсолютті пайызбен (0.60 = 60%) анықтайды
    maxAbsSpeed = 100  # максималды абсолютті жіберілетін жылдамдық мәні
    sendFreq = 10  # секундына 10 пакет жіберу
    
    parser = argparse.ArgumentParser()                                                #Веб сервер үшін порт пен ip адрессті анықтау
    parser.add_argument("-p", "--port", type=int, default=5000, help="Running port")
    parser.add_argument("-i", "--ip", type=str, default="0.0.0.0", help="Ip address")
    parser.add_argument(
        "-s", "--serial", type=str, default="/dev/ttyUSB0", help="Serial port"
    )
    args = parser.parse_args()
    
    serialPort = serial.Serial(args.serial, 9600)   # uart ашу
    
    def sender():
        """ функция цикличной отправки пакетов по uart """
        global controlX, controlY
        while True:
            speedA = maxAbsSpeed * (controlY + controlX)    
            speedB = maxAbsSpeed * (controlY - controlX)    
    
            speedA = max(-maxAbsSpeed, min(speedA, maxAbsSpeed))    # бірінші қозғалтқышқа жылдамдық беру
            speedB = max(-maxAbsSpeed, min(speedB, maxAbsSpeed))    # екінші қозғалтқышқа жылдамдық беру
    
            msg["speedA"], msg["speedB"] = speedScale * speedA, speedScale * speedB     # жылдамдықты өзгерту және орау
    
            serialPort.write(json.dumps(msg, ensure_ascii=False).encode("utf8"))  # json форматында пакет жіберу
            time.sleep(1 / sendFreq)
    
    threading.Thread(target=sender, daemon=True).start()    # UART арқылы пакеттерді жіберу

    app.run(debug=False, host=args.ip, port=5000)  # flask веб серверін іске қосу
