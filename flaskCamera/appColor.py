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
    """Генератор фреймов для вывода в веб-страницу, тут же можно поиграть с openCV"""
    global controlX, controlY
    while True:
        iSee = False

        success, frame = camera.read()
        if success:
            frame = cv2.resize(frame, (360, 240), interpolation=cv2.INTER_AREA)
            height, width = frame.shape[0:2]

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # переводим кадр из RGB в HSV
            binary = cv2.inRange(
                hsv, (18, 60, 100), (32, 255, 255)
            )  # пороговая обработка кадра (выделяем все желтое)

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if len(contours) != 0:
                maxc = max(contours, key=cv2.contourArea)
                moments = cv2.moments(maxc)  # получаем моменты этого контура

                if (
                    moments["m00"] > 20
                ):  # контуры с площадью меньше 20 пикселей не будут учитываться
                    cx = int(
                        moments["m10"] / moments["m00"]
                    )  # находим координаты центра контура по x
                    cy = int(
                        moments["m01"] / moments["m00"]
                    )  # находим координаты центра контура по y

                    iSee = True  # флаг

                    controlX = (
                        2 * (cx - width / 2) / width
                    )  # нормализация к диапазону [1:1]

                    cv2.drawContours(frame, maxc, -1, (0, 255, 0), 1)  # рисуем контур
                    cv2.line(
                        frame, (cx, 0), (cx, height), (0, 255, 0), 1
                    )  # рисуем линию линию по x
                    cv2.line(frame, (0, cy), (width, cy), (0, 255, 0), 1)  # линия по y

            if iSee:  # если был найден объект
                controlY = 0.5  # начинаем ехать вперед с 50% мощностью
            else:
                controlY = 0.0  # останавливаемся
                controlX = 0.0  # сбрасываем меру поворота

            miniBin = cv2.resize(
                binary,
                (
                    int(binary.shape[1] / 4),
                    int(binary.shape[0] / 4),
                ),  # накладываем поверх
                interpolation=cv2.INTER_AREA,
            )  # кадра маленькую
            miniBin = cv2.cvtColor(miniBin, cv2.COLOR_GRAY2BGR)  # битовую маску
            frame[
                -2 - miniBin.shape[0] : -2, 2 : 2 + miniBin.shape[1]
            ] = miniBin  # для наглядности

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
        "speedA": 0,  # в пакете посылается скорость на левый и правый борт тележки
        "speedB": 0  #
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
        """ функция цикличной отправки пакетов по uart """
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
