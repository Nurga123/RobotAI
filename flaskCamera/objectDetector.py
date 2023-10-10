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
model=YOLO("RobotAI/models/best.pt")
camera = cv2.VideoCapture(0)  # веб камера

controlX, controlY = 0.0, 0.0  # глобальные переменные вектора движения робота. Диапазоны: [-1, 1]

def getFramesGenerator():
    """ Генератор фреймов для вывода в веб-страницу, тут же можно поиграть с openCV"""
    global controlX, controlY
    while True:
        iSee = False  # флаг: был ли найден контур

        success, frame = camera.read()  # Получаем фрейм с камеры

        if success:
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)  # уменьшаем разрешение кадров (если
            # видео тупит, можно уменьшить еще больше)
            height, width = frame.shape[0:2]  # получаем разрешение кадра
            results = model(frame)
            annotated_frame = results[0].plot()

            
            if iSee:  # если был найден объект
                controlY = 0.5  # начинаем ехать вперед с 50% мощностью
            else:
                controlY = 0.0  # останавливаемся
                controlX = 0.0  # сбрасываем меру поворота

            # miniBin = cv2.resize(binary, (int(binary.shape[1] / 4), int(binary.shape[0] / 4)),  # накладываем поверх
            #                      interpolation=cv2.INTER_AREA)  # кадра маленькую
            # miniBin = cv2.cvtColor(miniBin, cv2.COLOR_GRAY2BGR)  # битовую маску
            # frame[-2 - miniBin.shape[0]:-2, 2:2 + miniBin.shape[1]] = miniBin  # для наглядности

            cv2.putText(frame, 'iSee: {};'.format(iSee), (width - 120, height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)  # добавляем поверх кадра текст
            cv2.putText(frame, 'controlX: {:.2f}'.format(controlX), (width - 70, height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)  # добавляем поверх кадра текст

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """ Генерируем и отправляем изображения с камеры"""
    return Response(getFramesGenerator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """ Крутим html страницу """
    return render_template('index.html')


if __name__ == '__main__':
    # # Arduino
    # msg = {
    #     "speedA": 0,  # в пакете посылается скорость на левый и правый борт тележки
    #     "speedB": 0  #
    # }
    #
    # # параметры робота
    # speedScale = 0.60  # определяет скорость в процентах (0.60 = 60%) от максимальной абсолютной
    # maxAbsSpeed = 100  # максимальное абсолютное отправляемое значение скорости
    # sendFreq = 10  # слать 10 пакетов в секунду
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=5000, help="Running port")
    parser.add_argument("-i", "--ip", type=str, default='0.0.0.0', help="Ip address")
    parser.add_argument('-s', '--serial', type=str, default='/dev/ttyUSB0', help="Serial port")
    args = parser.parse_args()
    #
    # serialPort = serial.Serial(args.serial, 9600)   # открываем uart
    #
    # def sender():
    #     """ функция цикличной отправки пакетов по uart """
    #     global controlX, controlY
    #     while True:
    #         speedA = maxAbsSpeed * (controlY + controlX)    # преобразуем скорость робота,
    #         speedB = maxAbsSpeed * (controlY - controlX)    # в зависимости от положения джойстика
    #
    #         speedA = max(-maxAbsSpeed, min(speedA, maxAbsSpeed))    # функция аналогичная constrain в arduino
    #         speedB = max(-maxAbsSpeed, min(speedB, maxAbsSpeed))    # функция аналогичная constrain в arduino
    #
    #         msg["speedA"], msg["speedB"] = speedScale * speedA, speedScale * speedB     # урезаем скорость и упаковываем
    #
    #         serialPort.write(json.dumps(msg, ensure_ascii=False).encode("utf8"))  # отправляем пакет в виде json файла
    #         time.sleep(1 / sendFreq)
    #
    # threading.Thread(target=sender, daemon=True).start()    # запускаем тред отправки пакетов по uart с демоном

    app.run(debug=False, host=args.ip, port=5000)  # запускаем flask приложение

