import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("../models/best.pt")
while True:
    ret, frame = cap.read()

    results = model(frame)
    # for result in results:
    #     boxes = result.boxes.numpy()
    #     labels = result.names.numpy()  # Получение названий классов
        
    #     xyxys = boxes.xyxy
    #     for xyxy in xyxys:
    #         x1, y1, x2, y2 = map(int, xyxy)
    #         class_id = result.get_field("labels")[0]  # Получение ID класса
    #         class_name = labels[class_id]  # Получение названия класса
    #         center_x = (x1 + x2) // 2  # Рассчет координат центра по X
    #         center_y = (y1 + y2) // 2  # Рассчет координат центра по Y

    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(frame, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)  # Рисование красной точки в центре объекта
    frame_annotated = results[0].plot()
    cv2.imshow('Frame', frame_annotated)
    cv2.waitKey(1)