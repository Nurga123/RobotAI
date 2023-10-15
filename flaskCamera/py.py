import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model=YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame_resized = cv2.resize(frame, (60,90), interpolation=cv2.INTER_AREA)
    results = model(frame)
    #frame_annotated = results[0].plot()

    for result in results:
        boxes=result.boxes.cpu().numpy()
        xyxys=boxes.xyxy

        for xyxy in xyxys:
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0))
    cv2.imshow('Result', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
