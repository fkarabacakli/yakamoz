import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("/Users/halilfurkankarabacakli/Desktop/Videos/Video01.mp4")
model = YOLO("best-2.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    for bbox in bboxes:
        print(bbox)
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "LandingZone", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):                            
        break
    
cap.release()
cv2.destroyAllWindows()