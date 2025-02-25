import cv2 as cv
import numpy as np
from ultralytics import YOLO

def nothing():
    pass

def main():
    cap = cv.VideoCapture("video.mp4")
    model = YOLO("best.pt")
    
    cv.namedWindow("Trackbar")

    cv.createTrackbar("LH","Trackbar",0,179, nothing)
    cv.createTrackbar("LS","Trackbar",0,255, nothing)
    cv.createTrackbar("LV","Trackbar",0,255, nothing)
    cv.createTrackbar("UH","Trackbar",0,179, nothing)
    cv.createTrackbar("US","Trackbar",0,255, nothing)
    cv.createTrackbar("UV","Trackbar",0,255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        
        lh = cv.getTrackbarPos("LH","Trackbar")
        ls = cv.getTrackbarPos("LS","Trackbar")
        lv = cv.getTrackbarPos("LV","Trackbar")
        uh = cv.getTrackbarPos("UH","Trackbar")
        us = cv.getTrackbarPos("US","Trackbar")
        uv = cv.getTrackbarPos("UV","Trackbar")

        lower_blue = np.array([lh,50,lv])
        upper_blue = np.array([uh,180,180])

        mask = cv.inRange(hsv,lower_blue,upper_blue)

        bitwise = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("Frame", frame)
        cv.imshow("Mask", mask)
        cv.imshow("Bitwise", bitwise)

        if cv.waitKey(1) & 0xFF ==  ord('q'):
            break

    cv.destroyAllWindows()
if __name__ == "__main__":
    main()