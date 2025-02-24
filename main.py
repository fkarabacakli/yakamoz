import cv2 as cv
import numpy as np

def nothing():
    pass

def main():
    #cap = cv.VideoCapture("Hsv_Object_Detection/hsv.mp4")
    cap = cv.VideoCapture(0)
    
    cv.namedWindow("Trackbar")

    cv.createTrackbar("LH","Trackbar",0,179, nothing)
    cv.createTrackbar("LS","Trackbar",0,255, nothing)
    cv.createTrackbar("LV","Trackbar",0,255, nothing)
    cv.createTrackbar("UH","Trackbar",0,179, nothing)
    cv.createTrackbar("US","Trackbar",0,255, nothing)
    cv.createTrackbar("UV","Trackbar",0,255, nothing)

    while True:
        _, frame = cap.read()
        frame = cv.resize(frame,(720,480))
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        
        lh = cv.getTrackbarPos("LH","Trackbar")
        ls = cv.getTrackbarPos("LS","Trackbar")
        lv = cv.getTrackbarPos("LV","Trackbar")
        uh = cv.getTrackbarPos("UH","Trackbar")
        us = cv.getTrackbarPos("US","Trackbar")
        uv = cv.getTrackbarPos("UV","Trackbar")

        lower_blue = np.array([lh,ls,lv])
        upper_blue = np.array([uh,us,uv])

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