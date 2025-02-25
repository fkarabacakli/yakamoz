import numpy as np 
import cv2 as cv


def get_hsv_bounds(rgb_color, hue_range=10, sat_min=100, val_min=50):
    bgr_color = np.uint8([[rgb_color[::-1]]])
    hsv_color = cv.cvtColor(bgr_color, cv.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_color

    lower_bound = np.array([max(h - hue_range, 0), sat_min, val_min])
    upper_bound = np.array([min(h + hue_range, 179), 255, 255])

    return lower_bound, upper_bound


def contour_corner(contour):
    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    corners = len(approx)
    return corners


def main():
    cap = cv.VideoCapture("red01.MOV")

    while True:
        ret, frame = cap.read()
        org = frame.copy()
        if not ret:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        #rgb_color = (252, 139, 50)
        #rgb_color = (215,202,73) /yellow
        rgb_color = (219,66,70) #red
        #rgb_color = (16,75,148) #blue
        
        lower_blue, upper_blue = get_hsv_bounds(rgb_color)  
        mask = cv.inRange(hsv, lower_blue, upper_blue)

        #bitwise = cv.bitwise_and(frame, frame, mask=mask)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 1000:
                corner = contour_corner(contour)
                cv.drawContours(frame, [contour], -1, (0, 255, 0), thickness=cv.FILLED)
                x, y, w, h = cv.boundingRect(contour)
                cv.putText(frame, f"Corners: {corner}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv.imshow("Frame", frame)
        cv.imshow("Org", org)
        cv.imshow("Mask", mask)
        #cv.imshow("Bitwise", bitwise)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
