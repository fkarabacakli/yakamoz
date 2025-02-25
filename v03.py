import numpy as np 
import cv2 as cv
import time


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
    cap = cv.VideoCapture("comb.MOV")

    rgb_color1 = (219, 66, 70)  # red
    rgb_color2 = (16, 75, 148)  # blue

    lower_blue1, upper_blue1 = get_hsv_bounds(rgb_color1)
    lower_blue2, upper_blue2 = get_hsv_bounds(rgb_color2)
    
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        org = frame.copy()
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        mask1 = cv.inRange(hsv, lower_blue1, upper_blue1)
        mask2 = cv.inRange(hsv, lower_blue2, upper_blue2)

        mask = cv.bitwise_or(mask1, mask2)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv.contourArea(contour)
            if area > 1000:
                corner = contour_corner(contour)
                cv.drawContours(frame, [contour], -1, (0, 255, 0), thickness=cv.FILLED)
                x, y, w, h = cv.boundingRect(contour)
                cv.putText(frame, f"Corners: {corner}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # FPS ekleme
        cv.putText(frame, f"FPS: {int(fps)}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Frame", frame)
        cv.imshow("Org", org)
        cv.imshow("Mask", mask)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
