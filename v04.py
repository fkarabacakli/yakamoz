import numpy as np 
import cv2 as cv
import time

kernel = np.array([
       [0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0]], dtype="uint8")

def get_hsv_bounds(rgb_color, hue_range=10, sat_min=50, val_min=50):
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
    cap = cv.VideoCapture("/Users/halilfurkankarabacakli/Desktop/Videos/Video01.MP4")

    rgb_color1 = (251,103,146)  # red
    rgb_color2 = (0,194,247)  # blue

    lower_blue1, upper_blue1 = get_hsv_bounds(rgb_color1)
    lower_blue2, upper_blue2 = get_hsv_bounds(rgb_color2)
    
    prev_time = time.time()
    valid_corner = [3,6]
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
        opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv.contourArea(contour)
            if area > 100:
                corner = contour_corner(contour)
                if corner not in valid_corner:
                    continue
                cv.drawContours(frame, [contour], -1, (0, 255, 0), thickness=cv.FILLED)
                x, y, w, h = cv.boundingRect(contour)
                cv.putText(frame, f"Corners: {corner}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                M = cv.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])  
                    cy = int(M["m01"] / M["m00"])
                    cv.circle(frame, (cx, cy), 5, (0, 0, 0), -1)

        # FPS ekleme
        cv.putText(frame, f"FPS: {int(fps)}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        bitwise = cv.bitwise_and(org, org, mask=mask)

        cv.imshow("Frame", frame)
        cv.imshow("Org", org)
        cv.imshow("Mask", mask)
        cv.imshow("Bitwise", bitwise)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
