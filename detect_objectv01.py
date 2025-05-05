import numpy as np 
import cv2 as cv
from ultralytics import YOLO

kernel = np.array([
       [0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0]], dtype="uint8")

def get_hsv_bounds(rgb_color, hue_range=10, sat_min=100, val_min=100):
    bgr_color = np.uint8([[rgb_color[::-1]]])
    hsv_color = cv.cvtColor(bgr_color, cv.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_color

    lower_bound = np.array([max(h - hue_range, 0), sat_min, val_min])
    upper_bound = np.array([min(h + hue_range, 179), 255, 255])

    return lower_bound, upper_bound


def main():
    img = cv.imread("dene01.png")
    #img = cv.imread("fred.jpg")
    img = cv.resize(img, (1280,720))

    rgb_color1 = (251,103,146)  # red
    rgb_color2 = (0,194,247)  # blue

    lower_blue1, upper_blue1 = get_hsv_bounds(rgb_color1)
    lower_blue2, upper_blue2 = get_hsv_bounds(rgb_color2)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask_red = cv.inRange(hsv, lower_blue1, upper_blue1)
    mask_blue = cv.inRange(hsv, lower_blue2, upper_blue2)

    if np.count_nonzero(mask_red) > 0 and np.count_nonzero(mask_blue) > 0:
        mask = cv.bitwise_not(mask_red, mask_blue)
    elif np.count_nonzero(mask_red) > 0:
        mask = cv.bitwise_not(mask_red)
    elif np.count_nonzero(mask_blue) > 0:
        mask = cv.bitwise_not(mask_blue)
    else:
        return
    
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel,iterations=2)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel,iterations=2)

    contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    """model = YOLO("teknofest.pt")
    results = model(img)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    for bbox in bboxes[:1]:
        (x, y, x2, y2) = bbox
        cv.rectangle(img, (x, y), (x2, y2), (0, 0, 225), 2)
        cv.putText(img, "Landing zone", (x, y - 5), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
"""

    max_area = 0
    frame_idx = -1
    for i, cnt in enumerate(contours):
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            frame_idx = i

    for i, cnt in enumerate(contours):
        if i == frame_idx:
            continue

        parent = hierarchy[0][i][3]
        if parent == frame_idx or parent == -1:
            color = (255, 0, 0)  
        else:
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  
                cy = int(M["m01"] / M["m00"])
                cv.circle(img, (cx, cy), 4, (0, 0, 0), -1)
            color = (0, 255, 0)  
        cv.drawContours(img, [cnt], -1, color, thickness=2)

    cv.imshow("Closing", closing)
    cv.imshow("img", img)


    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
