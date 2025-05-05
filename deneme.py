import cv2 as cv
import numpy as np

def get_hsv_bounds(rgb_color, hue_range=10, sat_min=100, val_min=50):
    bgr_color = np.uint8([[rgb_color[::-1]]])
    hsv_color = cv.cvtColor(bgr_color, cv.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_color
    lower_bound = np.array([max(h - hue_range, 0), sat_min, val_min])
    upper_bound = np.array([min(h + hue_range, 179), 255, 255])
    return lower_bound, upper_bound

def get_color_name(hsv_val):
    h, s, v = hsv_val
    if (h < 10 or h > 160) and s > 100:
        return "red"
    elif 90 < h < 130:
        return "blue"
    return "other"

def contour_corner(contour):
    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    return len(approx)

# 📷 GÖRSELİ YÜKLE
image_path = "dene01.png"  # buraya kendi görsel yolunu koy
img = cv.imread(image_path)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 🎯 Renk maskeleri
lower_red, upper_red = get_hsv_bounds((251, 103, 146))
lower_blue, upper_blue = get_hsv_bounds((0, 194, 247))

mask_red = cv.inRange(hsv, lower_red, upper_red)
mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
mask = cv.bitwise_or(mask_red, mask_blue)

# 🔍 Kontur ve hiyerarşi
contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

objects = []
for i, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if area < 100:
        continue

    temp_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv.drawContours(temp_mask, [contour], -1, 255, -1)
    mean_color = cv.mean(hsv, mask=temp_mask)[:3]
    color = get_color_name(mean_color)
    corners = contour_corner(contour)

    objects.append({
        "index": i,
        "contour": contour,
        "color": color,
        "corners": corners,
        "parent": hierarchy[0][i][3]
    })

# 🎯 KURAL KONTROLÜ
for obj in objects:
    if obj["parent"] == -1:
        continue

    parent = next((o for o in objects if o["index"] == obj["parent"]), None)
    if not parent:
        continue

    # Kırmızı üçgen içinde mavi
    if parent["color"] == "red" and parent["corners"] == 3:
        if obj["color"] == "blue":
            print("✅ Kırmızı üçgen içinde mavi bulundu!")
            cv.drawContours(img, [parent["contour"]], -1, (0, 255, 0), 3)
            cv.drawContours(img, [obj["contour"]], -1, (255, 0, 0), 3)

    # Mavi altıgen içinde kırmızı
    if parent["color"] == "blue" and parent["corners"] == 6:
        if obj["color"] == "red":
            print("✅ Mavi altıgen içinde kırmızı bulundu!")
            cv.drawContours(img, [parent["contour"]], -1, (0, 255, 255), 3)
            cv.drawContours(img, [obj["contour"]], -1, (0, 0, 255), 3)

# 📌 GÖSTER
cv.imshow("Tespit", img)
cv.waitKey(0)
cv.destroyAllWindows()
