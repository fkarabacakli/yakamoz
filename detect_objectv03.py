import numpy as np
import cv2 as cv
import time

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
    cuda_available = cv.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA kullanılabilir: {cuda_available}")

    cap = cv.VideoCapture("Videos/Video01.mp4")

    rgb_color1 = (251, 103, 146)  # pembe-kırmızı
    rgb_color2 = (0, 194, 247)    # açık mavi

    lower1, upper1 = get_hsv_bounds(rgb_color1)
    lower2, upper2 = get_hsv_bounds(rgb_color2)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cuda_available:
            gpu_frame = cv.cuda_GpuMat()
            gpu_frame.upload(frame)

            # BGR -> HSV
            gpu_hsv = cv.cuda.cvtColor(gpu_frame, cv.COLOR_BGR2HSV)

            # Maskeleme
            mask1 = cv.cuda.inRange(gpu_hsv, lower1, upper1)
            mask2 = cv.cuda.inRange(gpu_hsv, lower2, upper2)
            mask = cv.cuda.bitwise_or(mask1, mask2)

            final_mask = mask.download()
        else:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask1 = cv.inRange(hsv, lower1, upper1)
            mask2 = cv.inRange(hsv, lower2, upper2)
            final_mask = cv.bitwise_or(mask1, mask2)

        result = cv.bitwise_and(frame, frame, mask=final_mask)

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv.putText(result, f"FPS: {fps:.2f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Maske", final_mask)
        cv.imshow("Maske Uygulanmış", result)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
