import numpy as np
import cv2 as cv
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

kernel = np.array([
       [0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0]], dtype="uint8")

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * np.dtype(np.float32).itemsize
        device_mem = cuda.mem_alloc(size)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(device_mem)
        else:
            outputs.append(device_mem)
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    cuda.memcpy_htod_async(inputs[0], inputs[0], stream)
    context.execute_async_v2(bindings, stream.handle, None)
    cuda.memcpy_dtoh_async(outputs[0], outputs[0], stream)
    stream.synchronize()
    return outputs[0]

def get_hsv_bounds(rgb_color, hue_range=10, sat_min=100, val_min=100):
    bgr_color = np.uint8([[rgb_color[::-1]]])
    hsv_color = cv.cvtColor(bgr_color, cv.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_color
    lower_bound = np.array([max(h - hue_range, 0), sat_min, val_min])
    upper_bound = np.array([min(h + hue_range, 179), 255, 255])
    return lower_bound, upper_bound

def contour_corner(contour):
    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    return len(approx)

def main():
    cap = cv.VideoCapture("/Users/halilfurkankarabacakli/Desktop/Videos/Video03.MP4")
    engine_path = "best.engine"
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    rgb_color1 = (251, 103, 146)
    rgb_color2 = (0, 194, 247)
    lower_blue1, upper_blue1 = get_hsv_bounds(rgb_color1)
    lower_blue2, upper_blue2 = get_hsv_bounds(rgb_color2)
    
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, (854, 480))
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
            if cv.contourArea(contour) > 100:
                corner = contour_corner(contour)
                cv.drawContours(frame, [contour], -1, (0, 255, 0), thickness=cv.FILLED)
                x, y, w, h = cv.boundingRect(contour)
                cv.putText(frame, f"Corners: {corner}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                M = cv.moments(contour)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv.circle(frame, (cx, cy), 5, (0, 0, 0), -1)
        
        cv.putText(frame, f"FPS: {int(fps)}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        bitwise = cv.bitwise_and(org, org, mask=mask)
        
        input_data = bitwise.astype(np.float32).flatten()
        inputs[0] = input_data
        outputs[0] = np.empty(engine.get_binding_shape(1), dtype=np.float32)
        detections = do_inference(context, bindings, inputs, outputs, stream)
        
        for i in range(0, len(detections), 6):
            x, y, x2, y2, _, _ = detections[i:i+6].astype(int)
            cv.rectangle(bitwise, (x, y), (x2, y2), (0, 0, 225), 2)
            cv.putText(bitwise, "Landing zone", (x, y - 5), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
        
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
