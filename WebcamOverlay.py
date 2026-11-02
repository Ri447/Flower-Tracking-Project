import cv2
import time
import random
import glob
from FingerTracking import handDetector
import numpy as np

flower_size = 100
half_size = flower_size // 2
flower_images = []
for path in glob.glob("Flowers/*.png"):
    overlay = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if overlay is not None and overlay.shape[2] == 4:
        overlay = cv2.resize(overlay, (flower_size, flower_size))
        flower_images.append(overlay)
if len(flower_images) == 0:
    print("No valid transparent flower images found!")
    exit()
# overlay=cv2.imread("Flowers/2.png",cv2.IMREAD_UNCHANGED)
#
# if overlay is None:
#     print("Flower image not found!")
#     exit()
#
# if overlay.shape[2] != 4:
#     print("Flower must have an alpha channel!")
#     exit()
#
# overlay=cv2.resize(overlay,(150,150))
# alpha = overlay[:, :, 3] / 255.0
# flower = overlay[:, :, :3]

cap = cv2.VideoCapture(1)
detector = handDetector()
flowers = []
previousPoint = None
clear_gesture_active = False
previous_time = time.time()

while True:
    current_time = time.time()
    dt = min(current_time - previous_time, 0.05)
    previous_time = current_time
    success, frame = cap.read()
    if not success:
        print("Cannot access webcam!")
        break
    frame = cv2.flip(frame, 1)
    frame=detector.findHand(frame)
    lmList=detector.findPosition(frame,draw=False)

    #HAND GESTURE
    if len(lmList) != 0:
        x, y = lmList[8][1], lmList[8][2]
        index_up = lmList[8][2] < lmList[6][2]
        middle_up = lmList[12][2] < lmList[10][2]
        ring_up = lmList[16][2] < lmList[14][2]
        pinky_up = lmList[20][2] < lmList[18][2]
        fingers_up = [index_up, middle_up, ring_up, pinky_up]

        x -= half_size
        y -= half_size
        #DRAW FLOWERS
        if index_up and not middle_up and not ring_up and not pinky_up:
            if previousPoint is None or np.linalg.norm(np.array([x, y]) - np.array(previousPoint)) > 50:
                if 0 <= x <= frame.shape[1] - flower_size and 0 <= y <= frame.shape[0] - flower_size:
                    flowers.append({"x": x,"y": y,"start": time.time(),"phase": np.random.uniform(0, 2 * np.pi),"image":random.choice(flower_images),"state": "planted","vy": 0,"rotation": 0,"rotation_speed": np.random.uniform(-3, 3),"glow_start": None})
                    previousPoint = (x, y)
        clear_gesture = all(fingers_up)
        if clear_gesture and not clear_gesture_active:
            for item in flowers:
                item["state"] = "glowing"
                item["vy"] = 0
                item["glow_start"] = time.time()
            previousPoint = None
        clear_gesture_active = clear_gesture
    else:
        previousPoint = None
        clear_gesture_active = False

    output=frame.copy()
    remaining_flowers=[]
    for item in flowers:
        age=time.time()-item["start"]
        scale=min(age/0.4,1.0)
        angle=np.sin(age*2+item["phase"])*8
        if item["state"] == "glowing":
            if time.time()-item["glow_start"] > 0.15:
                item["state"] = "falling"
        if item["state"] == "falling":
            item["y"] += item["vy"] * dt * 60
            item["vy"] += 0.25 * dt * 60
            item["rotation"] += item["rotation_speed"] * dt * 60
            angle = item["rotation"]
            float_y = 0
        else:
            float_y = int(np.sin(age * 2.2 + item["phase"]) * 4)
        matrix=cv2.getRotationMatrix2D((half_size,half_size),angle,scale)
        animated = cv2.warpAffine(item["image"],matrix,(flower_size,flower_size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
        x, y = int(item["x"]), int(item["y"] + float_y)
        if y > output.shape[0]:
            continue
        h,w,_ = output.shape
        x1,y1=max(0,x),max(0,y)
        x2,y2=min(w,x+flower_size),min(h,y+flower_size)
        if x1<x2 and y1<y2:
            flower_crop = animated[y1 - y:y2 - y, x1 - x:x2 - x]
            if item["glow_start"] is not None:
                glow_age = time.time() - item["glow_start"]
                glow_strength = max(0, 1 - glow_age / 0.25)
                if glow_strength > 0:
                    glow_mask = cv2.GaussianBlur(animated[:, :, 3], (0, 0), 10)
                    glow_crop = glow_mask[y1 - y:y2 - y, x1 - x:x2 - x]
                    glow_alpha = (glow_crop[:, :, None] / 255.0) * glow_strength * 0.5
                    glow_color = np.zeros_like(output[y1:y2, x1:x2])
                    glow_color[:] = (180, 220, 255)
                    background = output[y1:y2, x1:x2]
                    output[y1:y2, x1:x2] = (glow_alpha * glow_color +(1 - glow_alpha) * background).astype(np.uint8)
            alpha = flower_crop[:, :, 3:4] / 255.0
            flower_rgb = flower_crop[:, :, :3]
            background = output[y1:y2, x1:x2]
            output[y1:y2, x1:x2] = (alpha * flower_rgb + (1 - alpha) * background).astype(np.uint8)
        remaining_flowers.append(item)
    flowers = remaining_flowers

    cv2.imshow("Flower Painting", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()