import cv2
import time
import random
import glob
from FingerTracking import handDetector
import numpy as np

flower_size = 100
# half_size = flower_size // 2
flower_images = []
flower_bag = []
last_flower = None
def get_random_flower():
    global flower_bag, last_flower
    if len(flower_bag) == 0:
        flower_bag = flower_images.copy()
        random.shuffle(flower_bag)
        if len(flower_bag) > 1 and flower_bag[-1] is last_flower:
            flower_bag[0], flower_bag[-1] = flower_bag[-1], flower_bag[0]
    flower = flower_bag.pop()
    last_flower = flower

    return flower
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

cap = cv2.VideoCapture(0)
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

        # x -= half_size
        # y -= half_size
        #DRAW FLOWERS
        if index_up and not middle_up and not ring_up and not pinky_up:
            if previousPoint is None or np.linalg.norm(np.array([x, y]) - np.array(previousPoint)) > 50:
                new_size = random.randint(65, 105)
                flower_x = lmList[8][1] - new_size // 2
                flower_y = lmList[8][2] - new_size // 2
                if 0 <= x <= frame.shape[1] - new_size and 0 <= y <= frame.shape[0] - new_size:
                    flowers.append({"x": flower_x,"y": flower_y,"start": time.time(),"phase": np.random.uniform(0, 2 * np.pi),"float_speed": random.uniform(1.5, 3.0),"float_amplitude": random.uniform(2, 7),"image":get_random_flower(),"state": "planted","vy":0,"vx": random.uniform(-1.5,1.5),"wind_phase": random.uniform(0, 2 * np.pi),"wind_speed": random.uniform(1.0, 2.5),"wind_strength": random.uniform(0.3, 1.0),"rotation": random.uniform(-25, 25),"rotation_speed": np.random.uniform(-3, 3),"glow_start": None,"size": new_size})
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
            wind = np.sin(age * item["wind_speed"] + item["wind_phase"]) * item["wind_strength"]
            item["x"] += (item["vx"] + wind) * dt * 60
            item["rotation"] += item["rotation_speed"] * dt * 60
            angle = item["rotation"]
            float_y = 0
        else:
            float_y = int(np.sin(age * item["float_speed"] + item["phase"]) * item["float_amplitude"])
        size = item["size"]
        half = size // 2
        matrix = cv2.getRotationMatrix2D((half, half), angle, scale)
        flower_image = cv2.resize(item["image"], (size, size))
        animated = cv2.warpAffine(flower_image, matrix, (size, size), flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        x, y = int(item["x"]), int(item["y"] + float_y)
        fade_start = output.shape[0] * 0.75
        fade_end = output.shape[0] - size
        fade = 1.0
        if y > fade_start:
            fade = max(0.0, (fade_end - y) / (fade_end - fade_start))
        if y >= output.shape[0]:
            continue
        h,w,_ = output.shape
        x1,y1=max(0,x),max(0,y)
        x2,y2=min(w,x+size),min(h,y+size)
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
            alpha = (flower_crop[:, :, 3:4] / 255.0) *fade
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