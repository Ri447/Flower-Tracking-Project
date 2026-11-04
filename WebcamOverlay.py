import cv2,time,random,glob,os,math
from FingerTracking import handDetector
import numpy as np
from Transformations.FaceMeshModule import FaceMeshDetector

flower_size = 100
flower_images = []
fps = 0
flower_bag = []
last_flower = None
crown_image = cv2.imread("Accessories/crown.png",cv2.IMREAD_UNCHANGED)
if crown_image is None or crown_image.ndim != 3 or crown_image.shape[2] != 4:
    print("Crown image not found or does not have transparency!")
    exit()
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

def overlay_transparent(background, overlay, x, y):
    h, w = background.shape[:2]
    overlay_h, overlay_w = overlay.shape[:2]
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + overlay_w, w)
    y2 = min(y + overlay_h, h)
    if x1 >= x2 or y1 >= y2:
        return background
    overlay_crop = overlay[y1-y:y2-y, x1-x:x2-x]
    alpha = overlay_crop[:, :, 3:4] / 255.0
    overlay_rgb = overlay_crop[:, :, :3]
    background_crop = background[y1:y2, x1:x2]
    background[y1:y2, x1:x2] = (alpha * overlay_rgb +(1 - alpha) * background_crop).astype(np.uint8)
    return background

def rotate_overlay(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image,matrix,(new_w, new_h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0, 0, 0, 0))
    return rotated

def handle_mouse(event, x, y, flags, param):
    global crown_mode_active
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= x <= 190 and 20 <= y <= 65:
            crown_mode_active = not crown_mode_active

cap = cv2.VideoCapture(1)
detector = handDetector()
face_detector=FaceMeshDetector(staticMode=False, maxFaces=3,minDetectionCon=0.5,minTrackCon=0.5)
crown_mode_active = False
cv2.namedWindow("Flower Painting")
cv2.setMouseCallback("Flower Painting", handle_mouse)
flowers = []
previousPoint = {"Left":None,"Right":None}
clear_gesture_active = False
current_flower_size = 100
previous_time = time.time()
os.makedirs("Captured Photos", exist_ok=True)

# CROWN SMOOTHING
crown_positions = {}
smoothing_factor = 0.25

while True:
    current_time = time.time()
    actual_dt = current_time - previous_time
    dt = min(actual_dt, 0.05)
    fps = 1 / actual_dt if actual_dt > 0 else 0
    previous_time = current_time
    success, frame = cap.read()
    if not success:
        print("Cannot access webcam!")
        break
    frame = cv2.flip(frame, 1)
    frame=detector.findHand(frame)
    hands=detector.findAllHands(frame)
    if crown_mode_active:
        frame=face_detector.findFaceMesh(frame,draw=False)

    #HAND GESTURE
    detected_hands=set()
    left_clear_gesture = False
    for hand in hands:
        lmList = hand["landmarks"]
        wrist_x = lmList[0][1]
        frame_middle = frame.shape[1] // 2
        if wrist_x < frame_middle:
            handType = "Left"
        else:
            handType = "Right"
        cv2.putText(frame,handType,(lmList[8][1], lmList[8][2] - 20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255),2)
        detected_hands.add(handType)
        x, y = lmList[8][1], lmList[8][2]
        index_up = lmList[8][2] < lmList[6][2]
        middle_up = lmList[12][2] < lmList[10][2]
        ring_up = lmList[16][2] < lmList[14][2]
        pinky_up = lmList[20][2] < lmList[18][2]
        fingers_up = [index_up, middle_up, ring_up, pinky_up]
        #LEFT HAND: PINCH SIZE CONTROL
        if handType == "Left":
            thumb_tip = lmList[4]
            index_tip = lmList[8]
            pinch_distance = np.linalg.norm(np.array([thumb_tip[1], thumb_tip[2]]) - np.array([index_tip[1], index_tip[2]]))
            current_flower_size=int(np.interp(pinch_distance,[55,150],[20,120]))
            cv2.putText(frame, f"Flower Size: {current_flower_size}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
            left_clear_gesture = all(fingers_up)

        #DRAW FLOWERS
        if (handType=="Right" and index_up and not middle_up and not ring_up and not pinky_up):
            if previousPoint[handType] is None or np.linalg.norm(np.array([x, y]) - np.array(previousPoint[handType])) > 50:
                new_size = current_flower_size
                flower_x = x - new_size // 2
                flower_y = y - new_size // 2
                if 0 <= flower_x <= frame.shape[1] - new_size and 0 <= flower_y <= frame.shape[0] - new_size:
                    flowers.append({"x": flower_x,"y": flower_y,"start": time.time(),"phase": np.random.uniform(0, 2 * np.pi),"float_speed": random.uniform(1.5, 3.0),"float_amplitude": random.uniform(2, 7),"image":get_random_flower(),"state": "planted","vy":0,"vx": random.uniform(-1.5,1.5),"wind_phase": random.uniform(0, 2 * np.pi),"wind_speed": random.uniform(1.0, 2.5),"wind_strength": random.uniform(0.3, 1.0),"rotation": random.uniform(-25, 25),"rotation_speed": np.random.uniform(-3, 3),"glow_start": None,"size": new_size})
                    previousPoint[handType] = (x, y)

    #LEFT HAND OPEN PALM: CLEAR FLOWERS
    if "Left" in detected_hands and left_clear_gesture and not clear_gesture_active:
        for item in flowers:
            item["state"] = "glowing"
            item["vy"] = 0
            item["glow_start"] = time.time()
        previousPoint["Left"]=None
        previousPoint["Right"] = None
    clear_gesture_active=left_clear_gesture

    for handType in ["Left","Right"]:
        if handType not in detected_hands:
            previousPoint[handType]=None

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

    #CROWN OVERLAY
    if (crown_mode_active and face_detector.results and face_detector.results.multi_face_landmarks):
        face_count = len(face_detector.results.multi_face_landmarks)
        for faceNo in range(face_count):
            face_landmarks=face_detector.findPosition(frame,faceNo=faceNo)
            forehead_points=face_detector.getForeheadLandmarks(frame,faceNo=faceNo)
            if not forehead_points:
                continue
            face_points = {point[0]: (point[1], point[2])for point in face_landmarks}
            required_points = [234, 454, 33, 263]
            if not all(point in face_points for point in required_points):
                continue
            face_width=abs(face_points[454][0]-face_points[234][0])
            eye_1 = face_points[33]
            eye_2 = face_points[263]
            left_eye, right_eye = sorted([eye_1, eye_2], key=lambda p: p[0])
            head_angle = math.degrees(math.atan2(right_eye[1] - left_eye[1],right_eye[0] - left_eye[0]))
            forehead_x=int(np.mean([point[1] for point in forehead_points]))
            forehead_y=int(np.mean([point[2] for point in forehead_points]))
            target_crown_width = int(face_width * 1.35)
            if faceNo not in crown_positions:
                crown_width = target_crown_width
                smoothed_angle = head_angle
            else:
                previous = crown_positions[faceNo]
                crown_width = int(previous["width"] +(target_crown_width - previous["width"]) * smoothing_factor)
                angle_difference = (head_angle - previous["angle"] + 180) % 360 - 180
                previous["angle"] += angle_difference * smoothing_factor
                smoothed_angle = previous["angle"]
            crown_height = int(crown_width * crown_image.shape[0] / crown_image.shape[1])
            resized_crown = cv2.resize(crown_image,(crown_width, crown_height),interpolation=cv2.INTER_AREA)
            rotated_crown = rotate_overlay(resized_crown, -smoothed_angle)
            target_x = forehead_x - rotated_crown.shape[1] // 2
            target_y = forehead_y - rotated_crown.shape[0] + int(crown_height * 0.15)
            if faceNo not in crown_positions:
                crown_positions[faceNo] = {"x": target_x,"y": target_y,"width": crown_width,"angle": head_angle}
            else:
                previous = crown_positions[faceNo]
                previous["x"] += (target_x - previous["x"]) * smoothing_factor
                previous["y"] += (target_y - previous["y"]) * smoothing_factor
                previous["width"] = crown_width
            crown_x = int(crown_positions[faceNo]["x"])
            crown_y = int(crown_positions[faceNo]["y"])
            output = overlay_transparent(output, rotated_crown, crown_x, crown_y)

    # CROWN BUTTON
    button_color = (180, 120, 200) if crown_mode_active else (80, 80, 80)
    button_text = "CROWN: ON" if crown_mode_active else "CROWN: OFF"
    cv2.rectangle(output,(20, 20),(190, 65),button_color,cv2.FILLED)
    cv2.putText(output,button_text,(35, 50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255), 2 )
    cv2.putText(output,f"FPS: {int(fps)}",(20, 100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)
    cv2.imshow("Flower Painting", output)
    key= cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = f"flower_art_{time.time_ns()}.png"
        save_path = os.path.join("Captured Photos", filename)
        success = cv2.imwrite(save_path, output)
        if success:
            print(f"Screenshot saved: {save_path}")
        else:
            print("Failed to save screenshot!")
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()