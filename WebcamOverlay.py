import cv2
import time
from FingerTracking import handDetector
import numpy as np

overlay=cv2.imread("Flowers/2.png",cv2.IMREAD_UNCHANGED)

if overlay is None:
    print("Flower image not found!")
    exit()

if overlay.shape[2] != 4:
    print("Flower must have an alpha channel!")
    exit()

overlay=cv2.resize(overlay,(150,150))
alpha = overlay[:, :, 3] / 255.0
flower = overlay[:, :, :3]

cap = cv2.VideoCapture(0)
detector = handDetector()
flowers = []
previousPoint = None
while True:
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

        x -= 75
        y -= 75
        #DRAW FLOWERS
        if index_up and not middle_up and not ring_up and not pinky_up:
            if previousPoint is None or np.linalg.norm(np.array([x, y]) - np.array(previousPoint)) > 50:
                if 0 <= x <= frame.shape[1] - 150 and 0 <= y <= frame.shape[0] - 150:
                    flowers.append({"x": x,"y": y,"start": time.time(),"phase": np.random.uniform(0, 2 * np.pi)})
                    previousPoint = (x, y)
        if all(fingers_up):
            flowers.clear()
            previousPoint = None
    else:
        previousPoint = None

    output=frame.copy()
    for item in flowers:
        age=time.time()-item["start"]
        scale=min(age/0.4,1.0)
        angle=np.sin(age*2+item["phase"])*8
        float_y=int(np.sin(age*2.2+item["phase"])*4)
        matrix=cv2.getRotationMatrix2D((75,75),angle,scale)
        animated = cv2.warpAffine(overlay,matrix,(150,150),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
        x,y=item["x"],item["y"]+float_y
        h,w,_ = output.shape
        x1,y1=max(0,x),max(0,y)
        x2,y2=min(w,x+150),min(h,y+150)
        if x1<x2 and y1<y2:
            flower_crop = animated[y1 - y:y2 - y, x1 - x:x2 - x]
            alpha = flower_crop[:, :, 3:4] / 255.0
            flower_rgb = flower_crop[:, :, :3]
            background = output[y1:y2, x1:x2]
            output[y1:y2, x1:x2] = (alpha * flower_rgb + (1 - alpha) * background).astype(np.uint8)

    cv2.imshow("Flower Painting", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()