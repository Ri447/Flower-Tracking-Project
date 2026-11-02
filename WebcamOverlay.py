import cv2

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

cap = cv2.VideoCapture(1)
canvas = None
while True:
    success, frame = cap.read()
    if not success:
        print("Cannot access webcam!")
        break
    if canvas is None:
        canvas = frame.copy()
    cv2.imshow("Canvas", canvas)
    frame = cv2.flip(frame, 1)
    x, y = 100, 100
    roi = frame[y:y+150, x:x+150]
    for c in range(3):
        roi[:, :, c] = (alpha * flower[:, :, c] + (1 - alpha) * roi[:, :, c])
    frame[y:y + 150, x:x + 150] = roi
    cv2.imshow("Webcam Flower Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
