import cv2
# import os
#
# folder = "Flowers"
# flowers=[]
# for filename in os.listdir(folder):
#     if filename.endswith(".png"):
#         path = os.path.join(folder, filename)
#         img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#         if img is not None:
#             flowers.append(img)
#
# print("Flowers loaded:", len(flowers))


overlay = cv2.imread("Flowers/1.png", cv2.IMREAD_UNCHANGED)
background = cv2.imread("Flowers/Background.jpg")
overlay=cv2.resize(overlay,(200,200))

if overlay is None:
    print("Flower image not found!")
    exit()

if overlay.shape[2] != 4:
    print("Flower image has no alpha channel!")
    exit()

alpha=overlay[:,:,3]/255.0
flower=overlay[:,:,:3]

x,y=100,100
roi=background[y:y+200,x:x+200]

for c in range(3):
    roi[:,:,c]=alpha*flower[:,:,c]+(1-alpha)*roi[:,:,c]  #ouput = a*flower+ (1-a)*background

background[y:y+200,x:x+200]=roi

# if overlay is None:
#     print("Image not found!")
# else:
#     print("Shape:", overlay.shape)
#     if overlay.shape[2] == 4:
#         print("Transparent PNG detected!")
#     else:
#         print("No alpha channel detected!")
cv2.imshow("Flower Overlay", background)
cv2.waitKey(0)
cv2.destroyAllWindows()