import cv2
import mediapipe as mp
import time
import math

class FaceMeshDetector():
    def __init__(self,staticMode=False,maxFaces=2,minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode =staticMode
        self.maxFaces =maxFaces
        self.minDetectionCon =minDetectionCon
        self.minTrackCon =minTrackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw=mp.solutions.drawing_utils
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        self.results = None
        self.foreheadIds=[10,67,69,103,104,108,109,297,299,332,333,337,338]

    def findFaceMesh(self,img,draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        # faces=[]
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
               if draw:
                   self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_TESSELATION,self.drawSpec,self.drawSpec)
               # for id,lm in enumerate(faceLms.landmark):
               #     # print(lm)
               #     ih, iw, ic = img.shape
               #     x,y=int(lm.x*iw), int(lm.y*ih)
               #     # print(id,x,y)
               #     face.append([x,y])
               # faces.append(face)
        return img

    def findPosition(self,img,faceNo=0):
        lmList=[]
        if self.results and self.results.multi_face_landmarks:
            face = self.results.multi_face_landmarks[faceNo]
            ih, iw, ic = img.shape
            for id, lm in enumerate(face.landmark):
                x = int(lm.x * iw)
                y = int(lm.y * ih)
                z = lm.z

                lmList.append([id, x, y, z])

        return lmList

    def getFeatureLandmarks(self, img, feature="left_eye", faceNo=0):

        featureConnections = {
            "left_eye": self.mpFaceMesh.FACEMESH_LEFT_EYE,
            "right_eye": self.mpFaceMesh.FACEMESH_RIGHT_EYE,
            "lips": self.mpFaceMesh.FACEMESH_LIPS,
            "nose": self.mpFaceMesh.FACEMESH_NOSE
        }
        if feature not in featureConnections:
            raise ValueError("Feature must be left_eye, right_eye, lips or nose")
        featureIds = set()
        for connection in featureConnections[feature]:
            featureIds.update(connection)
        lmList = self.findPosition(img, faceNo)
        return [lm for lm in lmList if lm[0] in featureIds]

    def getForeheadLandmarks(self, img, faceNo=0):
        lmList = self.findPosition(img, faceNo)
        return [
            lm for lm in lmList
            if lm[0] in self.foreheadIds
        ]

    def findDistance(self, p1, p2):
        return math.dist(p1, p2)

    def getEyeAspectRatio(self, img, faceNo=0):
        lmList = self.findPosition(img, faceNo)
        if not lmList:
            return None
        points = {lm[0]: (lm[1], lm[2]) for lm in lmList}
        leftEye = [362, 385, 387, 263, 373, 380]
        rightEye = [33, 160, 158, 133, 153, 144]

        def calculateEAR(eye):
            p1, p2, p3, p4, p5, p6 = [
                points[index] for index in eye
            ]
            vertical1 = self.findDistance(p2, p6)
            vertical2 = self.findDistance(p3, p5)
            horizontal = self.findDistance(p1, p4)
            return (vertical1 + vertical2) / (2 * horizontal)

        leftEAR = calculateEAR(leftEye)
        rightEAR = calculateEAR(rightEye)
        return (leftEAR + rightEAR) / 2

    def getMouthAspectRatio(self, img, faceNo=0):
        lmList = self.findPosition(img, faceNo)
        if not lmList:
            return None
        points = {lm[0]: (lm[1], lm[2]) for lm in lmList}
        upperLip = points[13]
        lowerLip = points[14]
        leftCorner = points[78]
        rightCorner = points[308]
        vertical = self.findDistance(upperLip, lowerLip)
        horizontal = self.findDistance(leftCorner, rightCorner)

        return vertical / horizontal

def main():
    cap = cv2.VideoCapture(1)
    # pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img=detector.findFaceMesh(img)
        lmList = detector.findPosition(img)
        eyes = detector.getFeatureLandmarks(img, "left_eye")
        lips = detector.getFeatureLandmarks(img, "lips")
        nose = detector.getFeatureLandmarks(img, "nose")
        forehead = detector.getForeheadLandmarks(img)
        ear = detector.getEyeAspectRatio(img)
        mar = detector.getMouthAspectRatio(img)
        if ear is not None:
            cv2.putText(
                img, f"EAR: {ear:.3f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2
            )
        if mar is not None:
            cv2.putText(
                img, f"MAR: {mar:.3f}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2
            )
        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        # cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__=='__main__':
    main()