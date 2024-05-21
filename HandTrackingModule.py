from math import sqrt
import cv2
import time
import mediapipe as mp


class handDetector:
    def __init__(
        self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.model_complexity,
            self.detectionCon,
            self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # convert img to rgb cause hands ony work with rgb
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # result.multi_hand_landmarks is a list of the landmarks of the hands that are detected (1 hand or 2 hands)
        # in each handLms of each hand we have the id, position of each hand-knuckle
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        return img

    def findPositions(self, img, handNo=0, draw=True):
        # The list that contains id and position of each hand-knuckle in img
        landmarkList = []
        # Get the coordinatens of the frame
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                landmarkList.append([id, lm.x * w, lm.y * h])

        return landmarkList

    def isFingerUp(self, img, handNo=0):
        # If finger up -> 1, if not -> 0
        # finger = 0 -> thumb
        # finger = 1 -> index finger
        # finger = 2 -> middle finger
        # finger = 3 -> ring finger
        # finger = 4 -> baby finger
        fingersUp = [0] * 5
        lmList = []

        lmList = self.findPositions(img, handNo)

        if len(lmList) != 0:

            if lmList[4][2:] < lmList[2][2:]:
                fingersUp[0] = 1
            if lmList[8][2:] < lmList[6][2:]:
                fingersUp[1] = 1
            if lmList[12][2:] < lmList[10][2:]:
                fingersUp[2] = 1
            if lmList[16][2:] < lmList[14][2:]:
                fingersUp[3] = 1
            if lmList[20][2:] < lmList[18][2:]:
                fingersUp[4] = 1

        return fingersUp

    def findDistance(self, img, finger1, finger2):
        distance = 0.0
        lmList = self.findPositions(img)

        if len(lmList) != 0:
            x1, y1 = lmList[finger1][1], lmList[finger1][2]
            x2, y2 = lmList[finger2][1], lmList[finger2][2]
            distance = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
        return distance


handDetector = handDetector()


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Can not oen camera")
        exit()

    while True:
        # Read each frame
        result, img = cap.read()
        img_flip = cv2.flip(img, 1)
        img = handDetector.findHands(img_flip)
        landmarkList = handDetector.findPositions(img)
        fingersUp = handDetector.isFingerUp(img)

        print(fingersUp)

        distance = handDetector.findDistance(img, 8, 12)
        print(distance)

        # print(landmarkList)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )

        cv2.imshow("Image", img)
        # Enter key to close OpenCV
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
