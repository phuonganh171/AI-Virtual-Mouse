import mediapipe as mp
import cv2
from HandTrackingModule import handDetector
import time
import autopy  # only work with python 3.8
import numpy as np

wCam, hCam = 640, 480
smoothen = 7

pX, pY = 0, 0
cX, cY = 0, 0

cTime = 0
pTime = 0

cap = cv2.VideoCapture(0)
handDetector = handDetector
wScreen, hScreen = autopy.screen.size()

if not cap.isOpened():
    print("Can not open camera")
    exit()

while True:
    ret, img_raw = cap.read()
    img = cv2.flip(img_raw, 1)

    img = handDetector.findHands(img)

    landmarksList = handDetector.findPositions(img)

    # check if index finger is up. If only index finger up -> moving
    fingersUp = handDetector.isFingerUp(img)

    if fingersUp[1] == 1 and fingersUp[2] == 0:
        # Moving mode
        x, y = landmarksList[8][1:]
        # convert coordinaten to the coordinaten of the screen
        x_scr, y_scr = (np.interp(x, (0, wCam), (0, wScreen))), np.interp(
            y, (0, wCam), (0, wScreen)
        )
        # Smoothen the movement
        x_smooth = pX + (x_scr - pX) / smoothen
        y_smooth = pY + (y_scr - pY) / smoothen

        pX = x_smooth
        pY = y_smooth

        autopy.mouse.move(x_smooth, y_smooth)

    # If index finger and middle finger up -> click if the distance < 40
    if fingersUp[1] == 1 and fingersUp[2] == 1:
        x1, y1 = landmarksList[8][1:]
        x2, y3 = landmarksList[12][1:]

        dis = handDetector.findDistance(img, 8, 12)
        if dis < 30:
            # Cliking mode
            print(dis)
            autopy.mouse.click()

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
    )

    cv2.imshow("image", img)
    cv2.waitKey(1)
