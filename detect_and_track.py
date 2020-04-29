import numpy as np
import cv2
import sys
import dlib
from time import time

import KCF

onDetecting = True
onTracking = False

detector = dlib.get_frontal_face_detector()
trackers = []

cap = cv2.VideoCapture(0)
loop = 0

while True:
    loop += 1
    ret, frame = cap.read()
    if not ret:
        break

    if onDetecting:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        face_cnt = len(faces)
        for face in faces:
            # compute the bounding box of the face and draw it on the frame
            bX, bY, bW, bH = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (bX, bY), (bW, bH), (0, 255, 0), 1)

            tracker = KCF.kcftracker(True, True, True, False)  # hog, fixed_window, multiscale, lab
            tracker.init([bX, bY, bW - bX, bH - bY], frame)
            trackers.append(tracker)

        # keep detecting until face found
        if face_cnt > 0:
            onDetecting = False
            onTracking = True

    elif onTracking:
        for tracker in trackers:
            boundingbox = tracker.update(frame)  # frame had better be contiguous

            # boundingbox = map(int, boundingbox)
            x1, y1, w, h = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[3]
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # run detector every 10 frames
        if loop % 10 == 0:
            trackers = []
            onDetecting = True
            onTracking = False

    cv2.imshow('tracking', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.stop()
