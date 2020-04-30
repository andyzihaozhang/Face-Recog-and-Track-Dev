import numpy as np
import cv2
import dlib
import time
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
from helpers import predict, get_forehead_coord, reshape_forehead_coord

import KCF

onDetecting = True
onTracking = False

onnx_path = 'model/ultra_light_320.onnx'  # OPTION: model/ultra_light_640.onnx
predictor_path = 'model/shape_predictor_5_face_landmarks.dat'
onnx_model = onnx.load(onnx_path)
detector = prepare(onnx_model)
predictor = dlib.shape_predictor(predictor_path)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

trackers = []

cap = cv2.VideoCapture(0)

# initiate loop and timer
loop = 0
start = time.time()

while True:
    loop += 1

    ret, frame = cap.read()
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert bgr to grey

    if onDetecting:
        # pre-process img acquired
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
        img = cv2.resize(img, (320, 240))  # OPTION: 640 * 480
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

        for i in range(boxes.shape[0]):
            face_box = boxes[i, :]
            bX, bY, bW, bH = face_box
            cv2.rectangle(frame, (bX, bY), (bW, bH), (0, 255, 0), 2)
            tracker = KCF.kcftracker(False, True, False, False)  # hog, fixed_window, multi-scale, lab
            tracker.init([bX, bY, bW - bX, bH - bY], frame)
            trackers.append(tracker)

            face_rect = dlib.rectangle(left=bX, top=bY, right=bW, bottom=bH)
            shape = predictor(gray, face_rect)  # get 5-point facial landmarks
            cv2.polylines(frame, get_forehead_coord(shape), True, (0, 255, 255), 2)  # draw forehead box

        # keep detecting until face found
        if boxes.shape[0] > 0:
            onDetecting = False
            onTracking = True

    elif onTracking:
        for tracker in trackers:
            face_bbox = tracker.update(frame)  # get tracked face bounding box
            f_x1, f_y1, f_w, f_h = face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]
            cv2.rectangle(frame, (f_x1, f_y1), (f_x1 + f_w, f_y1 + f_h), (0, 255, 0), 2)

            face_rect = dlib.rectangle(left=f_x1, top=f_y1, right=f_x1+f_w, bottom=f_y1+f_h)
            shape = predictor(gray, face_rect)  # get 5-point facial landmarks
            cv2.polylines(frame, get_forehead_coord(shape), True, (0, 255, 255), 2)  # draw forehead box

        # run detector every 10 frames
        if loop % 10 == 0:
            trackers = []
            forehead_trackers = []
            onDetecting = True
            onTracking = False

    cv2.imshow('tracking', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # end timer
        end = time.time()
        print("Average fps: ", loop / (end - start))
        break

cv2.destroyAllWindows()
cap.stop()
