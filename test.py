from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

dlib.DLIB_USE_CUDA = True
if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    webcam = cv2.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for i, rect in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                print(shape.shape)
                (x, y, w, h) = face_utils.rect_to_bb(rect)

                cv2.putText(frame, "Face #{}".format(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            cv2.imshow('webcam', frame)
            if cv2.waitKey(20) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break
