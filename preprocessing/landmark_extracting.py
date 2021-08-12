"""
Preprocessing Step 1:
Extracting landmarks form videos and save it in npz format.
"""


from imutils import face_utils
import numpy as np
import dlib
import cv2
import os
from glob import glob
from concurrent import futures


def extract_id(path):
    iD = os.path.basename(path)
    iD = iD.split('.')[0]
    return iD


def extract_person_id(path):
    return os.path.basename(os.path.dirname(path))


def extract_landmark(video_path):
    video = cv2.VideoCapture(video_path)
    person_id = extract_person_id(video_path)
    person_id_path = os.path.join('landmarks', person_id)
    os.makedirs(person_id_path, exist_ok=True)
    video_id = extract_id(video_path)
    landmarks = []
    while True:
        ret, frame = video.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for i, rect in enumerate(rects):  # since or videos have one face, i break at the end of loop
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                landmarks.append(dict(id=video_id, facial_landmarks=shape))
                break

        else:

            break
    landmarks = np.expand_dims(np.array(landmarks), axis=1)
    np.savez(os.path.join(person_id_path, video_id), data=landmarks)
    print(video_path + ' Done...')


dlib.DLIB_USE_CUDA = True
if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    video_paths = glob('../videos/*/*.mp4')

    with futures.ProcessPoolExecutor() as executor:
        executor.map(extract_landmark, video_paths)
