"""
Preprocessing Step 3:
Splitting data to train, test and val.
"""

import shutil
import os
from glob import glob
import random
from tqdm import tqdm


def abs_name(path):
    name = os.path.basename(path)
    abs_name = '.'.join(name.split('.')[:-1])
    return abs_name


if __name__ == '__main__':

    video_classes = glob('dataset_videos/*')
    landmark_classes = glob('dataset_landmarks/*')
    print(len(video_classes), len(landmark_classes))

    for class_ in tqdm(video_classes):
        class_videos = glob(os.path.join(class_, '*.mp4'))
        test_valid = random.sample(class_videos, int(len(class_videos) * 0.3))
        test = random.sample(test_valid, int(len(test_valid) * 0.4))
        cls_name = os.path.basename(class_)
        os.makedirs(os.path.join(class_, 'train'))
        os.makedirs(os.path.join(class_, 'test'))
        os.makedirs(os.path.join(class_, 'val'))
        os.makedirs(os.path.join('dataset_landmarks', cls_name, 'train'))
        os.makedirs(os.path.join('dataset_landmarks', cls_name, 'test'))
        os.makedirs(os.path.join('dataset_landmarks', cls_name, 'val'))
        for video in class_videos:
            landmark_name = abs_name(video)
            if video in test:
                shutil.move(video, os.path.join(class_, 'test'))
                shutil.move(os.path.join('dataset_landmarks', cls_name, landmark_name + '.npz'), os.path.join('dataset_landmarks', cls_name, 'test'))
            elif video in test_valid:
                shutil.move(video, os.path.join(class_, 'val'))
                shutil.move(os.path.join('dataset_landmarks', cls_name, landmark_name + '.npz'), os.path.join('dataset_landmarks', cls_name, 'val'))
            else:
                shutil.move(video, os.path.join(class_, 'train'))
                shutil.move(os.path.join('dataset_landmarks', cls_name, landmark_name + '.npz'), os.path.join('dataset_landmarks', cls_name, 'train'))