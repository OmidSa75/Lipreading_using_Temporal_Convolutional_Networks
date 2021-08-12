"""
Preprocessing Step 2:
Change data folders to class base folder.
You have to do this both for landmarks and videos so replace dataset_landmarks to dataset_videos or vise versa.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from concurrent import futures


def extract_class(path):
    name = os.path.basename(path).split('.')[0]
    cls = name[2:5]
    return cls


def copy_files(file):
    cls = extract_class(file)
    shutil.copy(file, os.path.join('dataset_landmarks', cls))
    print(file + ' copied')


if __name__ == '__main__':
    # both for landmarks and videos
    data = glob('landmarks/*/*.npz')
    clses = list(map(extract_class, data))

    os.makedirs('dataset_landmarks', exist_ok=True)
    for cls in set(clses):
        os.makedirs(os.path.join('dataset_landmarks', cls), exist_ok=True)

    for file in tqdm(data):
        cls = extract_class(file)
        shutil.copy(file, os.path.join('dataset_landmarks', cls))

