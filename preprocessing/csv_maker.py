"""
Preprocessing Step 4:
Create csv file for landmark paths.
"""

from glob import glob

if __name__ == '__main__':
    landmarks = glob('dataset_landmarks/*/*/*.npz')

    with open('landmarks_paths.csv', 'w') as f:
        for path in landmarks:
            path = '/'.join(path.split('/')[1:])
            f.write(path[:-4] + ', 0' + '\n')
