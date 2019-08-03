import cv2
import sys
import numpy as np
import os
import pickle
from tqdm import tqdm
from argparse import ArgumentParser

from retinaface import RetinaFace

parser = ArgumentParser()
parser.add_argument('--txt_file', default='data/test_cast.txt')
parser.add_argument('--pkl_file', default='results/test_cast.pkl')
parser.add_argument('--image_root', default='/data4/lixz/wider/body')
args = parser.parse_args()

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
thresh = 0.8

TEST_SCALES = [500, 800, 1100, 1400, 1700]
target_size = 800
max_size = 1200

gpuid = 0
detector = RetinaFace('model/R50', 0, gpuid, 'net3')

folder = os.path.dirname(args.pkl_file)
if not os.path.exists(folder):
    os.makedirs(folder)

with open(args.txt_file) as f:
    files = f.readlines()

files = [file.strip() for file in files]

num = 0
results = {}
for file in tqdm(files):
    img = cv2.imread(os.path.join(args.image_root, file))
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [float(scale) / target_size * im_scale for scale in TEST_SCALES]

    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=False)
    results[file] = {'file': file, 'faces': faces, 'landmarks': landmarks}

with open(args.pkl_file, 'wb') as f:
    pickle.dump(results, f)