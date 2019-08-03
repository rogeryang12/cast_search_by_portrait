import os
import cv2
import time
import pickle
import argparse
import numpy as np
from skimage import transform as trans
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test')
parser.add_argument('--image_root', default='/data4/lixz/wider/body')
parser.add_argument('--select', default=1, type=int)
args = parser.parse_args()


def select(weight, height, bboxes, landmarks):
    if bboxes is None:
        return None, None
    if len(bboxes) == 1:
        ind = 0
    else:
        dist, area = [], []
        midx, midy = weight / 2, height / 3
        for bbox in bboxes:
            x, y = (bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2
            dist.append(np.sqrt((x - midx) * (x - midx) + (y - midy) * (y - midy)))
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        dist = np.array(dist)
        area = np.array(area)
        p = area / dist
        ind = np.argsort(p)[-1]
    return bboxes[ind], landmarks[ind] if landmarks is not None else None


def cast_select(width, height, bboxes, landmarks, scores):
    if bboxes is None:
        return None, None
    if len(bboxes) == 1:
        ind = 0
    else:
        dist, area = [], []
        midx, midy = width / 2, height / 3
        for bbox in bboxes:
            x, y = (bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2
            dist.append(np.sqrt((x - midx) * (x - midx) + (y - midy) * (y - midy)))
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        dist = np.array(dist)
        area = np.array(area)
        scores = (scores > 0.95).astype(np.int)
        if np.sum(scores) == 0:
            return None, None
        p = scores * area / dist
        ind = np.argsort(p)[-1]
    return bboxes[ind], landmarks[ind] if landmarks is not None else None


def candi_select(width, height, bboxes, landmarks, scores):
    if bboxes is None:
        return None, None

    inds, dists, areas = [], [], []
    midx, midy = width / 2, height / 3
    thres = width * height / 50
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        if scores[i] > 0.9 and area > thres and (width / 6 < x < width * 5 / 6):
            inds.append(i)
            dists.append(np.sqrt((x - midx) * (x - midx) + (y - midy) * (y - midy)))
            areas.append(area)

    if len(inds) == 0:
        return None, None
    else:
        dists = np.array(dists)
        areas = np.array(areas)
        p = areas / dists
        ind = np.argsort(p)[-1]
        ind = inds[ind]
    return bboxes[ind], landmarks[ind] if landmarks is not None else None


def preprocess(img, bbox=None, landmark=None, image_size=(112, 112), margin=44):
    if landmark is not None:  # do align using landmark
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
        dst = landmark.reshape((2, 5)).T.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
        # src = src[0:3,:]
        # dst = dst[0:3,:]
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

        # tform3 = trans.ProjectiveTransform()
        # tform3.estimate(src, dst)
        # warped = trans.warp(img, tform3, output_shape=_shape)
        return warped

    else:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox[:4]

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret


with open('../data/retina_{}.pkl'.format(args.mode), 'rb') as f:
    dataset = pickle.load(f)


def get_img(movie):

    data = []

    cast = dataset[movie]['cast']
    files = [os.path.join(args.image_root, args.mode, file['file']) for file in cast]
    bboxes = [file['faces'] for file in cast]
    landmarks = [file['landmarks'] for file in cast]
    scores = [file['scores'] for file in cast]
    for i in range(len(cast)):
        img = cv2.imread(files[i])
        height, width, _ = img.shape
        if args.select == 0:
            bbox, landmark = select(width, height, bboxes[i], landmarks[i])
        else:
            bbox, landmark = cast_select(width, height, bboxes[i], landmarks[i], scores[i])
        img = preprocess(img, bbox, landmark, image_size=(112, 112), margin=44)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        data.append(img)

    candidates = dataset[movie]['candidates']
    files = [os.path.join(args.image_root, args.mode, file['file']) for file in candidates]
    bboxes = [file['faces'] for file in candidates]
    landmarks = [file['landmarks'] for file in candidates]
    scores = [file['scores'] for file in candidates]

    for i in range(len(candidates)):
        img = cv2.imread(files[i])
        height, width, _ = img.shape
        if args.select == 0:
            bbox, landmark = select(width, height, bboxes[i], landmarks[i])
        else:
            bbox, landmark = candi_select(width, height, bboxes[i], landmarks[i], scores[i])
        img = preprocess(img, bbox, landmark, image_size=(112, 112), margin=44)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        data.append(img)
    data = np.stack(data, axis=0)
    return data, movie


pool = Pool(40)

movies = sorted(list(dataset.keys()))
returns = []
for movie in movies:
    returns.append(pool.apply_async(get_img, (movie, )))

while True:
    incomplete_count = sum(1 for x in returns if not x.ready())

    if incomplete_count == 0:
        print()
        print("All done")
        break

    print('\r{} Tasks Remaining'.format(incomplete_count), flush=True, end=' ')
    time.sleep(0.25)

pool.close()
pool.join()


results = {}
for ret in returns:
    data, movie = ret.get()
    results[movie] = data

with open('{}_img.pkl'.format(args.mode), 'wb') as f:
    pickle.dump(results, f)
