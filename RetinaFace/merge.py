import os
import json
import pickle
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--mode', default='test')
args = parser.parse_args()

mode = args.mode
with open('../data/body_{}.pkl'.format(mode), 'rb') as f:
    origin = pickle.load(f)


dataset = {}
with open('results/{}_cast.pkl'.format(mode), 'rb') as f:
    cast_data = pickle.load(f)

for movie in tqdm(origin.keys()):
    dataset[movie] = {'cast': [], 'candidates': []}
    with open('results/{}_{}.pkl'.format(mode, movie), 'rb') as f:
        data = pickle.load(f)

    for item in origin[movie]['candidates']:
        file = item['file']
        sub_data = data[os.path.join(mode, file)]
        faces = sub_data['faces']
        if len(faces) == 0:
            faces = None
            scores = None
        else:
            scores = faces[:, 4]
            faces = faces[:, :4]
        landmarks = sub_data['landmarks']
        if len(landmarks) == 0:
            landmarks = None
        else:
            landmarks = landmarks.transpose(0, 2, 1).reshape(-1, 10)

        if mode == 'test':
            dd = {'faces': faces, 'landmarks': landmarks, 'scores': scores, 'file': file, 'id': item['id']}
        else:
            dd = {'faces': faces, 'landmarks': landmarks, 'scores': scores, 'file': file, 'id': item['id'], 'label': item['label']}
        dataset[movie]['candidates'].append(dd)

    for item in origin[movie]['cast']:
        file = item['file']
        sub_data = cast_data[os.path.join(mode, file)]
        faces = sub_data['faces']
        if len(faces) == 0:
            faces = None
            scores = None
        else:
            scores = faces[:, 4]
            faces = faces[:, :4]
        landmarks = sub_data['landmarks']
        if len(landmarks) == 0:
            landmarks = None
        else:
            landmarks = landmarks.transpose(0, 2, 1).reshape(-1, 10)
        dd = {'faces': faces, 'landmarks': landmarks, 'scores': scores, 'file': file, 'id': item['id'], 'label': item['label']}
        dataset[movie]['cast'].append(dd)

with open('../data/retina_{}.pkl'.format(mode), 'wb') as f:
    pickle.dump(dataset, f)