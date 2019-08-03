import time
import json
from PIL import Image
import os
import os.path as osp
import shutil
from tqdm import tqdm
import pickle
from multiprocessing import Pool
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--origin', default='/data4/lixz/wider/origin')
parser.add_argument('--body', default='/data4/lixz/wider/body')
parser.add_argument('--mode', default='test')
args = parser.parse_args()


root = args.origin
save_folder = args.body
mode = args.mode

with open('data/{}.json'.format(mode)) as f:
    data = json.load(f)

movies = sorted(data.keys())


def crop_body(key):
    candidates = data[key]['candidates']
    for candidate in candidates:
        file = candidate['img']
        bbox = candidate['bbox']
        img = Image.open(osp.join(root, mode, file))
        img = img.crop([bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]])
        file = '_'.join([file[:-4]] + [str(x) for x in bbox]) + file[-4:]
        file = osp.join(save_folder, mode, file)
        folder = osp.dirname(file)
        if not osp.exists(folder):
            os.makedirs(folder)
        img.save(file)


def save_body(mode):
    with open('data/{}.json'.format(mode)) as f:
        data = json.load(f)

    dataset = {}
    for movie in data.keys():
        dataset[movie] = {'cast': [], 'candidates': []}
        for cast in data[movie]['cast']:
            dd = {'id': cast['id'], 'file': cast['img'], 'label': cast['label']}
            dataset[movie]['cast'].append(dd)

        for candidate in data[movie]['candidates']:
            bbox = candidate['bbox']
            file = candidate['img']
            file = '_'.join([file[:-4]] + [str(x) for x in bbox]) + file[-4:]
            dd = {'id': candidate['id'], 'file': file}
            if mode != 'test':
                dd.update({'label': candidate['label']})
            dataset[movie]['candidates'].append(dd)

    with open('data/body_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(dataset, f)


def copy_cast(mode):
    with open('data/{}.json'.format(mode)) as f:
        data = json.load(f)

    movies = sorted(data.keys())
    for i, key in enumerate(movies, 1):
        print('{}/{}'.format(i, len(movies)) + ' movies')
        files = data[key]['cast']
        for file in tqdm(files):
            file = file['img']
            sour_file = osp.join(root, mode, file)
            save_file = osp.join(save_folder, mode, file)
            folder = osp.dirname(save_file)
            if not osp.exists(folder):
                os.makedirs(folder)
            shutil.copyfile(sour_file, save_file)


pool = Pool(20)
returns = []
for movie in movies:
    returns.append(pool.apply_async(crop_body, (movie,)))

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

copy_cast(mode)
save_body(mode)
