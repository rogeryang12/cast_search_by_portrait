import os
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--mode', default='test')
parser.add_argument('--root', default='/data4/lixz/wider/body')
args = parser.parse_args()

mode = args.mode
root = args.root

outpath = 'data'
if not os.path.exists(outpath):
    os.makedirs(outpath)

# cast
with open('../data/{}.json'.format(mode)) as f:
    data = json.load(f)
movies = sorted(list(data.keys()))
with open('{}/{}_cast.txt'.format(outpath, mode), 'w') as f:
    for movie in movies:
        files = data[movie]['cast']
        for file in files:
            f.writelines(os.path.join(mode, file['img']) + '\n')

# candidates
for movie in os.listdir(os.path.join(root, mode)):
    folder = os.path.join(root, mode, movie, 'candidates')
    files = os.listdir(folder)
    with open('{}/{}_{}.txt'.format(outpath, mode, movie), 'w') as f:
        for file in files:
            f.writelines(os.path.join(mode, movie, 'candidates', file) + '\n')
