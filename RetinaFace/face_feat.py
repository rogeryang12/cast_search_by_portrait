import pickle
import argparse
import numpy as np
import mxnet as mx
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--num', default=30, type=int)
args = parser.parse_args()


def get_model(ctx, image_size, model, layer, epoch):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (args.num, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


with open('{}_img.pkl'.format(args.mode), 'rb') as f:
    results = pickle.load(f)
print('load data done')

image_size = (112, 112)
model = get_model(mx.gpu(args.gpu), image_size, 'model/model-r100-ii/model', 'fc1', 0)

step = 1
steps = len(results)
dataset = {}
for movie, data in results.items():

    data = mx.nd.array(data)
    feats = []
    num = args.num
    cnt = len(data) // num
    for i in tqdm(range(cnt)):
        db = mx.io.DataBatch(data=(data[num*i:num*(i+1)],))
        model.forward(db, is_train=False)
        embedding = model.get_outputs()[0].asnumpy()
        feats.append(embedding)
    if len(data) > cnt * num:
        db = mx.io.DataBatch(data=(data[num * cnt: len(data)],))
        model.forward(db, is_train=False)
        embedding = model.get_outputs()[0].asnumpy()
        feats.append(embedding)
    feats = np.concatenate(feats)
    dataset[movie] = feats
    print('finish {}/{}'.format(step, steps))
    step += 1

with open('../data/face_{}.pkl'.format(args.mode), 'wb') as f:
    pickle.dump(dataset, f)


