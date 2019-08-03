import os
import sys
import time
import numpy as np
from datetime import timedelta
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


class Logger(object):
    """
    generate logger files(train.log and tensorboard log file) in experiment root
    """

    def __init__(self, args, mode='train'):
        if not os.path.exists(args.experiment_root):
            os.makedirs(args.experiment_root)
        else:
            print("Experiment root `{}` already exists".format(args.experiment_root))
            sys.exit()

        self.writer = SummaryWriter(args.experiment_root)

        print('{} using the following parameters:'.format(mode))
        for k, v in sorted(vars(args).items()):
            print('{}: {}'.format(k, v))
        # self.iters = args.iters
        self.iter_per_epoch = args.iter_per_epoch
        self.epochs = args.epochs
        self.iters = args.epochs * args.iter_per_epoch
        self.start_time = time.time()
        self.times = [0] * 20
        self.i = 0

    def save_log(self, epoch, step,  log):
        global_step = step + (epoch - 1) * self.iter_per_epoch
        p_time = time.time()
        self.times[self.i] = p_time - self.start_time
        self.start_time = p_time
        self.i = (self.i + 1) % 20
        eta = int((self.iters - global_step) * sum(self.times) / 20)

        info = 'Epoch {}/{} Iter {}/{} -> '.format(epoch, self.epochs, step, self.iter_per_epoch)
        for i in range(len(log) // 2):
            k, v = log[2 * i], log[2 * i + 1]
            self.writer.add_scalar(k, v, global_step)
            info += '{} : {:.3f}, '.format(k, v)
        print(info + 'ETA : {}'.format(timedelta(seconds=eta)))


class Trainer(object):
    """
    Train a model
    """

    def __init__(self, args, dataset, model, optimizer, scheduler=None, device=None, cudnn=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.loader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
        self.model = torch.nn.DataParallel(model).to(self.device).train()
        self.optimizer = optimizer
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, args.lr_decay_epoch, args.gamma)
        else:
            self.scheduler = scheduler
        args.iter_per_epoch = len(self.loader)
        self.args = args
        self.Log = Logger(args)
        if cudnn:
            torch.backends.cudnn.benchmark = True
        self.start_epoch = 1

    def train(self, cal_loss):
        for epoch in range(self.start_epoch, self.args.epochs+1):
            self.scheduler.step()
            for step, data in enumerate(self.loader, 1):
                self.optimizer.zero_grad()
                ret = cal_loss(data, self.model)
                loss, log = ret
                loss.backward()
                self.optimizer.step()
                self.Log.save_log(epoch, step, log)

            if epoch % self.args.save_epoch == 0:
                state_dict = self.model.cpu().module.state_dict()
                torch.save(state_dict, os.path.join(self.args.experiment_root, 'model{}.pkl'.format(epoch)))
                self.model.to(self.device)


class Tester(object):

    def __init__(self, args, model, dataset, cudnn=True):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.checkpoint is not None:
            model.load_state_dict(torch.load(os.path.join(args.experiment_root, args.checkpoint)), False)
        self.model = torch.nn.DataParallel(model).to(self.device).eval()

        self.loader = DataLoader(dataset, args.test_batch_size, num_workers=args.num_workers)

        torch.set_grad_enabled(False)
        if cudnn:
            torch.backends.cudnn.benchmark = True

    def image_feature(self):
        print('Compute image features')
        embs = []
        for images in tqdm(self.loader):
            images = images.to(self.device)
            b, au, c, h, w = images.size()
            emb = self.model(images.view(-1, c, h, w))
            _, *s = emb.size()
            emb = emb.view(b, au, *s).mean(dim=1)
            embs.append(emb.cpu().numpy())

        embs = np.concatenate(embs, axis=0)
        torch.cuda.empty_cache()
        return embs