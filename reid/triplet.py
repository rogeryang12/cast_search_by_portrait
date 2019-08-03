import torch
from torch import optim
from argparse import ArgumentParser

from utils.models import CNN
from utils.datasets import TrainKBatch, get_transform, TestImage
from utils.losses import CrossEntropyLoss, TripletLoss
from utils.trainer import Trainer, Tester

parser = ArgumentParser(description='Train a Reid network')
parser.add_argument('--experiment_root', default='TRI_')

parser.add_argument('--height', default=256, type=int)
parser.add_argument('--width', default=128, type=int)

parser.add_argument('--image_root', default='/data4/lixz/wider/body')
parser.add_argument('--pkl_file', default='../data/body_train.pkl')
parser.add_argument('--batch_size', default=18, type=int)
parser.add_argument('--image_num', default=4, type=int)
parser.add_argument('--use_val', default=False, action='store_true')

parser.add_argument('--cnn', default='resnet50')
parser.add_argument('--ckpt', default='/data4/lixz/models/resnet50.pth', help='imagenet pretrained model')
parser.add_argument('--stride', default=1, type=int)

parser.add_argument('--margin', default='soft')
parser.add_argument('--type', default='adaptive')

parser.add_argument('--epochs', default=450, type=int)
parser.add_argument('--lr', default=0.0003, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--save_epoch', default=50, type=int)
parser.add_argument('--lr_decay_epoch', default=150, type=int)
parser.add_argument('--gamma', default=0.1, type=float)

parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--num_gpu', default=torch.cuda.device_count())
parser.add_argument('--eval', default=False, action='store_true')

parser.add_argument('--checkpoint', default='model450.pkl')
parser.add_argument('--test_batch_size', default=256, type=int)
parser.add_argument('--test_root', default='/data4/lixz/wider/body/')
parser.add_argument('--mode', default='val')
args = parser.parse_args()

args.test_file = '../data/body_{}.pkl'.format(args.mode)
args.test_root += args.mode
args.experiment_root += args.cnn


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transform((args.height, args.width), 0.5, 10, (args.height, args.width), True, True)
    dataset = TrainKBatch(args.image_root, args.pkl_file, transform, args.image_num, args.use_val)
    model = CNN(num_classes=len(dataset.pids), cnn=args.cnn, ckpt=args.ckpt, stride=args.stride)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epoch, args.gamma)
    celoss = CrossEntropyLoss()
    triloss = TripletLoss(margin=args.margin, mode=args.type).to(device)

    trainer = Trainer(args, dataset, model, optimizer, scheduler)

    def cal_loss(data, model):
        images, labels = data
        b, k, c, h, w = images.size()
        images, labels = images.view(-1, c, h, w).to(device), labels.view(-1).to(device)
        embs, logits = model(images)
        loss1, prec = celoss(logits, labels)
        loss2, top1 = triloss(embs, labels)
        loss = loss1 + loss2
        log = ['loss', loss.item(), 'classloss', loss1.item(), 'triloss', loss2.item(), 'top1', top1, 'prec', prec]
        return loss, log

    trainer.train(cal_loss)


def evaluate():
    model = CNN(cnn=args.cnn, stride=args.stride)
    dataset = TestImage(args.test_root, args.test_file, (args.height, args.width), flip=True)
    tester = Tester(args, model, dataset)
    embs = tester.image_feature()
    dataset.save(embs, '{}/triplet_{}_{}.pkl'.format(args.experiment_root, args.cnn, args.mode))


if __name__ == '__main__':
    if not args.eval:
        train()
    else:
        evaluate()
