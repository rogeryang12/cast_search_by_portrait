import os
import random
import math
import torch
import pickle
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class RandomErasing(object):
    """
    @author:  liaoxingyu
    @contact: liaoxingyu2@jd.com
    """
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def get_transform(resize=None, flip=0.5, pad=None, crop=None, imagenet_norm=True, erase=False):
    transform = []
    if resize is not None:
        transform.append(T.Resize(resize, interpolation=3))
    if flip is not None:
        transform.append(T.RandomHorizontalFlip(flip))
    if pad is not None:
        transform.append(T.Pad(pad))
    if crop is not None:
        transform.append(T.RandomCrop(crop))

    if imagenet_norm:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    transform.extend([T.ToTensor(), T.Normalize(mean, std)])
    if erase:
        transform.append(RandomErasing(mean=mean))

    return T.Compose(transform)


def load_pkl(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


class TrainBase(Dataset):

    def __init__(self, image_root, pkl_file, transform, use_val=False):

        super(TrainBase, self).__init__()
        self.image_root = image_root
        self.dataset = {}

        if len(pkl_file) > 0:
            data = load_pkl(pkl_file)
            for movie, subs in data.items():
                for item in subs['candidates']:
                    label = '_'.join([movie, item['label']])
                    if label not in self.dataset:
                        self.dataset[label] = []
                    self.dataset[label].append(os.path.join('train', item['file']))

            if use_val:
                data = load_pkl(pkl_file.replace('train', 'val'))
                for movie, subs in data.items():
                    for item in subs['candidates']:
                        label = '_'.join([movie, item['label']])
                        if label not in self.dataset:
                            self.dataset[label] = []
                        self.dataset[label].append(os.path.join('val', item['file']))

        pids = [label for label in self.dataset.keys() if 'others' not in label]
        self.pids = sorted(pids)
        self.label_dict = {pid: label for label, pid in enumerate(self.pids)}
        self.num_classes = len(self.pids)
        self.trans = transform

    def transform(self, img):
        if isinstance(img, list):
            return torch.stack([self.transform(im) for im in img])
        img = Image.open(os.path.join(self.image_root, img))
        return self.trans(img)


class TrainLabel(TrainBase):
    def __init__(self, image_root, pkl_file, transform, use_val=False):
        super(TrainLabel, self).__init__(image_root, pkl_file, transform, use_val)
        self.images = []
        self.labels = []
        for pid in self.pids:
            files = self.dataset[pid]
            self.images.extend(files)
            self.labels.extend([self.label_dict[pid]] * len(files))
        del self.dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(self.images[item]), self.labels[item]


class TrainKBatch(TrainBase):
    """
    images: image_num x c x h x w    image_num images
    labels: image_num    person label of image_num images
    """

    def __init__(self, image_root, pkl_file, transform, image_num, use_val=False):
        super(TrainKBatch, self).__init__(image_root, pkl_file, transform, use_val)
        self.image_num = image_num

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, item):
        pid = self.pids[item]
        files = self.dataset[pid]
        index = list(range(len(files)))
        if len(files) < self.image_num:
            index = index * (self.image_num // len(files) + 1)
        index = random.sample(index, self.image_num)
        images = [files[i] for i in index]
        images = self.transform(images)
        return images, torch.LongTensor([self.label_dict[pid]] * self.image_num)


class TestImage(Dataset):

    def __init__(self, test_root, test_file, resize=None, crop_size=None, flip=False, imagenet_norm=True):
        super(TestImage, self).__init__()
        self.test_root = test_root

        self.resize = resize
        self.crop_size = crop_size
        self.flip = flip

        if imagenet_norm:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        trans = [T.ToTensor(), T.Normalize(mean, std)]
        self.totensor = T.Compose(trans)

        with open(test_file, 'rb') as f:
            data = pickle.load(f)

        self.dataset = {}
        self.files = []
        for movie in sorted(data.keys()):
            num = len(self.files)
            files = [item['file'] for item in data[movie]['candidates']]
            self.files.extend(files)
            self.dataset[movie] = [num, len(self.files)]

    def transform(self, img):
        img = Image.open(os.path.join(self.test_root, img))

        if self.resize is not None:
            img = T.Resize(self.resize, interpolation=3)(img)
        if self.crop_size is not None:
            imgs = T.FiveCrop(self.crop_size)(img)
        else:
            imgs = [img]
        if self.flip:
            imgs = [T.RandomHorizontalFlip(p=1)(img) for img in imgs] + list(imgs)
        imgs = [self.totensor(img) for img in imgs]
        return torch.stack(imgs)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return self.transform(self.files[item])

    def save(self, embs, name):
        results = {}
        for movie, idx in self.dataset.items():
            results[movie] = embs[idx[0]:idx[1]]

        with open(name, 'wb') as f:
            pickle.dump(results, f)