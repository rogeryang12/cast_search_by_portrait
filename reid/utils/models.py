import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

__all__ = [
    'ResNet50', 'ResNet101', 'ResNet152', 'ResNext50_32x4d', 'ResNext101_32x8d',
    'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201',
    'CNN',
]


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)


class ResNet(nn.Module):
    def __init__(self, cnn='resnet50', ckpt=None, stride=2):
        super(ResNet, self).__init__()
        resnet = getattr(torchvision.models, cnn)()
        if ckpt is not None:
            resnet.load_state_dict(torch.load(ckpt))
        else:
            resnet.apply(weight_init)
        resnet.layer4[0].downsample[0].stride = (stride, stride)
        resnet.layer4[0].conv2.stride = (stride, stride)

        self.dim = resnet.fc.in_features
        del resnet.fc
        del resnet.avgpool

        for name, module in resnet.named_children():
            setattr(self, name, module)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def get_part(self, start=None, end=None):
        models = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        start = 0 if start is None else models.index(start)
        end = 7 if end is None else models.index(end)
        models = [getattr(self, models[i]) for i in range(start, end+1)]
        return nn.Sequential(*models)


class ResNet50(ResNet):
    def __init__(self, ckpt=None, stride=2):
        super(ResNet50, self).__init__('resnet50', ckpt, stride)


class ResNet101(ResNet):
    def __init__(self, ckpt=None, stride=2):
        super(ResNet101, self).__init__('resnet101', ckpt, stride)


class ResNet152(ResNet):
    def __init__(self, ckpt=None, stride=2):
        super(ResNet152, self).__init__('resnet152', ckpt, stride)


class ResNext50_32x4d(ResNet):
    def __init__(self, ckpt=None, stride=2):
        super(ResNext50_32x4d, self).__init__('resnext50_32x4d', ckpt, stride)


class ResNext101_32x8d(ResNet):
    def __init__(self, ckpt=None, stride=2):
        super(ResNext101_32x8d, self).__init__('resnext101_32x8d', ckpt, stride)


def _load_state_dict(model, ckpt):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(ckpt)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


class DenseNet(nn.Module):
    def __init__(self, cnn='densenet121', ckpt=None, stride=2):
        super(DenseNet, self).__init__()
        densenet = getattr(torchvision.models, cnn)()
        if ckpt is not None:
            _load_state_dict(densenet, ckpt)
        else:
            densenet.apply(weight_init)
        self.features = densenet.features
        self.dim = densenet.classifier.in_features

    def forward(self, x):
        x = self.features(x)
        return x

    def get_part(self, start=None, end=None):
        models = ['conv0', 'norm0', 'relu0', 'pool0', 'denseblock1', 'transition1',
                  'denseblock2', 'transition2', 'denseblock3', 'transition3', 'denseblock4', 'norm5']
        start = 0 if start is None else models.index(start)
        end = 11 if end is None else models.index(end)
        models = [getattr(self.features, models[i]) for i in range(start, end+1)]
        return nn.Sequential(*models)


class DenseNet121(DenseNet):
    def __init__(self, ckpt=None, stride=2):
        super(DenseNet121, self).__init__('densenet121', ckpt, stride)


class DenseNet169(DenseNet):
    def __init__(self, ckpt=None, stride=2):
        super(DenseNet169, self).__init__('densenet169', ckpt, stride)


class DenseNet201(DenseNet):
    def __init__(self, ckpt=None, stride=2):
        super(DenseNet201, self).__init__('densenet201', ckpt, stride)


class DenseNet161(DenseNet):
    def __init__(self, ckpt=None, stride=2):
        super(DenseNet161, self).__init__('densenet161', ckpt, stride)


_models = {
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'resnext50_32x4d': ResNext50_32x4d,
    'resnext101_32x8d': ResNext101_32x8d,
    'densenet121': DenseNet121,
    'densenet169': DenseNet169,
    'densenet201': DenseNet201,
    'densenet161': DenseNet161,
}


class CNN(nn.Module):
    def __init__(self, cnn='resnet50', ckpt=None, stride=2, num_classes=None):
        super(CNN, self).__init__()

        self.backbone = _models[cnn](ckpt, stride)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(self.backbone.dim)
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.logits = nn.Linear(self.backbone.dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).view(x.size(0), x.size(1))
        x = self.bn(x)
        if self.training:
            logit = self.logits(x)
            return x, logit
        else:
            return x


class CNNAddMargin(nn.Module):
    def __init__(self, cnn='resnet50', ckpt=None, stride=2, num_classes=None, s=70, m=0.25):
        super(CNNAddMargin, self).__init__()

        self.backbone = _models[cnn](ckpt, stride)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(self.backbone.dim)

        self.s = s
        self.m = m

        self.num_classes = num_classes
        if self.num_classes is not None:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, self.backbone.dim))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        x = self.backbone(x)
        x = self.pool(x).view(x.size(0), x.size(1))
        x = self.bn(x)
        if not self.training:
            return x

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(x.device)
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output)
        return x, output


class MGN(nn.Module):
    def __init__(self, cnn='resnet50', ckpt=None, num_classes=None):
        super(MGN, self).__init__()

        if 'dense' in cnn:
            model = DenseNet(cnn, ckpt)
            self.backbone = model.get_part('conv0', 'denseblock3')
            self.dim = model.dim
            self.branch1 = nn.Sequential(DenseNet(cnn, ckpt).features.transition3,
                                         DenseNet(cnn, ckpt).features.denseblock4)
            self.branch2 = nn.Sequential(DenseNet(cnn, ckpt).features.transition3,
                                         DenseNet(cnn, ckpt).features.denseblock4)
            self.branch2[0].pool = nn.AvgPool2d(1, 1, 0)

        else:
            model = ResNet(cnn, ckpt)
            self.backbone = model.get_part('conv1', 'layer3')
            self.dim = model.dim
            self.branch1 = ResNet(cnn, ckpt, 2).layer4
            self.branch2 = ResNet(cnn, ckpt, 1).layer4

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d((2, 1))
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.dim) for _ in range(4)])
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.logits = nn.ModuleList([nn.Linear(self.dim, self.num_classes) for _ in range(4)])

    def forward(self, x):
        batch = x.size(0)
        x = self.backbone(x)

        b1 = self.branch1(x)
        emb1_g = self.pool1(b1).view(batch, self.dim)

        b2 = self.branch2(x)
        emb2_g = self.pool1(b2).view(batch, self.dim)
        emb2_p = self.pool2(b2).view(batch, self.dim, 2)
        embs = [emb1_g, emb2_g, emb2_p[:, :, 0].contiguous(), emb2_p[:, :, 1].contiguous()]
        embs = [self.bns[i](embs[i]) for i in range(4)]

        if self.num_classes is not None:
            logits = [self.logits[i](embs[i]) for i in range(4)]
            return embs, logits

        embs = [emb / torch.norm(emb) for emb in embs]
        return torch.cat(embs, dim=-1)
