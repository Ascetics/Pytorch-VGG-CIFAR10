import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, feature, n_class=1000):
        super(VGG, self).__init__()
        self.feature = feature
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # cifar10输入32x32，下采样后到1x1，拉成向量后直接连全连接层
            nn.ReLU(inplace=True),
            nn.Linear(512, n_class)
        )
        # self.avgpool = nn.AdaptiveAvgPool2d(7)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, n_class)
        # )
        pass

    def forward(self, x):
        x = self.feature(x)  # 下采样
        # x = self.avgpool(x)  # 不再需要平均池化
        x = x.view(x.size(0), -1)  # 拉成向量，接全连接层
        x = self.classifier(x)  # 全连接层
        return x


cfgs = {
    'A': [64, 'M',
          128, 'M',
          256, 256, 'M',
          512, 512, 'M',
          512, 512, 'M'],
    'B': [64, 64, 'M',
          128, 128, 'M',
          256, 256, 'M',
          512, 512, 'M',
          512, 512, 'M'],
    'D': [64, 64, 'M',
          128, 128, 'M',
          256, 256, 256, 'M',
          512, 512, 512, 'M',
          512, 512, 512, 'M'],
    'E': [64, 64, 'M',
          128, 128, 'M',
          256, 256, 256, 256, 'M',
          512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def make_feature(cfg, in_chans=3, batch_norm=False):
    """
    构造VGG层
    :param cfg:
    :param batch_norm:
    :return:
    """
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2))  # 最大池化
        else:
            layers.append(nn.Conv2d(in_chans, v, kernel_size=3, padding=1,
                                    bias=not batch_norm))  # 卷积
            if batch_norm:  # 加BN
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))  # 激活函数
            in_chans = v  # 更新下一个卷积层输入channel
    return nn.Sequential(*layers)


def vgg11(num_class=1000, in_channels=3):
    feature = make_feature(cfgs['A'], in_channels)
    return VGG(feature, num_class)


def vgg11_bn(num_class=1000, in_channels=3):
    feature = make_feature(cfgs['A'], in_channels, batch_norm=True)
    return VGG(feature, num_class)


def vgg13(num_class=1000, in_channels=3):
    feature = make_feature(cfgs['B'], in_channels)
    return VGG(feature, num_class)


def vgg13_bn(num_class=1000, in_channels=3):
    feature = make_feature(cfgs['B'], in_channels, batch_norm=True)
    return VGG(feature, num_class)


def vgg16(num_class=1000, in_channels=3):
    feature = make_feature(cfgs['D'], in_channels)
    return VGG(feature, num_class)


def vgg16_bn(num_class=1000, in_channels=3):
    feature = make_feature(cfgs['D'], in_channels, batch_norm=True)
    return VGG(feature, num_class)


def vgg19(num_class=1000, in_channels=3):
    feature = make_feature(cfgs['E'], in_channels)
    return VGG(feature, num_class)


def vgg19_bn(num_class=1000, in_channels=3):
    feature = make_feature(cfgs['E'], in_channels, batch_norm=True)
    return VGG(feature, num_class)


if __name__ == '__main__':
    vgg = vgg11()
    # vgg = vgg11_bn()
    # vgg = vgg13()
    # vgg = vgg13_bn()
    # vgg = vgg16()
    # vgg = vgg16_bn()
    # vgg = vgg19()
    # vgg = vgg19_bn()
    print(vgg)
    pass
