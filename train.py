import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
import os

import vgg
from config import Config

from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from datetime import datetime


def get_acc(output_label, label):
    """
    计算正确率

    :param output_label: 模型输出
    :param label: label
    :return: 正确率
    """
    total = output_label.shape[0]  # 取总数
    pred_label = torch.argmax(output_label, dim=1)  # 取得估值，dim=1在channel方向取最大值的坐标
    num_correct = (pred_label == label).sum().data.cpu().numpy()  # 与label一致的个数
    return num_correct / total  # 正确率


def dataset_split(dataset, valid_rate=0.2, shuffle=False):
    """
    将数据集划分为train_set和valid_set，返回划分两个数据集用的采样器

    :param dataset: 被划分的数据集
    :param valid_rate: 默认valid划分为20%
    :param shuffle: 是否随机采样
    :return: train_sampler, valid_sampler
    """
    dataset_size = len(dataset)  # 数据集长度
    indices = list(range(dataset_size))  # 采样的下标
    split = int(np.floor(dataset_size * valid_rate))  # 测试集数量
    if shuffle:
        random.shuffle(indices)  # 打乱下标
    train_indices, valid_indices = indices[split:], indices[:split]  # 划分下标
    train_sampler = SubsetRandomSampler(train_indices)  # 用下标生成train采样器
    valid_sampler = SubsetRandomSampler(valid_indices)  # 用下标生成valid采样器
    return train_sampler, valid_sampler


def train(net, num_epochs, optimizer, loss_func, train_loader, valid_loader):
    """
    训练函数

    :param net: 待训练的模型
    :param num_epochs: 训练的次数
    :param optimizer: 优化器，可以是Adam，也可以是SGD等其他的
    :param loss_func: loss函数
    :param train_loader: 训练集
    :param valid_loader: 验证集
    :return:
    """
    device = Config.DEVICE  # GPU或CPU
    net.to(device)  # 加载到GPU或CPU
    loss_func.to(device)  # loss函数加载到GPU或CPU

    for epoch in range(num_epochs):
        begin_time = datetime.now()  # 一个epoch开始的时间

        train_loss = 0  # 每个epoch的训练loss
        train_acc = 0  # 每个epoch的训练准确率
        net.train()  # 训练
        for i, (train_image, train_label) in enumerate(train_loader):
            train_image = Variable(train_image.to(device))
            train_label = Variable(train_label.to(device))

            train_out = net(train_image)  # 计算output
            loss = loss_func(train_out, train_label)  # 计算loss
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 步进

            train_loss += loss.data.cpu().numpy()  # 累加loss
            train_acc += get_acc(train_out, train_label)  # 累加正确率
            pass

        valid_loss = 0  # 每个epoch的验证loss
        valid_acc = 0  # 每个epoch的验证准确率
        net.eval()  # 推断
        for i, (valid_image, valid_label) in enumerate(valid_loader):
            valid_image = Variable(valid_image.to(device))
            valid_label = Variable(valid_label.to(device))

            valid_out = net(valid_image)  # 推断计算output
            loss = loss_func(valid_out, valid_label)  # 推断计算loss，不要进行反向传播

            valid_loss += loss.data.cpu().numpy()  # 累加loss
            valid_acc += get_acc(valid_out, valid_label)  # 累加正确率
            pass
        end_time = datetime.now()  # 一个epoch结束的时间

        mm, ss = divmod((end_time - begin_time).seconds, 60)  # 换算分钟，秒
        hh, mm = divmod(mm, 60)  # 换算小时，分钟
        epoch_str = ('Epoch: {:d} | Time: {:02d}:{:02d}:{:02d} | '
                     'Train Loss: {:.4f} | Train Acc: {:.4f} | '
                     'Valid Loss: {:.4f} | Train Acc: {:.4f} | ')
        print(epoch_str.format(epoch, hh, mm, ss,  # 第几个epoch，用时多久
                               train_loss / len(train_loader),  # 训练集loss，累加值/batch个数
                               train_acc / len(train_loader),  # 训练集正确率，累加值/batch个数
                               valid_loss / len(valid_loader),  # 验证集loss，累加值/batch个数
                               valid_acc / len(valid_loader)))  # 验证集正确率，累加值/batch个数
        path = os.path.dirname(os.path.dirname(os.getcwd()))
        path = os.path.join(path, Config.WEIGHT_SAVE_PATH)
        path = os.path.join(path, 'vgg16-' + str(epoch) + '.pkl')
        torch.save(net.state_dict(), path)  # 保存训练权重


if __name__ == '__main__':
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])  # 数据处理归一化

    dataset_cifar10 = CIFAR10(root=Config.DATASETS_ROOT, train=True,
                              transform=data_transform, download=False)  # CIFAR10数据集
    train_sampler, valid_sampler = dataset_split(dataset_cifar10, shuffle=True)  # 采样器

    train_data = DataLoader(
        dataset=dataset_cifar10,
        batch_size=Config.TRAIN_BATCH_SIZE,
        sampler=train_sampler
    )  # train数据。sampler不能和shuffle同时使用

    valid_data = DataLoader(
        dataset=dataset_cifar10,
        batch_size=Config.TRAIN_BATCH_SIZE,
        sampler=valid_sampler
    )  # valid数据。sampler不能和shuffle同时使用

    vgg = vgg.vgg11(10)
    print(vgg)  # 打印看看模型

    # optimizer = torch.optim.Adam(vgg.parameters())
    optimizer = torch.optim.SGD(vgg.parameters(), lr=Config.LEARN_RATE)
    loss_func = nn.CrossEntropyLoss()

    train(vgg, Config.EPOCHS, optimizer, loss_func, train_data, valid_data)

    pass
