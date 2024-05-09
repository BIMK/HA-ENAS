import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import torch
import torchvision
import torchvision.transforms as transforms
import logging
import time
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def Cifar224_data(root='../dataset/CIFAR10', train_batch_size=64, test_batch_size=100):
    # Data
    print('==> Preparing Cifar224 data..')
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


# 用来读取攻击后的数据集
def Cifar224_atk_data(fn='res50_fgsm', test_batch_size=16):
    path = 'dataset/atk_dataset/' + fn
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = ImageFolder(path, transform=transform_test)
    test_dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=True, num_workers=4)

    return test_dataloader
