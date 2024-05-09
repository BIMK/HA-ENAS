import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
import sys
import random
import torch.nn as nn


# 给网络编码，转成one-hot
def to_one_hot(x):
    ans = []
    for s in x:
        t = []
        for i in s:
            if i == 0:
                t.extend([1, 0, 0])
            if i == 1:
                t.extend([0, 1, 0])
            if i == 2:
                t.extend([0, 0, 1])
        ans.append(t)
    return ans


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccPredictorTrainer():
    def __init__(self, model, criterion, test_criterion, optimizer, lr_scheduler, train_loader, test_loader):
        super(AccPredictorTrainer, self).__init__()
        self.train_loss = AverageMeter()
        self.test_loss = AverageMeter()
        self.test_criterion = test_criterion
        self.best_loss = 1000.0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model = model
        self.criterion = criterion

    def train_epoch(self, cur_epoch, rank):
        self.train_loss.reset()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(rank), targets.to(rank)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.train_loss.update(loss.item(), inputs.size(0))
            self.optimizer.step()
        return self.train_loss.avg

    @torch.no_grad()
    def test_epoch(self, cur_epoch, rank):
        self.test_loss.reset()
        predicts = []
        reals = []
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.to(rank), targets.to(rank)

            outputs = self.model(inputs)
            predicts.extend([i * 100.0 for i in outputs.tolist()])
            reals.extend([i * 100.0 for i in targets.tolist()])
            loss = torch.sqrt(self.test_criterion(outputs, targets)) * 100.0  # 乘上100就是一个model的acc。
            self.test_loss.update(loss.item(), inputs.size(0))

        test_loss = self.test_loss.avg
        print(test_loss)
        # saving
        is_best = self.best_loss > test_loss
        self.best_loss = min(self.best_loss, test_loss)
        if is_best:
            # self.saving(epoch=cur_epoch, best=True, checkpoint_name='model_acc_predictor')
            net_state_dict = self.model.state_dict()
            state = {
                'state_dict': net_state_dict,
                'loss': loss,
                'epoch': cur_epoch,
            }
            torch.save(state, 'w_ckpt.pth')

            print(self.best_loss)


class RegDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, targets):
        super(RegDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


def build_acc_data_loader(batch_size=128, n_workers=2, path=''):
    x = []
    y = []
    acc_dict = json.load(open(path))
    with tqdm(total=len(acc_dict), desc='Loading data', colour='blue') as t:
        for k, v in acc_dict.items():
            x.append(json.loads(k))
            y.append(v / 100.)  # range: 0 - 1
            t.update()
    base_acc = np.mean(y)
    x = to_one_hot(x)
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y)

    # random shuffle
    shuffle_idx = torch.randperm(len(x))
    X_all = x[shuffle_idx]
    Y_all = y[shuffle_idx]

    # split data
    idx = X_all.size(0) // 5 * 4
    val_idx = X_all.size(0) // 5 * 4
    X_train, Y_train = X_all[:idx], Y_all[:idx]
    X_test, Y_test = X_all[val_idx:], Y_all[val_idx:]
    print('Train Size: %d,' % len(X_train), 'Valid Size: %d' % len(X_test))

    # build data loader
    train_dataset = RegDataset(X_train, Y_train)
    val_dataset = RegDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=n_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=n_workers
    )

    return train_loader, valid_loader, base_acc


def get_data_loader(batch_size=128, type='clean'):
    pth = '/home/E21201102/code_base1/SxkNAS/Job/Dataset_23-03-04/'
    if type == 'clean':
        pth = pth + 'clean.dict'
    elif type == 'brobust':
        pth = pth + 'rboust.dict'
    else:
        pth = '/home/E21201102/code_base1/SxkNAS/Job/Dataset2_23-03-07/white_robust.dict'
    return build_acc_data_loader(batch_size=batch_size, path=pth)


class AccuracyPredictor(nn.Module):
    def __init__(self, hidden_size=400, n_layers=3,
                 checkpoint_path=None, device='cuda:0'):
        super(AccuracyPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        # build layers
        layers = []
        layers.append(nn.Sequential(
            nn.Linear(48, self.hidden_size),
            nn.ReLU(inplace=True),
        ))

        for i in range(self.n_layers):
            layers.append(nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
            ))
        layers.append(nn.Linear(self.hidden_size, 1, bias=False))
        self.layers = nn.Sequential(*layers)
        self.base_acc = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=False)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            self.load_state_dict(checkpoint)
            print('Loaded checkpoint from %s' % checkpoint_path)

        self.layers = self.layers.to(self.device)

    def forward(self, x):
        y = self.layers(x).squeeze()
        return y + self.base_acc


def get_clean_acc_predictor(device):
    predictor = AccuracyPredictor(hidden_size=200, n_layers=6, device=device)
    ckpt = torch.load('checkpoints/predictor_ckpt/clean_ckpt.pth', map_location=device)
    predictor.load_state_dict(ckpt['state_dict'])
    return predictor


def get_black_robust_acc_predictor(device):
    predictor = AccuracyPredictor(hidden_size=200, n_layers=6, device=device)
    ckpt = torch.load('checkpoints/predictor_ckpt/black_ckpt.pth', map_location=device)
    predictor.load_state_dict(ckpt['state_dict'])
    return predictor


def get_white_robust_acc_predictor(device):
    predictor = AccuracyPredictor(hidden_size=200, n_layers=6, device=device)
    ckpt = torch.load('checkpoints/predictor_ckpt/white_ckpt.pth', map_location=device)
    predictor.load_state_dict(ckpt['state_dict'])
    return predictor

