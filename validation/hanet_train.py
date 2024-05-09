'''Train CIFAR10 with PyTorch.'''
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from hanet.hanet import HA_Net_A1
from utils import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

start_time = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
trainloader, testloader = Cifar224_data(train_batch_size=16, test_batch_size=32)

fname = 'HA_Net_A1'

# Model
print('==> Building model..' + fname + '!0n 224')

net = HA_Net_A1().to(device)

print(net)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=0.1,
    total_steps=300 * len(trainloader),
    three_phase=True)

path = make_path(fname)
logger = get_logger(path + '/logger.log')

print('==> Building model..' + fname + '!0n 224')

for epoch in range(start_epoch, start_epoch + 300):
    retrain(epoch, net, trainloader, optimizer, scheduler, criterion, logger=logger, device=device)
    best_acc = test(epoch, net, testloader, criterion, path, best_acc, logger, device)

current_time = time.time()

print('Running time: %s hours' % ((current_time - start_time) / 3600))

