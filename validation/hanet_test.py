'''Train CIFAR10 with PyTorch.'''
import torch.backends.cudnn as cudnn
from hanet.hanet import HA_Net_A1
from utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
_, testloader = Cifar224_data(test_batch_size=16)

# Model
print('==> Building model..!0n 224')

net = HA_Net_A1(pretrained=True)

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

best_acc = test_acc(net, testloader, device)
