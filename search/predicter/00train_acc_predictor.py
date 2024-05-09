import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
import sys
import random
import torch.nn as nn
from putils import AccPredictorTrainer,get_data_loader,AccuracyPredictor
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载数据集
train_loader,valid_loader,base_acc = get_data_loader(type='clean')

# 加载网络
predictor = AccuracyPredictor(hidden_size=200, n_layers=6, device=device)

# 加载训练器
loss_func = nn.SmoothL1Loss()
test_func = nn.MSELoss()
optimizer = optim.Adam(predictor.parameters(), lr=2.0e-4, weight_decay=1.e-5)


trainer = AccPredictorTrainer(model=predictor, criterion=loss_func, test_criterion=test_func, optimizer=optimizer,
                              lr_scheduler=None, train_loader=train_loader, test_loader=valid_loader)

start_epoch = 0
for epoch in range(start_epoch, 300):
    trainer.train_epoch(cur_epoch=epoch, rank=device)
    trainer.test_epoch(cur_epoch=epoch, rank=device)