import sys

import numpy as np

sys.path.append('../')
from predicter import putils
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

clean_model = putils.get_clean_acc_predictor(device)
black_robust_model = putils.get_black_robust_acc_predictor(device)
white_robust_model = putils.get_white_robust_acc_predictor(device)


# 返回预测的clean和black_robust
def get_2obj_value_predictor(pop):
    x = putils.to_one_hot(pop)
    x = torch.tensor(x, dtype=torch.float)
    x = x.to(device)
    clean_model.eval()
    black_robust_model.eval()
    y = clean_model(x)

    z = black_robust_model(x)
    y = y.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    y = y[:, np.newaxis]
    z = z[:, np.newaxis]
    t = np.concatenate((y, z), axis=1)
    return t


# 返回预测的 clean\black_robust\white_robust
def get_3obj_value_predictor(pop):
    x = putils.to_one_hot(pop)
    x = torch.tensor(x, dtype=torch.float)
    x = x.to(device)

    clean_model.eval()
    black_robust_model.eval()
    white_robust_model.eval()
    clean_acc = clean_model(x)
    b_acc = black_robust_model(x)
    w_acc = white_robust_model(x)

    clean_acc = clean_acc.cpu().detach().numpy()
    b_acc = b_acc.cpu().detach().numpy()
    w_acc = w_acc.cpu().detach().numpy()

    clean_acc = clean_acc[:, np.newaxis]
    b_acc = b_acc[:, np.newaxis]
    w_acc = w_acc[:, np.newaxis]
    t = np.concatenate((clean_acc, b_acc, w_acc), axis=1)

    return t


