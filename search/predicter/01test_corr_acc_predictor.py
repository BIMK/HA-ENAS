import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
import sys
import random
import torch.nn as nn
import math
from putils import AccPredictorTrainer, get_data_loader, AccuracyPredictor, AverageMeter, get_clean_acc_predictor, \
    get_black_robust_acc_predictor, get_white_robust_acc_predictor
import torch.optim as optim
from scipy.stats import kendalltau, pearsonr
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(model, testLoader, device):
    test_func = nn.MSELoss()
    losses = AverageMeter()
    predicts = []
    reals = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicts.extend([i * 100.0 for i in outputs.tolist()])
            reals.extend([i * 100.0 for i in targets.tolist()])
            loss = torch.sqrt(test_func(outputs, targets)) * 100.0  # 乘上100就是一个model的acc。
            losses.update(loss.item(), inputs.size(0))


        corr, _ = kendalltau(predicts, reals)
        corr2 = pearsonr(predicts, reals)
        rmse = math.sqrt(sum(list(map(lambda x: (x[0] - x[1]) ** 2, zip(predicts, reals)))) / len(
            predicts))  # 不需要乘100，因为已经是acc了而不是logit。

        # logger.info(
        #     'Test rmse_loss {:.8f}%\t  rmse_acc {:.8f}%\t kendall {:.8f}\t pearson {:.8f}\t Time {:.4f}s\n'
        #         .format(float(losses.avg), rmse, corr, corr2[0], (current_time - start_time))
        # )

        fig = plt.figure()  # 创建一个绘图对象
        ax = fig.add_subplot(111)  # 定义坐标系区域
        ax = sns.regplot(x=predicts, y=reals, scatter_kws={"s": 5})
        # ax.set_title('Per-Channel&Symmetric')
        ax.set_xlabel('predict Accuracy (%)')
        ax.set_ylabel('real Accuracy (%)')
        ax.annotate('kendall', xy=(0.7, 0.2), xytext=(0.7, 0.2), xycoords='figure fraction')
        ax.annotate("= {:.2f}".format(corr), xy=(0.7, 0.2), xytext=(0.79, 0.2), xycoords='figure fraction')
        ax.annotate('pearson', xy=(0.7, 0.2), xytext=(0.7, 0.15), xycoords='figure fraction')
        ax.annotate("= {:.2f}".format(corr2[0]), xy=(0.7, 0.2), xytext=(0.79, 0.15), xycoords='figure fraction')
        plt.show()

        # fig.savefig(cfg.OUT_DIR + '/corr-' + cfg.ARCH + 'all.png', dpi=600, format='png')
        print(str(round(corr, 2)), str(round(corr2[0], 2)))
    return float(losses.avg), corr


def main():
    # 加载网络
    predictor = get_clean_acc_predictor(device)

    # 加载数据集
    # clean - 干净
    # brobust - 黑盒
    # wrobust - 白盒
    train_loader, valid_loader, base_acc = get_data_loader(type='clean')

    test_rmse, corr = test(predictor, valid_loader, device)
    print(test_rmse)


if __name__ == '__main__':
    main()
