import time
import torch
import torch.nn as nn
from hanet.supernet import SuperHanet
import core.config as config
from datasets.cifar10_224 import Cifar224_data
from runner.scheduler import adjust_learning_rate_per_batch
from core.config import cfg
from logger import meter
from logger.utils import list_mean
from logger.logging import get_logger
from logger.logging import make_path
from logger import checkpoint

config.load_configs()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建工作路径和logger
exp_path = make_path(cfg.SPACE.NAME)
logger = get_logger(exp_path + '/logger.log')


def main():
    # 超网
    snet = SuperHanet(n_classes=cfg.LOADER.NUM_CLASSES)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.SGD(snet.parameters(), lr=cfg.OPTIM.BASE_LR,
                                momentum=cfg.OPTIM.MOMENTUM, weight_decay=cfg.OPTIM.WEIGHT_DECAY
                                )

    # 数据加载
    train_loader, test_loader = Cifar224_data(root=cfg.LOADER.DATA_PATH, train_batch_size=cfg.LOADER.BATCH_SIZE,
                                                      test_batch_size=cfg.LOADER.BATCH_SIZE)

    # 初始化训练器
    camnas_trainer = CamNASTraniner(
        model=snet,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=False,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    # 断点重启
    # start_epoch = bignas_trainer.loading() if cfg.SEARCH.AUTO_RESUME else 0
    start_epoch = camnas_trainer.loading() if cfg.RESUME else 0

    for cur_epoch in range(start_epoch, cfg.OPTIM.WARMUP_EPOCH + cfg.OPTIM.MAX_EPOCH):
        camnas_trainer.train_epoch(cur_epoch)
        camnas_trainer.validate(cur_epoch)


class CamNASTraniner():
    # 训练超网
    def __init__(self, model, criterion, optimizer, lr_scheduler, train_loader, test_loader):
        super(CamNASTraniner, self).__init__()
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sandwich_sample_num = cfg.SUPERNET_CFG.SANDWICH_NUM
        self.best_acc = 0

    def train_epoch(self, cur_epoch):
        self.model.train()

        # logger的输出频次
        print_freq = len(self.train_loader) // 10
        start_time = time.time()

        top1_avg, top5_avg, loss_avg = meter.AverageMeter(), meter.AverageMeter(), meter.AverageMeter()

        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 每次 iter 自适应调整学习率 lr
            cur_lr = adjust_learning_rate_per_batch(
                epoch=cur_epoch,
                n_iter=len(self.train_loader),
                iter=cur_iter,
                warmup=(cur_epoch < cfg.OPTIM.WARMUP_EPOCH)
            )

            # Rule: 常量结尾学习率
            cur_lr = max(cur_lr, 0.05 * cfg.OPTIM.BASE_LR)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = cur_lr

            # 三明治法则
            self.optimizer.zero_grad()
            for arch_id in range(0, self.sandwich_sample_num):
                if arch_id == self.sandwich_sample_num - 1:
                    self.model.set_conv_net()
                elif arch_id == self.sandwich_sample_num - 2:
                    self.model.set_att_net()
                elif arch_id == self.sandwich_sample_num - 3:
                    self.model.set_mlp_net()
                else:
                    self.model.set_random_net()

                # print(next(self.model.parameters()).device)

                preds = self.model(inputs)
                loss = self.criterion(preds, labels)
                loss.backward()

            # 只解决梯度爆炸问题，不解决梯度消失问题
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()

            top1_acc, top5_acc = meter.topk_acc(preds, labels, [1, 5])
            top1_acc, top5_acc, loss = top1_acc.item(), top5_acc.item(), loss.item()
            top1_avg.update(top1_acc, labels.size(0)), top5_avg.update(top5_acc, labels.size(0)), loss_avg.update(
                loss, labels.size(0))

            # 每1/freq输出一次精度
            # 用conv_net的精度来评估
            if cur_iter % print_freq == 0 and cur_iter != 0:
                current_time = time.time()
                cost_time = current_time - start_time
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'lr {:.4f}\t'
                    'Loss {:.4f}\t'
                    'Top1_acc {:.2f}%\t\t'
                    'Time {:.2f}s'.format(
                        cur_epoch, cur_iter, len(self.train_loader), cur_lr,
                        float(loss_avg.avg), float(top1_avg.avg), cost_time
                    )
                )
                start_time = current_time

    # 测试单个网络 cur_epoch是用来写在logger里的，待加入
    def test_epoch(self):
        self.model.eval()

        top1_avg, top5_avg, loss_avg = meter.AverageMeter(), meter.AverageMeter(), meter.AverageMeter()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            top1_acc, top5_acc = meter.topk_acc(preds, labels, [1, 5])
            top1_acc, top5_acc, loss = top1_acc.item(), top5_acc.item(), loss.item()
            top1_avg.update(top1_acc, labels.size(0)), top5_avg.update(top5_acc, labels.size(0)), loss_avg.update(
                loss, labels.size(0))

        return top1_avg.avg, top5_avg.avg, loss_avg.avg

    # 测试多个网络，求出平均acc
    def validate(self, cur_epoch):
        start_time = time.time()
        subnet_to_eval = {"convnet", "attnet", "mlpnet", "random", "random"}
        # subnet_to_eval = {"convnet"}
        top1_list, top5_list, loss_list = [], [], []

        with  torch.no_grad():
            for net_id in subnet_to_eval:
                if net_id == "convnet":
                    self.model.set_conv_net()
                elif net_id == "attnet":
                    self.model.set_att_net()
                elif net_id == "mlpnet":
                    self.model.set_mlp_net()
                else:
                    self.model.set_random_net()

                top1_acc, top5_acc, loss = self.test_epoch()
                top1_list.append(top1_acc), top5_list.append(top5_acc), loss_list.append(loss)

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(list_mean(loss_list), float(list_mean(top1_list)), (current_time - start_time)))

        if self.best_acc < list_mean(top1_list):
            self.best_acc = list_mean(top1_list)
            # 保存
            self.saving(epoch=cur_epoch, best_acc=list_mean(top1_list), ckpt_dir=exp_path + 'best_ckpt.pth')
        else:
            self.saving(epoch=cur_epoch, best_acc=list_mean(top1_list), ckpt_dir=exp_path + 'cur_ckpt.pth')


    def saving(self, epoch, best_acc, ckpt_dir=None):
        ms = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        state = {
            'state_dict': ms.state_dict(),
            'best_acc': best_acc,
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch + 1
        }
        torch.save(state, ckpt_dir)

    def loading(self):
        ckpt = torch.load(cfg.CKPT_PTH)
        logger.info("Resume checkpoint from epoch: {}".format(ckpt['epoch'] ))
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

        return ckpt['epoch']


if __name__ == "__main__":
    main()
