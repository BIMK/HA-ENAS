import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import torch
import torchvision
import torchvision.transforms as transforms
import logging
import time
from autoaugment import CIFAR10Policy
from torchvision.datasets import ImageFolder
import torchattacks
from torch.utils.data import DataLoader


# 使用方法
# 1、首先调用make_path()函数，生成一个工作路径，在Job/目录下，模型和log文件都会保存在文件目录下
# 2、train()和test()函数只需将相应参数传入即可
# 3、data_plt()会将log里的数据读出并绘制图像


def Cifar224_data(root='/home/data/CIFAR10', train_batch_size=64, test_batch_size=100):
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


# 初始化保存文件路径的函数,可以输入想要保存的文件名，否则创建默认文件名
def make_path(fn=''):
    dt = datetime.now()
    if fn == '':
        dt = str(dt).split('.')[0][2:]
        filename = 'job_' + dt
    else:
        dt = str(dt).split('.')[0][2:-9]
        filename = fn + '_' + dt

    path = './Job/' + filename
    os.makedirs(path, exist_ok=True)
    return path + '/'

# 用来训练的函数
def train(epoch, net, trainloader, optimizer, criterion, logger, device='cpu'):
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    print_freq = len(trainloader) // 10
    start_time = time.time()

    for batch, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 将loss和acc写入log文件
        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'lr {:.4f}\t'
                'Loss {:.4f}\t'
                'Accuracy {:2.2%}\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch, len(trainloader), lr,
                    float(train_loss / (batch + 1)), float(correct / total), cost_time
                )
            )
            start_time = current_time


# 用来测试的函数,会返回一个最佳准确率，记得接收
def test(epoch, net, testloader, criterion, path, best_acc, logger, device='cpu'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    current_time = time.time()
    logger.info(
        'Test Loss {:.4f}\tAccuracy {:2.2%}\t\tTime {:.2f}s\n'
            .format(float(test_loss / (batch + 1)), float(correct / total), (current_time - start_time))
    )

    # logger(path, f'Test: loss:{test_loss / (idx + 1)},acc:{correct / total}\n')

    # 保存网络
    acc = correct / total
    if acc > best_acc:
        print('Saving..')
        print(list(net.state_dict().keys())[0][:6])
        if list(net.state_dict().keys())[0][:6] == 'module':
            net_state_dict = net.module.state_dict()
        else:
            net_state_dict = net.state_dict()

        state = {
            'state_dict': net_state_dict,
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, path + 'ckpt.pth')
        best_acc = acc
    return best_acc


# 测试函数，用来测试网络准确率
def test_acc(model, testloader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        with tqdm(testloader, unit="b", ncols=100, colour='red') as tepoch:
            for idx, (inputs, targets) in enumerate(tepoch):
                tepoch.set_description("Test ")
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                tepoch.set_postfix(acc='{:2.2%}'.format(correct / total))
    print('------------------')
    print('acc = {:2.2%}'.format(correct / total))
    print('------------------')
    return  correct / total


# 用来从文件中读取loss和acc信息，并绘制图像的函数
def data_plt(path):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    lr = []
    with open(path + 'log.txt', 'r') as f:
        while True:
            # 读取一行内容
            text = f.readline()
            data = text.split(':')
            if data[0] == 'Epoch':
                lr.append(float(data[-1].split(' ')[-1]))
            elif data[0] == 'Train':
                train_loss.append(float(data[2].split(',')[0]))
                train_acc.append(float(data[-1]))
            elif data[0] == 'Test':
                test_loss.append(float(data[2].split(',')[0]))
                test_acc.append(float(data[-1]))
            # 判断是否读到内容
            if not text:
                break
    aplt(train_loss, test_loss, train_acc, test_acc, lr, path)


# 绘图函数
def aplt(a1, a2, a3, a4, a5, path):
    fig = plt.figure(num=1, figsize=(18, 6), dpi=150)  # 开启一个窗口，同时设置大小，分辨率
    ax1 = fig.add_subplot(1, 3, 1)  # 通过fig添加子图，参数：行数，列数，第几个。
    ax2 = fig.add_subplot(1, 3, 2)  # 通过fig添加子图，参数：行数，列数，第几个。
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.set_title('Loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')

    ax2.set_title('acc')
    ax2.set_ylabel('acc')
    ax2.set_xlabel('epoch')
    ax2.set_ylim(0, 1)

    ax3.set_title('lr')
    ax3.set_ylabel('lr')
    ax3.set_xlabel('epoch')

    x = [i for i in range(1, len(a1) + 1)]
    # 绘制图一
    ax1.plot(x, a1, label='train_loss')
    ax1.plot(x, a2, label='test_loss')
    ax1.legend(loc='upper left')

    # 绘制图2
    ax2.plot(x, a3, label='train_acc')
    ax2.plot(x, a4, label='test_acc')
    ax2.legend(loc='upper left')

    # 绘制图3
    ax3.plot(x, a5)

    plt.savefig(path + "plot.jpg", dpi=150)
    plt.show()


# 用来微调模型的函数,和train的主要区别在于把lr_scheduler放入数据里面进行更新了
# def retrain(epoch, net, trainloader, optimizer, scheduler, criterion, path, logger, device='cpu'):
#     lr = optimizer.state_dict()['param_groups'][0]['lr']
#     print(f'\nEpoch:{epoch + 1}   lr = {lr}')
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     with tqdm(trainloader, unit="b", ncols=100, colour='blue') as tepoch:
#         for idx, (inputs, targets) in enumerate(tepoch):
#             tepoch.set_description("Train")
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             # 加在这里了
#             scheduler.step()
#
#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#             tepoch.set_postfix(acc='{:2.2%}'.format(correct / total), loss='{:.3}'.format(train_loss / (idx + 1)))
#     # 将loss和acc写入log文件
#     logger(path, f'Epoch:{epoch + 1}   lr = {lr}\n')
#     logger(path, f'Train: loss:{train_loss / (idx + 1)},acc:{correct / total}\n')

# 用来微调模型的函数,和train的主要区别在于把lr_scheduler放入数据里面进行更新了
def retrain(epoch, net, trainloader, optimizer, scheduler, criterion, logger, device='cpu'):
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    print_freq = len(trainloader) // 10
    start_time = time.time()

    for batch, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # 加了梯度裁剪
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
        # 加在这里了
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 将loss和acc写入log文件
        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'lr {:.4f}\t'
                'Loss {:.4f}\t'
                'Accuracy {:2.2%}\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch, len(trainloader), lr,
                    float(train_loss / (batch + 1)), float(correct / total), cost_time
                )
            )
            start_time = current_time


# def get_logger(file_path):
#     logger = logging.getLogger('gal')
#     log_format = '%(asctime)s | %(message)s'
#     formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
#     file_handler = logging.FileHandler(file_path)
#     file_handler.setFormatter(formatter)
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(formatter)
#
#     logger.addHandler(file_handler)
#     logger.addHandler(stream_handler)
#     logger.setLevel(logging.INFO)
#
#     return logger


def get_logger(file_path,log_name='gal'):
    logger = logging.getLogger(log_name)
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger



def logger_plt(path):
    with open(path + 'logger.log', 'r') as f:
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        lr = []
        line = 1
        train_line = 10
        test_line = 11
        while True:
            # 读取一行内容
            text = f.readline()
            if line % train_line == 0:
                data = text.split(':')[-1].split('\t')
                lr.append(float(data[1].split(' ')[-1]))
                train_loss.append(float(data[2].split(' ')[-1]))
                train_acc.append(float(data[3].split(' ')[-1][:-1]) / 100)

            if line % test_line == 0:
                data = text.split(':')[-1].split('\t')
                test_loss.append(float(data[0].split(' ')[-1]))
                test_acc.append(float(data[1].split(' ')[-1][:-1]) / 100)

            if text == '\n':
                line = 0

            # 判断是否读到内容
            line += 1
            if not text:
                break
    aplt(train_loss, test_loss, train_acc, test_acc, lr, path)


def flops_test(model, device, input_size=(1, 3, 224, 224)):
    from thop import profile
    from thop import clever_format
    input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)  #
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)


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


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].contiguous().view(-1).float().sum() for k in ks]
    return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]


def topk_acc(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# 同时返回clean_acc和fgsm_acc
def fgsm_acc(net, testloader, device):
    start_time = time.time()

    # 定义攻击
    atk = torchattacks.FGSM(net, 8 / 255)

    clean_top1_avg = AverageMeter()
    fgsm_top1_avg = AverageMeter()
    net.eval()
    for cur_iter, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            preds = net(inputs)
        top1_acc, _ = topk_acc(preds, labels, [1, 5])
        top1_acc = top1_acc.item()
        clean_top1_avg.update(top1_acc, labels.size(0))

        # 攻击部分
        a_imgs = atk(inputs, labels)
        with torch.no_grad():
            preds = net(a_imgs)
        top_acc, _ = topk_acc(preds, labels, [1, 5])
        top_acc = top_acc.item()
        fgsm_top1_avg.update(top_acc)

    end_time = time.time()
    print('clean_acc = {:.2f}% \tfgsm_acc = {:.2f}% \t time:{:.2f}s'.format(clean_top1_avg.avg,
                                                                            fgsm_top1_avg.avg,
                                                                            end_time - start_time))
    return clean_top1_avg.avg, fgsm_top1_avg.avg


# 用来读取攻击后的数据集
def Cifar224_atk_data(fn='res101_fgsm', test_batch_size=64):
    path = '/home/E21201102/my_dataset/cifar10/' + fn
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = ImageFolder(path, transform=transform_test)
    test_dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=True, num_workers=4)

    return test_dataloader


# 测试在两个攻击过的数据集上的平均rob_acc
def blackbox_rob_acc(net, atk_loader, device):
    start_time = time.time()
    net.eval()

    top1_avg = AverageMeter()
    for data_loader in atk_loader:
        with torch.no_grad():
            for cur_iter, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                preds = net(inputs)
                top1_acc, _ = topk_acc(preds, labels, [1, 5])
                top1_acc = top1_acc.item()
                top1_avg.update(top1_acc, labels.size(0))

    end_time = time.time()
    print('black_atk_acc = {:.2f}% \t\t time:{:.2f}s'.format(top1_avg.avg, end_time - start_time))
    print()
    return top1_avg.avg


# 两个数据分开写
def blackbox_rob_acc2(net, atk_loader, device):
    start_time = time.time()
    net.eval()
    res_avg = AverageMeter()
    vit_avg = AverageMeter()
    with torch.no_grad():
        for cur_iter, (inputs, labels) in enumerate(atk_loader[0]):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = net(inputs)
            top1_acc, _ = topk_acc(preds, labels, [1, 5])
            top1_acc = top1_acc.item()
            res_avg.update(top1_acc, labels.size(0))

    with torch.no_grad():
        for cur_iter, (inputs, labels) in enumerate(atk_loader[1]):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = net(inputs)
            top1_acc, _ = topk_acc(preds, labels, [1, 5])
            top1_acc = top1_acc.item()
            vit_avg.update(top1_acc, labels.size(0))
    end_time = time.time()
    print('res_atk_acc = {:.2f}% \t\tvit_atk_acc = {:.2f}%\t\t black_atk_acc = {:.2f}%\t\t time:{:.2f}s'.format(
        res_avg.avg, vit_avg.avg, (res_avg.avg + vit_avg.avg) / 2, end_time - start_time))
    print()


def atk_acc(net, testloader, device):
    start_time = time.time()

    # 定义攻击

    atk_names = ['FGSM',  'PGD', 'FFGSM', 'MIFGSM']
    atks = [torchattacks.FGSM(net, 8 / 255),
            torchattacks.PGD(net, 8 / 255, alpha=1 / 255, steps=10),
            torchattacks.FFGSM(net, 8 / 255), torchattacks.MIFGSM(net, 8 / 255)]

    # atks = [torchattacks.FGSM(net, 4 / 255), torchattacks.BIM(net, 4 / 255),
    #         torchattacks.PGD(net, 4 / 255, alpha=1 / 255, steps=10),
    #         torchattacks.FFGSM(net, 4 / 255), torchattacks.MIFGSM(net, 4 / 255)]

    net.eval()
    for i, atk in enumerate(atks):
        atk_top1_avg = AverageMeter()
        for cur_iter, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 攻击部分
            a_imgs = atk(inputs, labels)
            with torch.no_grad():
                preds = net(a_imgs)
            top_acc, _ = topk_acc(preds, labels, [1, 5])
            top_acc = top_acc.item()
            atk_top1_avg.update(top_acc)

        end_time = time.time()
        print('{}_acc = {:.2f}% \t time:{:.2f}s'.format(atk_names[i], atk_top1_avg.avg, end_time - start_time))
        start_time = time.time()
