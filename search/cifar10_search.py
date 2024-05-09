import numpy as np

from core.config import cfg
import core.config as config
from hanet.supernet import SuperHanet
from runner.my_utils import *
from datasets.cifar10_224 import Cifar224_data
from datasets.cifar10_224 import Cifar224_atk_data
from logger import meter
import time
from logger.logging import get_logger
from logger.logging import make_path
from nsga2 import NDsort, F_distance, F_mating, P_generator, F_EnvironmentSelect
from runner.Surrogate import get_2obj_value_predictor


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.load_configs()

# 数据集加载
_, testloader = Cifar224_data(test_batch_size=16)
res50_atk_cifar10_dataloader, vit16_atk_cifar10_dataloader = Cifar224_atk_data(fn='res50_fgsm'), Cifar224_atk_data(
    fn='vit_fgsm')
atk_loader = [res50_atk_cifar10_dataloader, vit16_atk_cifar10_dataloader]

test_size = 10  # 只使用1/10的测试集数据

# log日志
# 创建工作路径和logger
exp_path = make_path('EA_Search_surrogate')
logger = get_logger(exp_path + '/logger.log')

print('EA_Search_surrogate')
# 超网加载
snet = SuperHanet(n_classes=cfg.LOADER.NUM_CLASSES)
checkpoint = torch.load(cfg.CKPT_PTH)['state_dict']
snet.load_state_dict(checkpoint)
snet.to(device)


# 把一维的编码转换为二维的列表编码
def to_code(ncode):
    ncode = ncode.tolist()
    stages = cfg.SUPERNET_CFG.RATIO
    sum = 0
    code = []
    for i in stages:
        code.append(ncode[sum:sum + i])
        sum += i
    return code


# 返回clean_acc, fgsm_acc
def clean_acc(code):
    code = to_code(code)

    start_time = time.time()
    snet.eval()
    snet.set_net(code)
    snet.get_codings()

    freq = len(testloader) // test_size
    clean_top1_avg = meter.AverageMeter()

    for cur_iter, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            preds = snet(inputs)
        top1_acc, _ = meter.topk_acc(preds, labels, [1, 5])
        top1_acc = top1_acc.item()
        clean_top1_avg.update(top1_acc, labels.size(0))
        if cur_iter == freq:
            break;

    end_time = time.time()
    print('clean_acc = {:.2f}% \t time:{:.2f}s'.format(clean_top1_avg.avg, end_time - start_time))
    return clean_top1_avg.avg


# 测试在两个攻击过的数据集上的平均rob_acc
def blackbox_rob_acc(code):
    code = to_code(code)

    start_time = time.time()
    snet.eval()
    snet.set_net(code)

    freq = len(atk_loader[0]) // test_size
    top1_avg = meter.AverageMeter()
    for data_loader in atk_loader:
        with torch.no_grad():
            for cur_iter, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                preds = snet(inputs)
                top1_acc, _ = meter.topk_acc(preds, labels, [1, 5])
                top1_acc = top1_acc.item()
                top1_avg.update(top1_acc, labels.size(0))

                if cur_iter == freq:
                    break;

    end_time = time.time()
    print('black_atk_acc = {:.2f}% \t\t time:{:.2f}s'.format(top1_avg.avg, end_time - start_time))
    print()
    return top1_avg.avg


# 获取种群中每个个体的目标函数值
def get_object_value(Pop, input):
    st = time.time()
    # 目标函数  第0列 clean_acc  第1列 黑盒攻击 balck_atk_acc
    output = np.zeros((Pop, 2))
    for i, p in enumerate(input):
        output[i][0] = clean_acc(p)
        output[i][1] = blackbox_rob_acc(p)
    ed = time.time()
    print("total_time:", ed - st)
    return output


def OffspringSelect(Population):
    FunctionValue = get_2obj_value_predictor(Population)

    FrontValue = NDsort.NDSort(-FunctionValue, PopSize)[0]
    CrowdDistance = F_distance.F_distance(FunctionValue, FrontValue)

    for gen in range(20):
        MatingPool = F_mating.F_mating(Population, FrontValue, CrowdDistance)

        Offspring = P_generator.P_generator(MatingPool)

        FunctionValue_Offspring = get_2obj_value_predictor(Offspring)

        Population = np.vstack((Population, Offspring))
        FunctionValue = np.vstack((FunctionValue, FunctionValue_Offspring))

        Population, FunctionValue, FrontValue, CrowdDistance, MaxFront = F_EnvironmentSelect.F_EnvironmentSelect(
            Population, FunctionValue, PopSize)

        # 非支配前沿面
        FunctionValueNon = FunctionValue[(FrontValue == 1)[0], :]
        PopulationNon = Population[(FrontValue == 1)[0], :]

    return Population


if __name__ == "__main__":
    Generations = 30
    PopSize = 50
    Population = np.random.randint(low=0, high=3, size=(PopSize, sum(cfg.SUPERNET_CFG.RATIO)), dtype='int')
    FunctionValue = get_object_value(PopSize, Population)

    FrontValue = NDsort.NDSort(-FunctionValue, PopSize)[0]
    CrowdDistance = F_distance.F_distance(FunctionValue, FrontValue)

    since = time.time()

    for Gene in range(Generations):
        MatingPool = F_mating.F_mating(Population, FrontValue, CrowdDistance)

        # Offspring = P_generator.P_generator(MatingPool)
        Offspring = OffspringSelect(MatingPool)

        FunctionValue_Offspring = get_object_value(PopSize, Offspring)

        Population = np.vstack((Population, Offspring))
        FunctionValue = np.vstack((FunctionValue, FunctionValue_Offspring))

        Population, FunctionValue, FrontValue, CrowdDistance, MaxFront = F_EnvironmentSelect.F_EnvironmentSelect(
            Population, FunctionValue, PopSize)

        # 非支配前沿面
        FunctionValueNon = FunctionValue[(FrontValue == 1)[0], :]
        PopulationNon = Population[(FrontValue == 1)[0], :]

        logger.info(f'Gen :{Gene + 1}\t\t{FunctionValueNon[:3]}\n{PopulationNon[:3]}')

    np.set_printoptions(threshold=np.inf)
    logger.info(f'Gen :{Gene + 1}\t\t{FunctionValueNon}\n{PopulationNon}')

    print(time.time() - since)
