# import sys
#
# sys.path.append('validation/model_zoo/')

import model_zoo
from hanet.hanet import HA_Net_A1
from utils import *

import argparse

parser = argparse.ArgumentParser(description='Random search of Auto-attack')

parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size for data loader')
parser.add_argument('--test_model', default='HA_Net', help='resnet50 | vit | mixer ')

args = parser.parse_args()

start = time.time()

_, test_dataloader = Cifar224_data(test_batch_size=args.batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_list = [model_zoo.vit_b16_224_cifar10(pretrained=True), model_zoo.res50_224_cifar10(pretrained=True),
              model_zoo.mixer_b16_224(pretrained=True)]
model_names = ['vit', 'res50', 'mlp']

if args.test_model == 'HA_Net':
    model_b = HA_Net_A1(pretrained=True).to(device)
elif args.test_model == 'resnet50':
    model_b = model_zoo.res50_224_cifar10(pretrained=True).to(device)
elif args.test_model == 'vit':
    model_b = model_zoo.vit_b16_224_cifar10(pretrained=True).to(device)
elif args.test_model == 'mixer':
    model_b = model_zoo.mixer_b16_224(pretrained=True).to(device)


atk_names = ['FFGSM', 'MI-FGSM', 'PGD']

test_acc(model_b, test_dataloader, device)

for model_a, model_name in zip(model_list, model_names):
    model_a = model_a.to(device)
    print()
    print(f'{model_name}->{args.test_model}')
    atks = [torchattacks.FFGSM(model_a, 8 / 255),
            torchattacks.MIFGSM(model_a, eps=8 / 255), torchattacks.PGD(model_a, eps=8 / 255, alpha=1 / 255, steps=10)]

    for atk, atk_name in zip(atks, atk_names):

        m2m_acc = 0
        total = 0
        model_a.eval()
        model_b.eval()

        for idx, data in enumerate(test_dataloader):
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            atk_imgs = atk(imgs, targets)

            with torch.no_grad():
                atk_output = model_b(atk_imgs)

            atk_idx = atk_output.argmax(1)
            m2m = (atk_idx == targets).sum()

            m2m_acc += m2m
            total += targets.size(0)

        print(atk_name, end=': ')
        print(
            '{:2.2%}'.format(
                m2m_acc / total,
            ))

end = time.time()
print('Running time: %s Seconds' % (end - start))
