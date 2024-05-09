import torchvision.models as models
import torch

def res18_224_cifar10(pretrained):
    model = models.resnet18(pretrained=True)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, 10)
    if pretrained:
        checkpoint_path = 'checkpoints/res18_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)
    return model

def res34_224_cifar10(pretrained):
    model = models.resnet34(pretrained=True)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, 10)
    if pretrained:
        checkpoint_path = 'checkpoints/res34_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)
    return model

def res50_224_cifar10(pretrained):
    model = models.resnet50(pretrained=True)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, 10)
    if pretrained:
        checkpoint_path = 'checkpoints/res50_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)
    return model

def res101_224_cifar10(pretrained):
    model = models.resnet101(pretrained=False)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, 10)
    if pretrained:
        checkpoint_path = 'checkpoints/res101_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)
    return model


def res152_224_cifar10(pretrained):
    model = models.resnet152(pretrained=False)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, 10)
    if pretrained:
        checkpoint_path = 'checkpoints/res152_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)
    return model


def vgg16_224_cifar10(pretrained):
    model = models.vgg16(pretrained=False)
    fc_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(fc_features, 10)
    if pretrained:
        checkpoint_path = 'checkpoints/vgg16_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)
    return model


def vgg19_224_cifar10(pretrained):
    model = models.vgg19(pretrained=False)
    fc_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(fc_features, 10)
    if pretrained:
        checkpoint_path = 'checkpoints/vgg19_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)
    return model


def wideRes50_224_cifar10(pretrained):
    model = models.wide_resnet50_2(pretrained=False)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, 10)
    if pretrained:
        checkpoint_path = 'checkpoints/wideRes50_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)
    return model

def shuffleNet_224_cifar10(pretrained):
    model = models.shufflenet_v2_x0_5(pretrained=False)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, 10)
    if pretrained:
        checkpoint_path = 'checkpoints/shuffleNet_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)

    return model


def mobilenet_v2_224_cifar10(pretrained):
    model = models.mobilenet_v2(pretrained=False)
    fc_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(fc_features, 10)
    if pretrained:
        checkpoint_path = 'checkpoints/mobilenet_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)
    return model
