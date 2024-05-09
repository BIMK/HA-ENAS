import timm
import torch

def mixer_b16_224(pretrained):
    mixer_b_16 = timm.create_model('mixer_b16_224', pretrained=True,num_classes=10)
    if pretrained:
        checkpoint_path = 'checkpoints/mixer_b16_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        mixer_b_16.load_state_dict(checkpoint)
    return mixer_b_16

def resmlp_24_224(pretrained):
    resmlp_12_224 = timm.create_model('resmlp_24_224', pretrained=True,num_classes=10)
    if pretrained:
        checkpoint_path = 'checkpoints/resmlp_24_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        resmlp_12_224.load_state_dict(checkpoint)
    return resmlp_12_224

def vit_s16_224(pretrained=False):
    vit_s16_224 = timm.create_model('vit_small_patch16_224', pretrained=True,num_classes=10)
    if pretrained:
        checkpoint_path = 'checkpoints/vit_s16_cifar10.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        vit_s16_224.load_state_dict(checkpoint)
    return vit_s16_224