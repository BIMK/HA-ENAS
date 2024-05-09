
import os
import torch


def save_checkpoint(model, epoch, checkpoint_dir=None, **kwargs):

    print("saving")
    ms = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    sd = ms.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
    }
    for k, v in kwargs.items():
        print(v)
        print(k)
        # vsd = v.state_dict()
        # checkpoint[k] = vsd
    # Write the checkpoint
    torch.save(checkpoint, checkpoint_dir)

