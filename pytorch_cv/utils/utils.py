import torch
import numpy as np
import torchsummary
from torch_lr_finder import LRFinder


def Denormalize(image, mean, std, out_type='np_array'):
    """Un-normalize a given image,i.e., tensor or numpy array
    Args:
        image: A 3-D numpy array or 3-D tensor.
            If tensor, it should be in CPU.
        mean: Mean value. It can be a single value or
            a tuple with 3 values (one for each channel).
        std: Standard deviation value. It can be a single value or
            a tuple with 3 values (one for each channel).
        out_type: Out type of the normalized image.
            `np_array` -> then numpy array is returned
            `tensor` ->  then torch tensor is returned.
    """

    if type(image) == torch.Tensor:
        image = np.transpose(image.clone().numpy(), (1, 2, 0))

    normal_image = image * std + mean
    if out_type == 'tensor':
        return torch.Tensor(np.transpose(normal_image, (2, 0, 1)))
    elif out_type == 'np_array':
        return normal_image
    return None


def ToNumpy(tensor):
    """tensor -> (C,H,W) to numpy array -> (H,W,C)"""
    return np.transpose(tensor.clone().numpy(), (1, 2, 0))


def ToTensor(np_array):
    """numpy array -> (H,W,C) to tensor -> (C,H,W)"""
    return torch.Tensor(np.transpose(np_array, (2, 0, 1)))

def PrintSummary(pModel, PInpSize):
    print(torchsummary.summary(pModel, input_size=PInpSize))

def FindLR(pModel, pOptimizer, pCriterion, pTrainLoader, pDevice="cpu"):
    """Find learning rate for using One Cyclic LRFinder
    Args:
        pModel (instace): torch instace of defined model
        pOptimizer (instance): optimizer to be used
        pCriterion (instance): criterion to be used for calculating loss
        pTrainLoader (instance): torch dataloader instace for trainig set
        pDevice (instance): device to be used, cpu or gpu
    """
    lr_finder = LRFinder(pModel, pOptimizer, pCriterion, device=pDevice)
    lr_finder.range_test(pTrainLoader, end_lr=10, num_iter=200, step_mode="exp")
    lr_finder.plot()
    min_loss = min(lr_finder.history['loss'])
    ler_rate = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
    print(f"Min loss ({min_loss}) is at LR: {ler_rate}")
    return ler_rate
