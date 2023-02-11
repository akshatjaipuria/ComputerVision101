import torch
from torch.utils.data import DataLoader


def GetDataLoaders(pDatasetClass, pTrainTransforms, pTestTransforms, pBatchSize=128):
    train_dataset = pDatasetClass(
        "./data", train=True, download=True, transform=pTrainTransforms
    )
    test_dataset = pDatasetClass(
        "./data", train=False, download=True, transform=pTestTransforms
    )

    SEED = 1
    cuda = torch.cuda.is_available()  # CUDA?
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    kwargs = {
        "batch_size": pBatchSize,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
    }

    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)

    return train_loader, test_loader
