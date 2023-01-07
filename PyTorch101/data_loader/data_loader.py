import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


class CustomMNISTDataset(Dataset):
    def __init__(self, pDataFolderPath="./data/", pIsTrainData=False):
        self.InputData1 = torchvision.datasets.MNIST(
            root=pDataFolderPath,
            train=pIsTrainData,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        self.DataLen = len(self.InputData1)

        self.InputData2 = torch.floor(10 * torch.rand(self.DataLen))

    def __getitem__(self, index):
        return {
            "input_mnist_image": self.InputData1[index][0],
            "input_number": self.InputData2[index],
            "mnist_gt": self.InputData1[index][1],
            "sum_gt": self.InputData2[index] + self.InputData1[index][1],
        }

    def __len__(self):
        return self.DataLen


def GetData(pDataFolderPath="./data/", batch_size=32):
    train_dataset = CustomMNISTDataset(
        pDataFolderPath=pDataFolderPath, pIsTrainData=True
    )
    valid_dataset = CustomMNISTDataset(
        pDataFolderPath=pDataFolderPath, pIsTrainData=False
    )

    SEED = 1
    cuda = torch.cuda.is_available()  # CUDA?
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader
