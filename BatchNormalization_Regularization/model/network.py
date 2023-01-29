import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from enum import Enum


class Empty:
    def __call__(pVal):
        return pVal


class Normalization(Enum):
    BatchNormalization = 1
    LayerNormalization = 2
    GroupNormalization = 3


class Network(nn.Module):
    def __init__(self, pDrouput=0.02, pNormalization=Normalization.BatchNormalization):
        super(Network, self).__init__()

        # Normalizations
        self.norm1 = Empty()
        self.norm2 = Empty()
        self.norm3 = Empty()
        self.norm4 = Empty()
        self.norm5 = Empty()
        self.norm6 = Empty()
        self.norm7 = Empty()
        self.norm8 = Empty()

        self.SetNormalization(pNormalization)

        # Activation and Dropout (can be commenly used since no learnable parameters)

        self.relu_dropout = nn.Sequential(nn.ReLU(), nn.Dropout(pDrouput))

        # ---------Input BLOCK 2---------

        # Input size = 28
        self.convblock1 = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=(3, 3), padding=0, bias=False
        )  # output_size = 26  RF = 3

        # -------------------------------

        # ------CONVOLUTION BLOCK 1------

        self.convblock2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=(3, 3), padding=0, bias=False
        )  # output_size = 24  RF = 5

        self.convblock3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False
        )  # output_size = 22  RF = 7

        self.convblock4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False
        )  # output_size = 20  RF = 9

        # -------------------------------

        # -------TRANSITION BLOCK 1------

        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 10   RF = 10

        self.convblock5 = nn.Conv2d(
            in_channels=16, out_channels=4, kernel_size=(1, 1), padding=0, bias=False
        )  # output_size = 10  RF = 10

        # -------------------------------

        # ------CONVOLUTION BLOCK 2------

        self.convblock6 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=(3, 3), padding=0, bias=False
        )  # output_size = 8  RF = 14

        self.convblock7 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False
        )  # output_size = 6  RF = 18

        self.convblock8 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False
        )  # output_size = 4  RF = 22

        # -------------------------------

        # ---------OUTPUT BLOCK----------

        self.gap = nn.AvgPool2d(kernel_size=4)  # output_size = 1 RF = 28

        self.convblock9 = nn.Conv2d(
            in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False
        )  # output_size = 1  RF = 28

        # -------------------------------

    def SetNormalization(self, pNormalization):
        if pNormalization == Normalization.BatchNormalization:
            self.norm1 = nn.BatchNorm2d(4)
            self.norm2 = nn.BatchNorm2d(8)
            self.norm3 = nn.BatchNorm2d(16)
            self.norm4 = nn.BatchNorm2d(16)
            self.norm5 = nn.BatchNorm2d(4)
            self.norm6 = nn.BatchNorm2d(8)
            self.norm7 = nn.BatchNorm2d(16)
            self.norm8 = nn.BatchNorm2d(16)

        elif pNormalization == Normalization.LayerNormalization:
            self.norm1 = nn.GroupNorm(1, 4) # nn.LayerNorm([4, 26, 26])
            self.norm2 = nn.GroupNorm(1, 8) # nn.LayerNorm([8, 24, 24])
            self.norm3 = nn.GroupNorm(1, 16) # nn.LayerNorm([16, 22, 22])
            self.norm4 = nn.GroupNorm(1, 16) # nn.LayerNorm([16, 20, 20])
            self.norm5 = nn.GroupNorm(1, 4) # nn.LayerNorm([4, 10, 10])
            self.norm6 = nn.GroupNorm(1, 8) # nn.LayerNorm([8, 8, 8])
            self.norm7 = nn.GroupNorm(1, 16) # nn.LayerNorm([16, 6, 6])
            self.norm8 = nn.GroupNorm(1, 16) # nn.LayerNorm([16, 4, 4])

        elif pNormalization == Normalization.GroupNormalization:
            self.norm1 = nn.GroupNorm(2, 4)
            self.norm2 = nn.GroupNorm(2, 8)
            self.norm3 = nn.GroupNorm(2, 16)
            self.norm4 = nn.GroupNorm(2, 16)
            self.norm5 = nn.GroupNorm(2, 4)
            self.norm6 = nn.GroupNorm(2, 8)
            self.norm7 = nn.GroupNorm(2, 16)
            self.norm8 = nn.GroupNorm(2, 16)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.norm1(x)
        x = self.relu_dropout(x)

        x = self.convblock2(x)
        x = self.norm2(x)
        x = self.relu_dropout(x)

        x = self.convblock3(x)
        x = self.norm3(x)
        x = self.relu_dropout(x)

        x = self.convblock4(x)
        x = self.norm4(x)
        x = self.relu_dropout(x)

        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.norm5(x)
        x = self.relu_dropout(x)

        x = self.convblock6(x)
        x = self.norm6(x)
        x = self.relu_dropout(x)

        x = self.convblock7(x)
        x = self.norm7(x)
        x = self.relu_dropout(x)

        x = self.convblock8(x)
        x = self.norm8(x)
        x = self.relu_dropout(x)

        x = self.gap(x)
        x = self.convblock9(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


def PrintSummary(pModel, PInpSize=(1, 28, 28)):
    print(torchsummary.summary(pModel, input_size=PInpSize))
