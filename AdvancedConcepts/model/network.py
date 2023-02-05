import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class Network(nn.Module):
    def __init__(self, pDropout=0.0):
        super(Network, self).__init__()
        # Input Block
        # Input size = 32
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(pDropout),
        )  # output_size = 32  RF = 3
        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(pDropout),
        )  # output_size = 32  RF = 5

        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(pDropout),
        )  # output_size = 32  RF = 7

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 16   RF = 8

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(pDropout),
        )  # output_size = 16  RF = 12
        self.convblock5 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                padding=2,
                bias=False,
                dilation=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(pDropout),
        )  # output_size = 16  RF = 20

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 8   RF = 22

        # CONVOLUTION BLOCK 3
        self.convblock6 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(pDropout),
        )  # output_size = 8  RF = 30
        self.convblock7 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(pDropout),
        )  # output_size = 8  RF = 38
        self.convblock8 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
                groups=64,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(pDropout),
        )  # output_size = 8  RF = 46

        # TRANSITION BLOCK 4
        self.pool3 = nn.MaxPool2d(2, 2)  # output_size = 4   RF = 50

        self.convblock9 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(pDropout),
        )  # output_size = 4  RF = 66
        self.convblock10 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(pDropout),
        )  # output_size = 4  RF = 82
        self.convblock11 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(pDropout),
        )  # output_size = 4  RF = 98

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )  # output_size = 1 RF = 110

        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0),
        )  # output_size = 1  RF = 110

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.pool3(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.convblock11(x)
        x = self.gap(x)
        x = self.convblock12(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


def PrintSummary(pModel, PInpSize):
    print(torchsummary.summary(pModel, input_size=PInpSize))
