import torch.nn as nn


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        # Input size = 32
        self.prep = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.rb1 = self.make_resblock(128)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.rb2 = self.make_resblock(512)

        self.pool1 = nn.Sequential(nn.MaxPool2d(4, 4))

        self.fc = nn.Sequential(nn.Linear(512, 10, bias=False))

    def make_resblock(self, kernels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=kernels,
                out_channels=kernels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(kernels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=kernels,
                out_channels=kernels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(kernels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.conv1(x)
        r1 = self.rb1(x)
        x = x + r1
        x = self.conv2(x)
        x = self.conv3(x)
        r2 = self.rb2(x)
        x = x + r2
        x = self.pool1(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        # x = nn.Softmax(x) # Note that we don't explicitely apply softmax as CrossEntropy by default calculates Softmax and then calculates loss
        return x
