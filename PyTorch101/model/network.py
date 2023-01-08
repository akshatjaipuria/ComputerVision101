import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):

        super().__init__()

        self.Conv3_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.Conv3_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.Conv3_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        self.Conv1_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1)

        self.Conv3_4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3)
        self.Conv3_5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.Conv3_6 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        self.Conv1_2 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1)

        self.Fc_1 = nn.Linear(in_features=10 + 10, out_features=32)
        self.Fc_2 = nn.Linear(in_features=32, out_features=19)

    def forward(self, pInput1, pInput2):

        x1 = pInput1  # ------------------------ 28 x 28 x 1

        x1 = self.Conv3_1(x1)  # ------------------------ 28 x 28 x 1  -> 26 x 26 x 8
        x1 = F.relu(x1)

        x1 = self.Conv3_2(x1)  # ------------------------ 26 x 26 x 1  -> 24 x 24 x 16
        x1 = F.relu(x1)

        x1 = self.Conv3_3(x1)  # ------------------------ 24 x 24 x 16 -> 22 x 22 x 32
        x1 = F.relu(x1)

        x1 = F.max_pool2d(x1, kernel_size=2, stride=2)  # 22 x 22 x 32 -> 11 x 11 x 32
        x1 = self.Conv1_1(x1)  # ------------------------ 11 x 11 x 32 -> 11 x 11 x 8
        x1 = F.relu(x1)

        x1 = self.Conv3_4(x1)  # ------------------------ 11 x 11 x 8  -> 9  x 9  x 8
        x1 = F.relu(x1)

        x1 = self.Conv3_5(x1)  # ------------------------ 9  x 9  x 8  -> 7  x 7  x 16
        x1 = F.relu(x1)

        x1 = self.Conv3_6(x1)  # ------------------------ 7  x 7  x 16 -> 5  x 5  x 32
        x1 = F.relu(x1)

        x1 = F.avg_pool2d(x1, kernel_size=5)  # --------- 5  x 5  x 32 -> 1  x 1  x 32
        x1 = self.Conv1_2(x1)  # ------------------------ 1  x 1  x 32 -> 1  x 1  x 10

        x1 = x1.view(-1, 10)
        output1 = x1

        x2 = torch.cat((x1, pInput2), dim=1)

        x2 = self.Fc_1(x2)
        x2 = F.relu(x2)

        x2 = self.Fc_2(x2)
        x2 = F.relu(x2)

        output2 = x2

        return output1, output2
