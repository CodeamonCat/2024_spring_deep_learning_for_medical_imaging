import torch
import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self, num_classes: int = 3) -> None:
        super(LeNet, self).__init__()

        # allow channel=3
        in_channels = 3
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1 * in_channels,
                      6 * in_channels,
                      kernel_size=5,
                      padding=2,
                      stride=1), nn.Sigmoid())
        self.conv_2 = nn.Sequential(
            nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
            nn.Sigmoid())

        self.fc_1 = nn.Sequential(
            nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
            nn.Sigmoid())
        self.fc_2 = nn.Sequential(
            nn.Linear(120 * in_channels, 84 * in_channels), nn.Sigmoid())
        self.fc_3 = nn.Linear(84 * in_channels, num_classes)

        self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)
        """
        # original
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2, stride=1), nn.Sigmoid())
        self.conv_2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5),
                                    nn.Sigmoid())

        self.fc_1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.Sigmoid())
        self.fc_2 = nn.Sequential(nn.Linear(120, 84), nn.Sigmoid())
        self.fc_3 = nn.Linear(84, num_classes)

        self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N x 1 x 32 x 32
        x = self.conv_1(x)
        # N x 6 x 28 x 28
        x = self.avgpool2d(x)
        # N x 6 x 14 x 14
        x = self.conv_2(x)
        # N x 16 x 10 x 10
        x = self.avgpool2d(x)
        # N x 16 x 5 x 5
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        # N x 120
        x = self.fc_2(x)
        # N x 84
        x = self.fc_3(x)
        # N x 3 (num_classes)
        return x
