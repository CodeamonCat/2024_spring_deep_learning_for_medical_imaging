import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional


class BasicConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Inception, self).__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        # 1x1 conv branch
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1))

        # 1x1 conv -> 5x5 conv branchs
        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # kernel_size=3 instead of kernel_size=5 is a known bug (from PyTorch)
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
            # one more time
            conv_block(ch5x5, ch5x5, kernel_size=3, padding=1))

        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            # ceil_mode=True from PyTorch
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),
            conv_block(in_channels, pool_proj, kernel_size=1))

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        ouputs = self._forward(x)
        return torch.cat(ouputs, 1)


class GoogLeNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 3,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
    ) -> None:
        super(GoogLeNet, self).__init__()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # ceil_mode=True from PyTorch
        self.maxpool_3 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        # kernel_size = 3, and without padding=1, ceil_mode=True from PyTorch
        self.maxpool_4 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # without using nn.Dropout from PyTorch
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)
        # pass

    def forward(self, x):

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################

        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception_3a(x)
        # N x 256 x 28 x 28
        x = self.inception_3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool_3(x)
        # N x 480 x 14 x 14
        x = self.inception_4a(x)
        # N x 512 x 14 x 14
        x = self.inception_4b(x)
        # N x 512 x 14 x 14
        x = self.inception_4c(x)
        # N x 512 x 14 x 14
        x = self.inception_4d(x)
        # N x 528 x 14 x 14
        x = self.inception_4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool_4(x)
        # N x 832 x 7 x 7
        x = self.inception_5a(x)
        # N x 832 x 7 x 7
        x = self.inception_5b(x)
        # N x 1024 x 7 x 7
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        # without using nn.Dropout from PyTorch
        # x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x
        # pass
