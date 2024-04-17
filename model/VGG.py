import torch
import torch.nn as nn
from typing import cast


class VGG(nn.Module):

    def __init__(self, num_classes: int = 3, dropout: float = 0.5) -> None:
        super(VGG, self).__init__()
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                        nn.ReLU(True), nn.Dropout(p=dropout),
                                        nn.Linear(4096, 4096), nn.ReLU(True),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(4096, num_classes))
        # ignore init_weights

    def make_layers(self) -> nn.Module:

        A = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        B = [
            64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512,
            "M"
        ]
        D = [
            64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M",
            512, 512, 512, "M"
        ]
        E = [
            64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512,
            512, "M", 512, 512, 512, 512, "M"
        ]

        layers = list()
        in_channels = 3
        self.type = B

        for v in B:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                v = cast(int, v)
                layers.append(
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=3))
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N x 3 x 224 x 224
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
