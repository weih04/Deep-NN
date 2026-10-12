import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def conv_block(in_channels, out_channels, kernel_size, stride=1):
    """Conv (no bias, SAME padding) -> BatchNorm -> ReLU.
    The bias is dropped because BatchNorm's shift makes it redundant.
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class AlexNetLike(nn.Module):
    """AlexNet-like architecture for chest X-ray classification."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16, 5),
            nn.MaxPool2d(2),
            conv_block(16, 32, 5),
            nn.MaxPool2d(2),
            conv_block(32, 64, 3),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # Dropout only on the two hidden fc layers, as the figure specifies.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, images):
        return self.classifier(self.features(images)).squeeze(1)

class VGGLike(nn.Module):
    """VGG-like architecture for chest X-ray classification."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 8, 3),
            conv_block(8, 8, 3),
            nn.MaxPool2d(2),
            conv_block(8, 16, 3),
            conv_block(16, 16, 3),
            nn.MaxPool2d(2),
            conv_block(16, 32, 3),
            conv_block(32, 32, 3),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, images):
        return self.classifier(self.features(images)).squeeze(1)


class ResidualBlock(nn.Module):
    """Two 3×3 convs plus a shortcut; ReLU after the addition.

    The second conv has no ReLU of its own (standard ResNet): the ReLU is
    applied once, after the residual addition.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv_block(in_channels, out_channels, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # A 1×1 conv projection is needed only when the shape changes.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class ResNetLike(nn.Module):
    """ResNet-like architecture for chest X-ray classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16, 3),
            nn.MaxPool2d(2),
            ResidualBlock(16, 16),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64, 1))

    def forward(self, images):
        return self.classifier(self.features(images)).squeeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_flops(model, input_shape=(1, 1, 64, 64)):
    """FLOPs for one forward pass on one image.

    FlopCounterMode counts conv and linear layers only, at 2 FLOPs per
    multiply-add; BatchNorm, ReLU and pooling are not counted.
    """
    model.eval()
    with FlopCounterMode(display=False) as counter:
        model(torch.zeros(input_shape))
    return counter.get_total_flops()
