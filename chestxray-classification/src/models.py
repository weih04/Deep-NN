from torch import nn


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
    """question b1."""

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
