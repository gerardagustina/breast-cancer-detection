"""Model definitions: DenseNet121 classifier and Grad-CAM helpers."""
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class BreastNet(nn.Module):
    """DenseNet121 pretrained on ImageNet, with a single-logit head (malignancy score)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = densenet121(weights=weights)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, 1)  # 1000 classes → 1 logit

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)          # grayscale → 3 channels (ImageNet format)
        return self.net(x).squeeze(1)     # (N, 1) → (N,)


class CamWrapper(nn.Module):
    """Same forward pass as BreastNet, with a non-in-place final ReLU that Grad-CAM can hook."""

    def __init__(self, m):
        super().__init__()
        self.features = m.net.features     # shared modules: same weights, not copies
        self.relu = nn.ReLU(inplace=False)
        self.classifier = m.net.classifier

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        f = self.relu(self.features(x))                          # (N, 1024, 7, 7): Grad-CAM target
        f = nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)   # (N, 1024)
        return self.classifier(f)                                # (N, 1)


def minmax(x):
    """Per-image min-max scaling to [0, 1]. MONAI's default normaliser inverts the map."""
    x = x - x.amin(dim=(2, 3), keepdim=True)
    return x / (x.amax(dim=(2, 3), keepdim=True) + 1e-8)