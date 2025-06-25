import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18FeatureExtractor, self).__init__()
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])  # remove fc layer

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2FeatureExtractor, self).__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)
