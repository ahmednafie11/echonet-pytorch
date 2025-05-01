import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50EF(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT, num_classes=1):
        super(ResNet50EF, self).__init__()
        self.model = resnet50(weights=weights)
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def get_resnet_model(weights=ResNet50_Weights.DEFAULT):
    return ResNet50EF(weights=weights)
