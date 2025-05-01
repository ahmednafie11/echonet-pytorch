import torch
import torchvision.models.video as video_models
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
class R2Plus1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 512)
        self.classifier = nn.Linear(512, 2)  # Normal vs. abnormal
    def forward(self, x):
        ef = self.model(x)
        anomaly = self.classifier(ef)
        return ef, anomaly
    def r2plus1d_18(pretrained=False, **kwargs):
        """Constructs a R(2+1)D-18 model."""
        model = video_models.r2plus1d_18(pretrained=pretrained, **kwargs)
        return model
