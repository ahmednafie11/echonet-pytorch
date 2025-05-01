import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class R2Plus1DAnomaly(nn.Module):
    def __init__(self):
        super(R2Plus1DAnomaly, self).__init__()
        self.model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 512)
        self.classifier = nn.Linear(512, 2)  # Normal vs. abnormal
        
    def forward(self, x):
        ef = self.model(x)
        anomaly = self.classifier(ef)
        return ef, anomaly
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50EF(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT, num_classes=1):
        super(ResNet50EF, self).__init__()
        # Load pre-trained ResNet-50
        self.model = resnet50(weights=weights)
        
        # Modify the first convolutional layer to accept 1-channel input (grayscale echo frames)
        self.model.conv1 = nn.Conv2d(
            in_channels=1,  # Echocardiogram frames are typically grayscale
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Replace the fully connected layer for EF regression
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        # Input x: (batch_size, channels=1, height, width)
        return self.model(x)

def get_resnet_model(weights=ResNet50_Weights.DEFAULT):
    """
    Returns ResNet-50 model configured for EF regression.
    
    Args:
        weights: Pre-trained weights (default: ResNet50_Weights.DEFAULT)
    
    Returns:
        ResNet50EF: Configured model
    """
    return ResNet50EF(weights=weights)
