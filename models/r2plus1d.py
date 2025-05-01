import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class ResNet50EF(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT, num_classes=1):
        super(ResNet50EF, self).__init__()
        # Load pre-trained ResNet-50
        self.model = resnet50(weights=weights)
        
        # Modify the first convolutional layer to accept 1-channel input 
(grayscale echo frames)
        self.model.conv1 = nn.Conv2d(
            in_channels=1,  # Echocardiogram frames are typically 
grayscale
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

class R2Plus1DAnomaly(nn.Module):
    def __init__(self, weights=R2Plus1D_18_Weights.DEFAULT):
        super(R2Plus1DAnomaly, self).__init__()
        # Load pre-trained R2+1D-18
        self.model = r2plus1d_18(weights=weights)
        
        # Modify the fully connected layer to output a 512-dimensional 
feature vector
        self.model.fc = nn.Linear(self.model.fc.in_features, 512)
        
        # Add a classifier for anomaly detection (normal vs. abnormal)
        self.classifier = nn.Linear(512, 2)  # Binary classification
        
        # Additional head for EF regression
        self.ef_head = nn.Linear(512, 1)  # EF regression output
        
    def forward(self, x):
        # Input x: (batch_size, channels=3, frames, height, width)
        features = self.model(x)
        ef = self.ef_head(features)  # EF regression output
        anomaly = self.classifier(features)  # Anomaly classification 
output
        return ef, anomaly

def get_resnet_model(weights=ResNet50_Weights.DEFAULT):
    """
    Returns ResNet-50 model configured for EF regression.
    
    Args:
        weights: Pre-trained weights (default: ResNet50_Weights.DEFAULT)
    
    Returns:
        ResNet50EF: Configured model
    """
    return ResNet50EF(weights=weights)

def get_r2plus1d_anomaly_model(weights=R2Plus1D_18_Weights.DEFAULT):
    """
    Returns R2+1D model configured for EF regression and anomaly 
detection.
    
    Args:
        weights: Pre-trained weights (default: 
R2Plus1D_18_Weights.DEFAULT)
    
    Returns:
        R2Plus1DAnomaly: Configured model
    """
    return R2Plus1DAnomaly(weights=weights)
