import torch
import torch.nn as nn
import torchvision.models.video as video_models

def r2plus1d_18(pretrained=False, **kwargs):
    """Constructs a R(2+1)D-18 model."""
    model = video_models.r2plus1d_18(pretrained=pretrained, **kwargs)
    return model
