import torch.nn as nn
from torchvision import models

def get_resnet18_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained == True:
        model = models.resnet18(weights='IMAGENET1K_V1')
    else:
        model = models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model