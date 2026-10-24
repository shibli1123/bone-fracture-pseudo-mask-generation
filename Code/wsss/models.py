from __future__ import annotations

import torch.nn as nn
from torchvision import models

CLASS_NAMES = ["fractured", "not_fractured"]


def build_classifier(name: str, pretrained: bool = True, num_classes: int = 2):
    key = name.casefold()
    if key == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(model.fc.in_features, num_classes))
    elif key == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(model.classifier.in_features, num_classes))
    else:
        raise ValueError(f"Unsupported classifier {name!r}; choose resnet50 or densenet121.")
    return model


def cam_layer(model, model_name: str):
    name = model_name.casefold()
    if name == "resnet50":
        return model.layer4[-1]
    if name == "densenet121":
        return model.features.denseblock4
    raise ValueError(f"No CAM layer configured for {model_name!r}.")
