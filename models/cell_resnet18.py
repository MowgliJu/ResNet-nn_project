import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import torchvision.models as models
import time
from PIL import Image
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

from torchvision import io  # input/output
import torchutils as tu
import json
import numpy as np
import matplotlib.pyplot as plt
import zipfile


class CellsResNet(nn.Module):
    def __init__(self, num_classes=4) -> None:
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer2.parameters():
            param.requires_grad = True
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():  # последний полностью связанный слой
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


def load_model1(weights_path, device):
    model = CellsResNet()
    state_dict = torch.load(weights_path, map_location=device)
    
    # Добавляем префикс "model." к ключам
    adapted_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    
    model.load_state_dict(adapted_state_dict, strict=False)  # strict=False для частичного совпадения
    model.to(device)
    return model
