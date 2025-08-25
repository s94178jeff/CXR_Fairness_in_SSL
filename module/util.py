import torch
import torch.nn as nn
import os
from module.resnet1 import resnet20
from module.resnet2 import resnet18_wd4, resnet10
from module.mlp import MLP, bias_MLP, MLP_DISENTANGLE
from torchvision.models import resnet18, squeezenet1_1, mobilenet_v2, ResNet18_Weights, MobileNet_V2_Weights
from torchvision import models
import logging

# 設定日誌記錄
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def create_resnet18(num_classes, num_channel, disentangle=False, weights=None):
    model = resnet18(weights=weights)
    model.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if disentangle:
        model.fc = nn.Linear(1024, num_classes)
    else:
        model.fc = nn.Linear(512, num_classes)
    return model

def create_mobilenet_v2(num_classes, num_channel, disentangle=False, weights=None):
    model = mobilenet_v2(weights=weights)
    model.features[0][0] = nn.Conv2d(num_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    if disentangle:
        model.features[18][0] = nn.Conv2d(320, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        model.features[18][1] = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        model.classifier[1] = nn.Linear(1024, num_classes)
    return model

def get_model(model_tag, num_classes, num_channel, ssl_feature=512):
    logger.info('Loading model: %s', model_tag)

    model_functions = {
        "ResNet20": lambda: resnet20(num_classes, num_channel),
        "ResNet20_OURS": lambda: nn.Sequential(resnet20(num_classes, num_channel), nn.Linear(128, num_classes)),
        "ResNet18": lambda: create_resnet18(num_classes, num_channel, disentangle=False, weights=ResNet18_Weights.DEFAULT),
        "resnet_DISENTANGLE": lambda: create_resnet18(num_classes, num_channel, disentangle=True, weights=ResNet18_Weights.IMAGENET1K_V1),
        "SqueezeNet": lambda: create_squeezenet1_1(num_classes, num_channel),
        "ResNet18_wd4": lambda: resnet18_wd4(in_channels=1, num_classes=4),
        "ResNet10": lambda: resnet10(in_channels=1, num_classes=4),
        "MobileNet": lambda: create_mobilenet_v2(num_classes, num_channel, weights=MobileNet_V2_Weights.DEFAULT),
        "MobileNet_DISENTANGLE": lambda: create_mobilenet_v2(num_classes, num_channel, disentangle=True, weights=MobileNet_V2_Weights.DEFAULT),
        "1linear": lambda: nn.Sequential(nn.Linear(ssl_feature, num_classes, bias=True)),
        "2linear": lambda: nn.Sequential(nn.Linear(ssl_feature, ssl_feature // 2, bias=False), nn.ReLU(), nn.Linear(ssl_feature // 2, num_classes, bias=False)),
        "MLP": lambda: MLP(num_classes=num_classes),
        "bias_MLP": lambda: bias_MLP(num_classes=num_classes, num_channel=num_channel),
        "mlp_DISENTANGLE": lambda: MLP_DISENTANGLE(num_classes=num_classes)
    }

    if model_tag in model_functions:
        return model_functions[model_tag]()
    else:
        raise NotImplementedError(f"Model '{model_tag}' is not implemented.")

def get_vanilla_model(dataset, shortcut_type, shortcut_skew):
    ckpt_path = os.path.join('result', dataset, f'{shortcut_type}{shortcut_skew}_vanilla', 'result', 'best_model.th')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file '{ckpt_path}' does not exist.")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 4)  # Make sure to match the expected output dimension
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    return model