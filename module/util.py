''' Modified from https://github.com/alinlab/LfF/blob/master/module/util.py '''

import torch.nn as nn
from module.resnet import resnet20
from module.resnet_ import resnet18_wd4,resnet10
from module.mlp import *
from torchvision.models import resnet18, resnet50, squeezenet1_1,mobilenet_v2
from torchvision import models

def get_model(model_tag, num_classes,num_channel,ssl_feature=512):
    print(model_tag)
    if model_tag == "ResNet20":
        model =  resnet20(num_classes,num_channel)
    elif model_tag == "ResNet20_OURS":
        model = resnet20(num_classes,num_channel)
        model.fc = nn.Linear(128, num_classes)

    elif model_tag == "ResNet18":
 
        print('bringing no pretrained resnet18 ...')
        #model = resnet18(pretrained=False)
        model = models.__dict__['resnet18'](num_classes=num_classes)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    elif model_tag == "MLP":
        return MLP(num_classes=num_classes)
    elif model_tag == "SqueezeNet":

        model = squeezenet1_1(weights=None,num_classes=num_classes)
        model.features[0] = nn.Conv2d(1,64,kernel_size=(3,3),stride=(2,2))

    elif model_tag == 'ResNet18_wd4':
        model = resnet18_wd4(in_channels = 1,num_classes=4)

    elif model_tag == 'ResNet10':
        model = resnet10(in_channels = 1,num_classes=4)
        print(model)
        return model
    elif model_tag == 'MobileNet':
        model = mobilenet_v2(weights=None,num_classes=num_classes)
        #print(model)
        model.features[0][0] = nn.Conv2d(num_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    elif model_tag == "bias_MLP":
        return bias_MLP(num_classes=num_classes,num_channel=num_channel)
    
    elif model_tag == "mlp_DISENTANGLE":
        model =  MLP_DISENTANGLE(num_classes=num_classes)
    elif model_tag == 'resnet_DISENTANGLE':
        print('bringing no pretrained resnet18 disentangle...')
        model = resnet18(pretrained = False)
        model.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#modify torchvision
        model.fc = nn.Linear(1024, num_classes)

    elif model_tag == 'MobileNet_DISENTANGLE':
        print('bringing no pretrained MobileNetV2 disentangle...')
        model = mobilenet_v2(weights=None,num_classes=num_classes)
        model.features[0][0] = nn.Conv2d(num_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.features[18][0] = nn.Conv2d(320, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        model.features[18][1] = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        model.classifier[1] = nn.Linear(1024, num_classes)

    elif model_tag == '1linear':
        model = torch.nn.Sequential(
            torch.nn.Linear(ssl_feature, num_classes, bias=True),
        )
    elif model_tag == '2linear':
        model = torch.nn.Sequential(
            torch.nn.Linear(ssl_feature, int(ssl_feature/2), bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(int(ssl_feature/2), num_classes, bias=False),
        )
    else:
        raise NotImplementedError
    return model
