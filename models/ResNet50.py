import torch
import torch.nn as nn
from torchvision import models

#
# ResNet50 结构
# ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
#

class N_ResNet50(nn.Module):
    def __init__(self, fm_ids,pretrained = True):
        super(N_ResNet50, self).__init__()
        self.layers = models.resnet50(pretrained = pretrained)
        self.fm_ids = fm_ids
        print(">>> Built N_ResNet50, features map names: ", fm_ids)
    def forward(self,x):
        fms = []
        for name, module in self.layers._modules.items():
            if name is "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.fm_ids:
                fms.append(x)
            if name == self.fm_ids[-1]:
                break  # stop forward at last required feature map
        return fms
                

# 对应vgg16，两个尺寸为 64x64 , 32x32, 16x16 (1024)
# resnet50 中为 res3b3_relu, res4b5_relu, res5c_relu. 分别为 layer2, layer3, layer4 (2048) 最后一层
def get_resnet50_fms(fm_ids = ['layer2','layer3' ,'layer4'], pretrained=True):
    return  N_ResNet50(fm_ids,pretrained=True)


if __name__ == "__main__":
    fms = get_resnet50_fms(pretrained=True)
