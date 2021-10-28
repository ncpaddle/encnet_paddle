##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNet variants"""
import math
import torch
import torch.nn as nn

from ...nn import SplAtConv2d, DropBlock2D, GlobalAvgPool2d, RFConv2d
from ..model_store import get_model_file

__all__ = ['ResNet', 'Bottleneck',
           'resnet50', 'resnet101', 'resnet152']
all_id = 0
from reprod_log import ReprodLogger
reprod_logger = ReprodLogger()

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """

    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if all_id > 3:
            exit()
        return out

    # def forward(self, x):
    #
    #     global  all_id
    #     all_id += 1
    #     print("all_id:", all_id)
    #
    #     residual = x
    #     reprod_logger.add("block_" + str(all_id) + "_input", x.cpu().detach().numpy())
    #     out = self.conv1(x)
    #     print("block_" + str(all_id) + "_conv1" + "_param:", self.conv1.state_dict())
    #     reprod_logger.add("block_" + str(all_id) + "_conv1", out.cpu().detach().numpy())
    #     out = self.bn1(out)
    #     reprod_logger.add("block_" + str(all_id) + "_bn1", out.cpu().detach().numpy())
    #     out = self.relu(out)
    #     reprod_logger.add("block_" + str(all_id) + "_relu1", out.cpu().detach().numpy())
    #
    #     out = self.conv2(out)
    #     reprod_logger.add("block_" + str(all_id) + "_conv2", out.cpu().detach().numpy())
    #     out = self.bn2(out)
    #     reprod_logger.add("block_" + str(all_id) + "_bn2", out.cpu().detach().numpy())
    #     out = self.relu(out)
    #     reprod_logger.add("block_" + str(all_id) + "_relu2", out.cpu().detach().numpy())
    #
    #     out = self.conv3(out)
    #     reprod_logger.add("block_" + str(all_id) + "_conv3", out.cpu().detach().numpy())
    #     out = self.bn3(out)
    #     reprod_logger.add("block_" + str(all_id) + "_bn3", out.cpu().detach().numpy())
    #     if self.downsample is not None:
    #         residual = self.downsample(x)
    #         reprod_logger.add("block_" + str(all_id) + "_downsample", residual.cpu().detach().numpy())
    #
    #     out += residual
    #     out = self.relu(out)
    #
    #     if all_id > 3:
    #         exit()
    #     return out

class ResNet(nn.Module):
    """ResNet Variants

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=True, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     # nn.init.constant_(m.weight, 1)
            #     # nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def create(self):
        reprod_logger.save("/home/wzl/Desktop/encnet_reprod/diff/forward_pytorch.npy")


    def forward(self, x):
        feat_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        feat_list.append(c1)
        c2 = self.layer2(c1)
        feat_list.append(c2)
        c3 = self.layer3(c2)
        feat_list.append(c3)
        c4 = self.layer4(c3)
        feat_list.append(c4)

        return feat_list




def resnet50(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50', root=root)), strict=False)
    return model


def resnet101(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet101', root=root)), strict=False)
    return model


def resnet152(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet152', root=root)), strict=False)
    return model
