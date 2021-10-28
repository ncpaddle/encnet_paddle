##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNet variants"""
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils

from reprod_log import ReprodLogger
reprod_logger = ReprodLogger()

__all__ = ['ResNet', 'Bottleneck', 'ResNet50w', 'ResNet101w', 'ResNet152w']
all_id = 0

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class Bottleneck(nn.Layer):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.relu = nn.ReLU()
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
    #     print("########################")
    #
    #     global  all_id
    #     all_id += 1
    #     print("all_id:", all_id)
    #     residual = x
    #     reprod_logger.add("block_" + str(all_id) + "_input", x.cpu().detach().numpy())
    #
    #     print("block_" + str(all_id) + "_conv1" + "_size:" , x.shape)
    #
    #     out = self.conv1(x)
    #
    #     print("block_" + str(all_id) + "_conv1" + "_param:", self.conv1.state_dict())
    #     reprod_logger.add("block_" + str(all_id) + "_conv1", out.cpu().detach().numpy())
    #     out = self.bn1(out)
    #     reprod_logger.add("block_" + str(all_id) + "_bn1", out.cpu().detach().numpy())
    #     out = self.relu(out)
    #     reprod_logger.add("block_" + str(all_id) + "_relu1", out.cpu().detach().numpy())
    #
    #     print("block_" + str(all_id) + "_conv2" + "_size:" , out.shape)
    #
    #     out = self.conv2(out)
    #     reprod_logger.add("block_" + str(all_id) + "_conv2", out.cpu().detach().numpy())
    #     out = self.bn2(out)
    #     reprod_logger.add("block_" + str(all_id) + "_bn2", out.cpu().detach().numpy())
    #     out = self.relu(out)
    #     reprod_logger.add("block_" + str(all_id) + "_relu2", out.cpu().detach().numpy())
    #
    #     print("block_" + str(all_id) + "_conv3" + "_size:" , out.shape)
    #     out = self.conv3(out)
    #     reprod_logger.add("block_" + str(all_id) + "_conv3", out.cpu().detach().numpy())
    #     out = self.bn3(out)
    #     reprod_logger.add("block_" + str(all_id) + "_bn3", out.cpu().detach().numpy())
    #
    #     if self.downsample is not None:
    #         residual = self.downsample(x)
    #         reprod_logger.add("block_" + str(all_id) + "_downsample", residual.cpu().detach().numpy())
    #     out += residual
    #     out = self.relu(out)
    #     if all_id > 3:
    #         exit()
    #     return out

class ResNet(nn.Layer):
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
                 last_gamma=False, norm_layer=nn.BatchNorm2D):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2D(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2D):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     # nn.init.constant_(m.weight, 1)
            #     # nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feat_list = []

        # reprod_logger.add("input", x.cpu().detach().numpy())
        x = self.conv1(x)
        # reprod_logger.add("conv1", x.cpu().detach().numpy())
        x = self.bn1(x)
        # reprod_logger.add("bn1", x.cpu().detach().numpy())

        x = self.relu(x)
        # reprod_logger.add("relu", x.cpu().detach().numpy())
        x = self.maxpool(x)
        # reprod_logger.add("maxpool", x.cpu().detach().numpy())

        c1 = self.layer1(x)
        # reprod_logger.add("c1", c1.cpu().detach().numpy())
        feat_list.append(c1)
        c2 = self.layer2(c1)
        # reprod_logger.add("c2", c2.cpu().detach().numpy())
        feat_list.append(c2)
        c3 = self.layer3(c2)
        # reprod_logger.add("c3", c3.cpu().detach().numpy())
        feat_list.append(c3)
        c4 = self.layer4(c3)
        # reprod_logger.add("c4", c4.cpu().detach().numpy())
        feat_list.append(c4)

        reprod_logger.save("/home/wzl/Desktop/encnet_reprod/diff/forward_paddle.npy")
        #
        # print("x:", x)
        # print("c1:", c1)
        # print("c2:", c2)
        # print("c3:", c3)
        # print("c4:", c4)
        return feat_list



@manager.BACKBONES.add_component
def ResNet50w(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

@manager.BACKBONES.add_component
def ResNet101w(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

@manager.BACKBONES.add_component
def ResNet152w(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model