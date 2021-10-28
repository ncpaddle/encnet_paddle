# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn
import paddle.nn.functional as F

import paddle
from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
import numpy as np



@manager.MODELS.add_component
class EncNet(nn.Layer):
    """
    A simple implementation for FCN based on PaddlePaddle.

    The original article refers to
    Evan Shelhamer, et, al. "Fully Convolutional Networks for Semantic Segmentation"
    (https://arxiv.org/abs/1411.4038).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone networks.
        backbone_indices (tuple, optional): The values in the tuple indicate the indices of output of backbone.
            Default: (-1, ).
        channels (int, optional): The channels between conv layer and the last layer of FCNHead.
            If None, it will be the number of channels of input features. Default: None.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(-1,),
                 channels=None,
                 align_corners=False,
                 pretrained=None,
                 aux=False,
                 se_loss=True,
                 lateral=False,
                 norm_layer=nn.BatchNorm,
                 data_format="NCHW"):
        super(EncNet, self).__init__()

        if data_format != 'NCHW':
            raise ('fcn only support NCHW data format')
        self.backbone = backbone
        self.aux = aux
        # backbone_channels = [backbone.feat_channels[i] for i in backbone_indices]
        self.head = EncHead(
            2048,
            num_classes,
            se_loss=se_loss,
            lateral=lateral,
            norm_layer=norm_layer,
            )

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.data_format = data_format
        self.init_weight()

        if aux:
            self.auxlayer = FCNHead(1024, num_classes, norm_layer=norm_layer)

    def forward(self, x):

        imsize = x.shape
        imsize = imsize[2:]
        feat_list = self.backbone(x)
        x = list(self.head(feat_list))
        x[0] = F.interpolate(
                x[0],
                imsize,
                mode='bilinear',
                align_corners=self.align_corners)
        if self.aux:
            auxout = self.auxlayer(feat_list[2])
            auxout = F.interpolate(auxout, imsize, mode='bilinear',
                                    align_corners=self.align_corners)
            x.append(auxout)

        return tuple(x)

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class EncModule(nn.Layer):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2D(in_channels, in_channels, 1, bias_attr=False),
            norm_layer(in_channels),
            nn.ReLU(),
            Encoding(D=in_channels, K=ncodes),
            norm_layer(ncodes),
            nn.ReLU(),
            Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.shape
        gamma = self.fc(en)
        y = paddle.reshape(gamma, (b, c, 1, 1))
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class EncHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 se_loss=True,
                 lateral=False,
                 norm_layer=None,
                 up_kwargs=None,
                 ):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(
            nn.Conv2D(in_channels, 512, 3, padding=1, bias_attr=False),
            norm_layer(512),
            nn.ReLU())
        if lateral:
            self.connect = nn.LayerList([
                nn.Sequential(
                    nn.Conv2D(512, 512, kernel_size=1, bias_attr=False),
                    norm_layer(512),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Conv2D(1024, 512, kernel_size=1, bias_attr=False),
                    norm_layer(512),
                    nn.ReLU()),
            ])
            self.fusion = nn.Sequential(
                    nn.Conv2D(3*512, 512, kernel_size=3, padding=1, bias_attr=False),
                    norm_layer(512),
                    nn.ReLU())
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout(p=0), #(0.1)
                                   nn.Conv2D(512, out_channels, 1))

    def forward(self, feat_list):
        # print("feat_list[-1]:", feat_list[-1])

        feat = self.conv5(feat_list[-1])
        if self.lateral:
            c2 = self.connect[0](feat_list[1])
            c3 = self.connect[1](feat_list[2])
            feat = self.fusion(paddle.concat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))

        outs[0] = self.conv6(outs[0])
        return outs


class Encoding(nn.Layer):
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords = paddle.create_parameter(shape=[K, D], dtype='float32',
                                                 default_initializer=paddle.fluid.initializer.UniformInitializer(low=-std1, high=std1))
        self.scale = paddle.create_parameter(shape=[K], dtype='float32',
                                             default_initializer=paddle.fluid.initializer.UniformInitializer(low=-1, high=0))

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.shape[1] == self.D)
        B, D = X.shape[0], self.D
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = paddle.reshape(X, [B, D, -1])
            X = paddle.transpose(X, perm=[0, 2, 1])
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        A = F.softmax(self.scaled_l2(X, self.codewords, self.scale), axis=2)
        # aggregate
        E = self.aggregate(A, X, self.codewords)
        return E

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, channels = codewords.shape
        batch_size = x.shape[0]
        reshaped_scale = paddle.reshape(scale, (1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.shape[1], num_codes, channels))
        reshaped_codewords = paddle.reshape(codewords, (1, 1, num_codes, channels))

        scaled_l2_norm = reshaped_scale * (
                expanded_x - reshaped_codewords).pow(2).sum(axis=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, channels = codewords.shape
        reshaped_codewords = paddle.reshape(codewords,(1, 1, num_codes, channels))
        batch_size = x.shape[0]
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.shape[1], num_codes, channels))
        encoded_feat = (assignment_weights.unsqueeze(3) *
                        (expanded_x - reshaped_codewords)).sum(axis=1)
        return encoded_feat

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'

class Mean(nn.Layer):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)



class FCNHead(nn.Layer):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, with_global=False):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        if with_global:
            self.conv5 = nn.Sequential(nn.Conv2D(in_channels, inter_channels, 3, padding=1, bias_attr=False),
                                       norm_layer(inter_channels),
                                       nn.ReLU(),
                                       ConcurrentModule([
                                            Identity(),
                                            GlobalPooling(inter_channels, inter_channels,
                                                          norm_layer, self._up_kwargs),
                                       ]),
                                       nn.Dropout(0.1, False),
                                       nn.Conv2D(2*inter_channels, out_channels, 1))
        else:
            self.conv5 = nn.Sequential(nn.Conv2D(in_channels, inter_channels, 3, padding=1, bias_attr=False),
                                       norm_layer(inter_channels),
                                       nn.ReLU(),
                                       nn.Dropout(0.1, False),
                                       nn.Conv2D(inter_channels, out_channels, 1))
    def forward(self, x):
        return self.conv5(x)

class ConcurrentModule(nn.LayerList):
    r"""Feed to a list of modules concurrently.
    The outputs of the layers are concatenated at channel dimension.

    Args:
        modules (iterable, optional): an iterable of modules to add
    """
    def __init__(self, modules=None):
        super(ConcurrentModule, self).__init__(modules)

    def forward(self, x):
        outputs = []
        for layer in self:
            outputs.append(layer(x))
        return paddle.concat(outputs, 1)

class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class GlobalPooling(nn.Layer):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2D(1),
                                 nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return [
            F.interpolate(
                pool,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        #return interpolate(pool, (h,w), **self._up_kwargs)

