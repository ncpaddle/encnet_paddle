###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter


from ...utils import batch_pix_accuracy, batch_intersection_union

from ..backbone import *

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['BaseNet', 'MultiEvalModule']

def get_backbone(name, **kwargs):
    models = {
        # resnet

        'resnet101': resnet101,

        # resnest
        'resnest50': resnest50,
        'resnest101': resnest101,
        'resnest200': resnest200,
        'resnest269': resnest269,
        # resnet other variants
        'resnet50s': resnet50s,
        'resnet101s': resnet101s,
        'resnet152s': resnet152s,
        'resnet50d': resnet50d,
        'resnet101d': resnet101d,
        'resnext50_32x4d': resnext50_32x4d,
        'resnext101_32x8d': resnext101_32x8d,
        # other segmentation backbones
        'xception65': xception65,
        'wideresnet38': wideresnet38,
        'wideresnet50': wideresnet50,
        }
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net

class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss, dilated=True, norm_layer=None,
                 base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='~/.encoding/models', *args, **kwargs):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        self.BACKBONE = backbone

        self.backbone = get_backbone(backbone, pretrained=False, dilated=dilated,
                                       norm_layer=norm_layer, root=root,
                                       *args, **kwargs)
        self.backbone.fc = None
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        if self.BACKBONE.startswith('wideresnet'):
            x = self.backbone.mod1(x)
            x = self.backbone.pool2(x)
            x = self.backbone.mod2(x)
            x = self.backbone.pool3(x)
            x = self.backbone.mod3(x)
            x = self.backbone.mod4(x)
            x = self.backbone.mod5(x)
            c3 = x.clone()
            x = self.backbone.mod6(x)
            x = self.backbone.mod7(x)
            x = self.backbone.bn_out(x)
            return None, None, c3, x
        else:

            # reprod_logger.add("input", x.cpu().detach().numpy())
            x = self.backbone.conv1(x)
            # reprod_logger.add("conv1", x.cpu().detach().numpy())

            x = self.backbone.bn1(x)
            # reprod_logger.add("bn1", x.cpu().detach().numpy())

            x = self.backbone.relu(x)
            # reprod_logger.add("relu", x.cpu().detach().numpy())
            x = self.backbone.maxpool(x)
            # reprod_logger.add("maxpool", x.cpu().detach().numpy())

            c1 = self.backbone.layer1(x)
            # reprod_logger.add("c1", c1.cpu().detach().numpy())
            c2 = self.backbone.layer2(c1)
            # reprod_logger.add("c2", c2.cpu().detach().numpy())
            c3 = self.backbone.layer3(c2)
            # reprod_logger.add("c3", c3.cpu().detach().numpy())
            c4 = self.backbone.layer4(c3)

            # self.backbone.create()
            #
            # print("x:", x)
            # print("c1:", c1)
            # print("c2:", c2)
            # print("c3:", c3)
            # print("c4:", c4)
        return c1, c2, c3, c4

    def test_forward(self, x):
        x = self.backbone.conv1(x)
        return x

    def evaluate(self, x, target=None):
        pred = self.forward(x)

        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union


class MultiEvalModule(DataParallel):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        super(MultiEvalModule, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.scales = scales
        self.flip = flip
        print('MultiEvalModule: base_size {}, crop_size {}'. \
            format(self.base_size, self.crop_size))

    def parallel_forward(self, inputs, **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        inputs = [(input.unsqueeze(0).cuda(device),)
                  for input, device in zip(inputs, self.device_ids)]

        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])

        print("-------------:", inputs)
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        #for out in outputs:
        #    print('out.size()', out.size())
        return outputs

    def forward(self, image):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        batch, _, h, w = image.size()
        assert(batch == 1)
        stride_rate = 2.0/3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch,self.nclass,h,w).zero_().cuda()

        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            """
            short_size = int(math.ceil(self.base_size * scale))
            if h > w:
                width = short_size
                height = int(1.0 * h * short_size / w)
                long_size = height
            else:
                height = short_size
                width = int(1.0 * w * short_size / h)
                long_size = width
            """
            # resize image to current size
            cur_img = resize_image(image, height, width, **self.module._up_kwargs)
            if long_size <= crop_size:
                pad_img = pad_image(cur_img, self.module.mean,
                                    self.module.std, crop_size)
                outputs = module_inference(self.module, pad_img, self.flip)
                outputs = crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = pad_image(cur_img, self.module.mean,
                                        self.module.std, crop_size)
                else:
                    pad_img = cur_img
                _,_,ph,pw = pad_img.size()
                assert(ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch,self.nclass,ph,pw).zero_().cuda()
                    count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = pad_image(crop_img, self.module.mean,
                                                 self.module.std, crop_size)
                        output = module_inference(self.module, pad_crop_img, self.flip)
                        outputs[:,:,h0:h1,w0:w1] += crop_image(output,
                            0, h1-h0, 0, w1-w0)
                        count_norm[:,:,h0:h1,w0:w1] += 1
                assert((count_norm==0).sum()==0)
                outputs = outputs / count_norm
                outputs = outputs[:,:,:height,:width]

            score = resize_image(outputs, h, w, **self.module._up_kwargs)
            scores += score

        return scores


def module_inference(module, image, flip=True):
    output = module.evaluate(image)
    if flip:
        fimg = flip_image(image)
        foutput = module.evaluate(fimg)
        output += flip_image(foutput)
    return output.exp()

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)
