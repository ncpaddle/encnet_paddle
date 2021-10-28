import paddle
from paddle import nn
import paddle.nn.functional as F
from .cross_entropy_loss import CrossEntropyLoss
from paddleseg.cvlibs import manager
from paddleseg.models import losses

@manager.LOSSES.add_component
class SegmentationLoss(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=True, se_weight=0.2, nclass=19,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=255):
        super(SegmentationLoss, self).__init__(weight=weight, ignore_index=ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLoss, self).forward(pred1, target)
            loss2 = super(SegmentationLoss, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass)
            loss1 = super(SegmentationLoss, self).forward(pred, target)
            loss2 = self.bceloss(paddle.fluid.layers.sigmoid(se_pred), se_target.astype('float32'))
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass)

            loss1 = super(SegmentationLoss, self).forward(pred1, target)
            loss2 = super(SegmentationLoss, self).forward(pred2, target)
            loss3 = self.bceloss(paddle.fluid.layers.sigmoid(se_pred), se_target.astype('float32'))
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.shape[0]
        tvect = paddle.zeros([batch, nclass], dtype='bool')
        for i in range(batch):
            hist = paddle.histogram(target[i],
                                    bins=nclass, min=0,
                                    max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


@manager.LOSSES.add_component
class Seloss(nn.Layer):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self,  nclass=19,  weight=None, ignore_index=0):
        super(Seloss, self).__init__()
        self.nclass = nclass
        self.bceloss = nn.BCELoss(weight)
        self.ignore_index = ignore_index

    def forward(self, logits, label):
        se_target = self._get_batch_label_vector(label, nclass=self.nclass)
        loss = self.bceloss(paddle.fluid.layers.sigmoid(logits), se_target.astype('float32'))
        return loss


    def _get_batch_label_vector(self, target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.shape[0]
        tvect = paddle.zeros([batch, nclass], dtype='bool')
        for i in range(batch):
            hist = paddle.histogram(target[i],
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect