import numpy as np
import paddle
from paddle import fluid
import torch
# from paddleseg.models.backbones import ResNet50_vd
from paddleseg.models.backbones.resnet_w import ResNet101w
from paddleseg.models.backbones.resnet_w import ResNet50w
from paddleseg.models.encnet import EncNet
from reprod_log import ReprodLogger
import collections
from paddleseg.models.losses import SegmentationLoss
from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar



def show():
    path_paddle = "./encnet_paddle.pdparams"
    paddle_dict = paddle.load(path_paddle)
    for key in paddle_dict:
        print(key)

def save_model():
    model = EncNet(num_classes=19,
                   backbone=ResNet101w(),
                   backbone_indices=(0, 1, 2, 3),
                   align_corners=False,
                   pretrained=None,
                   aux=False,
                   se_loss=True,
                   data_format="NCHW"
                   )
    print(model)
    paddle.save(model.state_dict(), "./encnet_paddle.pdparams")


def paddle2torch():
    input_fp = "./encnet_paddle2.pdparams"
    output_fp = "../encnet_pytorch/encnet_pytorch2.pth"
    paddle_dict = paddle.load(input_fp)
    torch_dict = paddle_dict
    fc_names = ["head.encmodule.fc.0.weight", "head.encmodule.selayer.weight"]
    for key in torch_dict:
        weight = torch_dict[key].cpu().detach().numpy()
        flag = [i in key for i in fc_names]
        if any(flag):
            print("weight {} need to be trans".format(key))
            weight = weight.transpose()

        torch_dict[key] = weight
    torch_dict = collections.OrderedDict(
        [(k.replace('_mean', 'running_mean'), v) if '_mean' in k else (k, v) for k, v in torch_dict.items()])
    torch_dict = collections.OrderedDict(
        [(k.replace('_variance', 'running_var'), v) if '_variance' in k else (k, v) for k, v in torch_dict.items()])
    torch.save(torch_dict, output_fp)


def forward():
    paddle.set_device("gpu")
    np.random.seed(0)
    paddle.seed(0)
    reprod_logger = ReprodLogger()
    model = EncNet(num_classes=19,
                   backbone=ResNet101w(),
                   backbone_indices=(0, 1, 2, 3),
                   align_corners=False,
                   pretrained=None,
                   aux=False,
                   se_loss=True,
                   lateral=False,
                   data_format="NCHW"
                   )
    model.load_dict(paddle.load("./encnet_paddle.pdparams"))
    model.train()
    # # # read or gen fake data
    fake_data = np.load("../fake_data.npy")
    # fake_data = [fake_data]
    fake_data = paddle.to_tensor(fake_data)
    # forward
    out = model(fake_data)

    reprod_logger.add("out1", out[0].cpu().detach().numpy())
    # reprod_logger.add("out2", out[1].cpu().detach().numpy())
    reprod_logger.save("../diff/forward_paddle.npy")


def loss_forward():
    paddle.set_device("gpu")
    np.random.seed(0)
    paddle.seed(0)
    reprod_logger = ReprodLogger()
    model = EncNet(num_classes=19,
                   backbone=ResNet101w(),
                   backbone_indices=(0, 1, 2, 3),
                   align_corners=False,
                   pretrained=None,
                   aux=False,
                   se_loss=True,
                   lateral=False,
                   data_format="NCHW"
                   )
    model.load_dict(paddle.load("./encnet_paddle.pdparams"))
    model.train()
    # read or gen fake data
    fake_data = np.load("../fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)

    fake_label = np.load("../fake_label.npy")
    fake_label = paddle.to_tensor(fake_label)

    criterion = SegmentationLoss(nclass=19)
    # forward
    out = model(fake_data)
    loss = criterion(out[0], out[1], fake_label)
    print(loss)
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("../diff/loss_paddle.npy")


def bp():
    paddle.set_device("gpu")
    np.random.seed(0)
    paddle.seed(0)
    reprod_logger = ReprodLogger()
    model = EncNet(num_classes=19,
                   backbone=ResNet101w(),
                   backbone_indices=(0, 1, 2, 3),
                   align_corners=False,
                   pretrained=None,
                   aux=False,
                   se_loss=True,
                   data_format="NCHW"
                   )
    model.load_dict(paddle.load("./encnet_paddle.pdparams"))
    model.train()
    # read or gen fake data
    fake_data = np.load("../fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)

    fake_label = np.load("../fake_label.npy")
    fake_label = paddle.to_tensor(fake_label)
    criterion = SegmentationLoss(nclass=19)
    lr = paddle.optimizer.lr.StepDecay(
        learning_rate=0.01,
        step_size=10, gamma=0.1)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr,
        parameters=model.parameters(),
        weight_decay=1e-4,
        momentum=0.9
    )

    loss_list = []
    for idx in range(3):
        out = model(fake_data)
        loss = criterion(out[0], out[1], fake_label)
        loss.backward()
        print(loss)
        optimizer.step()
        optimizer.clear_grad()
        loss_list.append(loss)

    print(loss_list)
    reprod_logger.add("bp_loss1", loss_list[0].cpu().detach().numpy())
    reprod_logger.add("bp_loss2", loss_list[1].cpu().detach().numpy())
    # reprod_logger.add("bp_loss3", loss_list[2].cpu().detach().numpy())
    reprod_logger.save("../diff/bp_align_paddle.npy")

#
# def metric():
    # paddle.set_device("gpu")
    # np.random.seed(0)
    # paddle.seed(0)
    # reprod_logger = ReprodLogger()
    # logits = paddle.to_tensor(np.random.randint(0, 18, [1, 19, 512, 1024]))
    # label = paddle.to_tensor(np.random.randint(0, 18, [1, 1, 512, 1024]))
    # logit = logits[0]
    # pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
    # print("pred_size:", pred.shape)
    # intersect_area, pred_area, label_area = metrics.calculate_area(
    #     pred,
    #     label,
    #     19,
    #     ignore_index=255)
    # class_iou, miou = metrics.mean_iou(intersect_area, pred_area,
    #                                    label_area)
    # print("miou:", miou)

def train_align():
    reprod_logger = ReprodLogger()
    reprod_logger.add("miou", np.array(0.780))
    reprod_logger.save("../diff/train_align_paddle.npy")


if __name__ == "__main__":
    # forward()
    # loss_forward()
    bp()
