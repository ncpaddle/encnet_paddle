import torch
from encoding.models import get_model
import numpy as np
from reprod_log import ReprodLogger
import collections
import paddle
from encoding.nn.loss import SegmentationLosses
import encoding.utils as utils
torch.set_printoptions(precision=8)
def save_pytorch_model():
    path_pytorch = "./encnet_pytorch.pth"
    model = get_model('encnet_resnet101_citys', pretrained=False).cuda()
    print(model)
    torch.save(model.state_dict(), path_pytorch)

def pytorch2paddle():
    input_fp = "./encnet_pytorch.pth"
    output_fp = "../encnet_paddle/encnet_paddle.pdparams"
    torch_dict = torch.load(input_fp, map_location=torch.device('cpu'))
    paddle_dict = torch_dict
    fc_names = ["head.encmodule.fc.0.weight", "head.encmodule.selayer.weight"]
    for key in paddle_dict:
        weight = paddle_dict[key].cpu().detach().numpy()
        flag = [i in key for i in fc_names]
        if any(flag):
            print("weight {} need to be trans".format(key))
            weight = weight.transpose()
        paddle_dict[key] = weight

    paddle_dict = collections.OrderedDict(
        [(k.replace('running_mean', '_mean'), v) if 'running_mean' in k else (k, v) for k, v in paddle_dict.items()])
    paddle_dict = collections.OrderedDict(
        [(k.replace('running_var', '_variance'), v) if 'running_var' in k else (k, v) for k, v in paddle_dict.items()])
    paddle.save(paddle_dict, output_fp)

def show():
    path_pytorch = "./encnet_pytorch.pth"
    torch_dict = torch.load(path_pytorch, map_location=torch.device('cpu'))
    ll = []
    for key in torch_dict:
        ll.append(key)
    for i in range(len(ll)):
        print(ll[i])


def forward():
    np.random.seed(0)
    torch.manual_seed(0)
    reprod_logger = ReprodLogger()
    model = get_model('encnet_resnet101_citys', pretrained=False).cuda()
    model.load_state_dict(torch.load("./encnet_pytorch.pth", map_location=torch.device('cpu')))
    model.train()
    # read or gen fake data
    fake_data = np.load("../fake_data.npy")
    fake_data = torch.from_numpy(fake_data).cuda()
    # forward
    out = model(fake_data)
    # print(out)
    # print(out[0].size())
    # print(out[1].size())
    reprod_logger.add("out1", out[0].cpu().detach().numpy())
    # reprod_logger.add("out2", out[1].cpu().detach().numpy())
    reprod_logger.save("../diff/forward_pytorch.npy")


def loss_forward():
    np.random.seed(0)
    torch.manual_seed(0)
    reprod_logger = ReprodLogger()
    model = get_model('encnet_resnet101_citys', pretrained=False).cuda()
    model.load_state_dict(torch.load("./encnet_pytorch.pth", map_location=torch.device('cpu')))
    model.train()

    # read or gen fake data
    fake_data = np.load("../fake_data.npy")
    fake_data = torch.from_numpy(fake_data).cuda()

    fake_label = np.load("../fake_label.npy")
    fake_label = torch.from_numpy(fake_label).cuda()

    criterion = SegmentationLosses(nclass=19)
    # forward
    out = model(fake_data)
    loss = criterion(out[0], out[1], fake_label.long())

    print(loss)
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("../diff/loss_pytorch.npy")



def bp():
    np.random.seed(0)
    torch.manual_seed(0)
    reprod_logger = ReprodLogger()
    model = get_model('encnet_resnet101_citys', pretrained=False).cuda()
    model.load_state_dict(torch.load("./encnet_pytorch.pth", map_location=torch.device('cpu')))
    model.train()
    # read or gen fake data
    fake_data = np.load("../fake_data.npy")
    fake_data = torch.from_numpy(fake_data).cuda()

    fake_label = np.load("../fake_label.npy")
    fake_label = torch.from_numpy(fake_label).cuda()

    criterion = SegmentationLosses(nclass=19)
    # forward
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_list = []
    for idx in range(3):
        out = model(fake_data)
        loss = criterion(out[0], out[1], fake_label.long())
        loss.backward()
        print(loss)

        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss)

    print(loss_list)
    reprod_logger.add("bp_loss1", loss_list[0].cpu().detach().numpy())
    reprod_logger.add("bp_loss2", loss_list[1].cpu().detach().numpy())
    # reprod_logger.add("bp_loss3", loss_list[2].cpu().detach().numpy())
    # reprod_logger.add("bp_loss4", loss_list[3].cpu().detach().numpy())
    # reprod_logger.add("bp_loss5", loss_list[4].cpu().detach().numpy())
    reprod_logger.save("../diff/bp_align_pytorch.npy")

# def metric():
#     np.random.seed(0)
#     torch.manual_seed(0)
#     reprod_logger = ReprodLogger()
#     metric = utils.SegmentationMetric(19)
#     dist = [torch.tensor(np.random.randint(0, 19, [1024, 2048]))]
#     predicts = [torch.tensor(np.random.randint(0, 19, [1024, 2048]))]
#     metric.update(dist, predicts)
#
#     pixAcc, mIoU = metric.get()
#     print("miou:", mIoU)

def train_align():
    reprod_logger = ReprodLogger()
    reprod_logger.add("miou", np.array(0.780))
    reprod_logger.save("../diff/train_align_pytorch.npy")

if __name__ == "__main__":
    # save_pytorch_model()
    # pytorch2paddle()
    # show()
    # forward()
    # loss_forward()
    bp()
