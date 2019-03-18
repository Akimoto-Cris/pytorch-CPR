import torch.nn as nn
import math
import torch
import hiddenlayer as hl

XAVIER_NORM=True

def init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if XAVIER_NORM:
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            else:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def make_standard_block(feat_in, feat_out, kernel, stride=1, padding=1, use_bn=True):
    layers = [nn.Conv2d(feat_in, feat_out, kernel, stride, padding)]
    if use_bn:
        layers += [nn.BatchNorm2d(feat_out, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def visualize_net(model, save_dir):
    transforms = [
        hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
        hl.transforms.Fold("Conv > BatchNorm", "ConvBn"),
        hl.transforms.FoldDuplicates()
    ]
    hl.build_graph(model, torch.zeros([1, 3, 224, 224]).cuda().float(), transforms=transforms)
    hl.save(save_dir + "/model.pdf")
    print("Net Visualization saved to {}".format(save_dir + "/model.pdf"))
    return True

def adaptive_padding(up1, up2):
    hw1, hw2 = tuple(up1.shape[2:]), list(up2.shape[2:])
    assert (hw1[0] >= hw2[0] and hw1[1] >= hw2[1])

    offset = list(map(lambda x, y: x - y, tuple(hw1), tuple(hw2)))

    return (0,      # padding left
            offset[1],                   # padding right
            0,      # padding top
            offset[0])                   # padding down
