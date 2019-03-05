import torch.nn as nn
import torch.nn.functional as F
from .helper import init, make_standard_block
import torch

NUM_JOINTS = 18
NUM_LIMBS = 38


def make_paf_block_stage1(inp_feats, output_feats):
    layers = [make_standard_block(inp_feats, 128, 3),       # conv + bn + relu
              make_standard_block(128, 128, 3),
              make_standard_block(128, 128, 3),
              make_standard_block(128, 512, 1, 1, 0)]
    layers += [nn.Conv2d(512, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)


def make_paf_block_stage2(inp_feats, output_feats):
    layers = [make_standard_block(inp_feats, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 1, 1, 0)]
    layers += [nn.Conv2d(128, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)


def blockFactory(inp_feats, output_feats, blocktype="standard", stage1=False):
    if stage1:
        return make_paf_block_stage1(inp_feats, output_feats)
    else:
        if blocktype == "standard":
            return make_paf_block_stage2(output_feats, output_feats)
        elif blocktype == "hg":
            layers = [Hourglass(Bottleneck,
                                num_blocks=2,
                                planes=inp_feats,
                                depth=3),
                      make_standard_block(inp_feats, output_feats, 1, 1, 0)
                      ]
            return nn.Sequential(*layers)
        else:
            raise Exception("Block type {} is not supported".format(blocktype))


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth): # planes : channels
        super(Hourglass, self).__init__()
        self.depth = depth
        # self.upsample = F.interpolate(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes, int(planes / block.expansion)))      # output channel: planes (n_joints / n_paf)
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[2][0](x)
        up1 = self.hg[2][1](up1)
        if n > 1:
            up1 = self._hour_glass_forward(n-1, up1)
        pool = nn.MaxPool2d(2,stride=2)(x)
        low1 = self.hg[1][0](pool)
        low2 = self.hg[1][1](low1)
        if n == 1:
            low2 = self.hg[0][0](low2)
            low2 = self.hg[0][1](low2)

        sam = F.interpolate(low2, scale_factor=2)

        out = up1 + sam
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class CPRmodel(nn.Module):
    def __init__(self, backend, backend_outp_feats, n_joints, n_paf, n_stages=7, blocktype="standard"):
        super(CPRmodel, self).__init__()
        assert (n_stages > 0)
        self.backend = backend
        self.n_stages = n_stages
        self.stages = nn.ModuleList([Stage(backend_outp_feats,
                                           n_joints,
                                           n_paf,
                                           True,
                                           blocktype=blocktype),
                                     Stage(backend_outp_feats,
                                           n_joints,
                                           n_paf,
                                           False,
                                           blocktype=blocktype)])

    def forward(self, x):
        img_feats = self.backend(x)
        cur_feats = img_feats
        heatmap_outs, paf_outs = [], []

        # stage 1
        heatmap_out, paf_out = self.stages[0](cur_feats)
        heatmap_outs.append(heatmap_out)
        paf_outs.append(paf_out)
        cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)

        # stage >= 2, recursive forwarding through same module (or seen as shared weights)
        for _ in range(self.n_stages):
            heatmap_out, paf_out = self.stages[1](cur_feats)
            heatmap_outs.append(heatmap_out); paf_outs.append(paf_out)
            cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)

        return heatmap_outs, paf_outs


class Stage(nn.Module):
    def __init__(self, backend_outp_feats, n_joints, n_paf, stage1, blocktype="standard"):
        super(Stage, self).__init__()
        if stage1:
            self.block1 = blockFactory(backend_outp_feats, n_joints, blocktype, True)
            self.block2 = blockFactory(backend_outp_feats, n_paf, blocktype, True)
        else:
            inp_feats = backend_outp_feats + n_joints + n_paf
            self.block1 = blockFactory(inp_feats, n_joints, blocktype)
            self.block2 = blockFactory(inp_feats, n_paf, blocktype)
        init(self.block1)
        init(self.block2)

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(x)
        return y1, y2

