import torch.nn as nn
import torch.nn.functional as F
from .helper import init, make_standard_block, adaptive_padding
import torch

NUM_JOINTS = 18
NUM_LIMBS = 38
HG_DEPTH=3
HG_NUM_BLOCKS=3

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


class blockFactory:
    def __init__(self, inp_feats, output_feats, blocktype="standard", stage1=False):
        if stage1:
            self.block = make_paf_block_stage1(inp_feats, output_feats)
        else:
            if blocktype == "standard":
                self.block = make_paf_block_stage2(output_feats, output_feats)
            elif blocktype == "hg":
                layers = [Hourglass(Bottleneck,
                                    num_blocks=HG_NUM_BLOCKS,
                                    planes=inp_feats,
                                    depth=HG_DEPTH),
                          make_standard_block(inp_feats, output_feats, 1, 1, 0)]
                self.weights = layers[0].get_weights()
                self.block = nn.Sequential(*layers)
            else:
                raise Exception("Block type {} is not supported".format(blocktype))

    def get_weights(self):
        # tuples (layer, weight)
        return self.weights

    def get_block(self):
        return self.block


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def get_layer_weight_tuples(self):
        return {
            "conv_1": (self.conv1, self.conv1.weight),
            "conv_2": (self.conv2, self.conv2.weight),
            "conv_3": (self.conv3, self.conv3.weight)
        }

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
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for _ in range(num_blocks + int(i == 0)):
                res.append(block(planes, int(planes / block.expansion)))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def get_weights(self):
        weights = dict()
        for i, res in enumerate(self.hg):
            for j, block in enumerate(res):
                # store tuples (layer, weight)
                weights["dep{}/block{}".format(i, j)] = block.get_layer_weight_tuples()
        return weights

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        print("depth: {} up1: {}".format(n, up1.shape))
        low1 = nn.MaxPool2d(2, stride=2)(up1)
        low1 = self.hg[n - 1][1](low1)

        low2 = self._hour_glass_forward(n - 1, low1) if n > 1 \
            else self.hg[n - 1][HG_NUM_BLOCKS](low1)

        low2 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low2, scale_factor=2, mode='bilinear')
        print("depth: {} up2: {}".format(n, up1.shape))

        up2 = nn.ZeroPad2d(adaptive_padding(up1, up2))(up2)

        return up1 + up2

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class CPRmodel(nn.Module):
    def __init__(self, backend, backend_outp_feats, n_joints, n_paf, n_stages=7, blocktype="standard"):
        super(CPRmodel, self).__init__()
        assert (n_stages > 0)
        self.backend = backend
        self.n_stages = n_stages
        layers = [Stage(backend_outp_feats, n_joints, n_paf, True, blocktype=blocktype),
                  Stage(backend_outp_feats, n_joints, n_paf, False, blocktype=blocktype)]

        self.source = layers[1]

        '''for _ in range(n_stages - 2):
            stage = Stage(backend_outp_feats, n_joints, n_paf, False, False, blocktype=blocktype)
            self._copy_weights(self.source, stage)
            layers += [stage]'''
        self.stages = nn.ModuleList(layers)

    def _copy_weights(self, source_block, dest_block):
        for b_name, b_weights in dest_block.get_weights().items():
            for i in range(HG_DEPTH):
                for j in range(HG_NUM_BLOCKS):
                    for layer_index, l_w_t in b_weights["dep{}/block{}".format(i, j)].items():
                        conv_layer, _ = l_w_t
                        _, source_weight = source_block.get_weights()[b_name]["dep{}/block{}".format(i, j)][layer_index]
                        conv_layer.weight = source_weight

    def forward(self, x):
        img_feats = self.backend(x)
        cur_feats = img_feats
        heatmap_outs, paf_outs = [], []
        heatmap_out, paf_out = self.stages[0](cur_feats)
        heatmap_outs.append(heatmap_out); paf_outs.append(paf_out)
        cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)

        for _ in range(self.n_stages - 2):
            heatmap_out, paf_out = self.stages[1](cur_feats)
            heatmap_outs.append(heatmap_out); paf_outs.append(paf_out)
            cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)

        return heatmap_outs, paf_outs


class Stage(nn.Module):
    def __init__(self, backend_outp_feats, n_joints, n_paf, stage1, blocktype="standard"):
        super(Stage, self).__init__()
        if stage1:
            block_fact1 = blockFactory(backend_outp_feats, n_joints, blocktype, True)
            block_fact2 = blockFactory(backend_outp_feats, n_paf, blocktype, True)

            self.block1 = block_fact1.get_block()
            self.block2 = block_fact2.get_block()
        else:
            inp_feats = backend_outp_feats + n_joints + n_paf
            block_fact1 = blockFactory(inp_feats, n_joints, blocktype)
            block_fact2 = blockFactory(inp_feats, n_paf, blocktype)

            self.block1 = block_fact1.get_block()
            self.block2 = block_fact2.get_block()
            self.weights = {
                "block1": block_fact1.get_weights(),
                "block2": block_fact2.get_weights()
            }
        init(self.block1)
        init(self.block2)

    def get_weights(self):
        return self.weights

    def get_blocks(self):
        return [self.block1, self.block2]

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(x)
        return y1, y2

