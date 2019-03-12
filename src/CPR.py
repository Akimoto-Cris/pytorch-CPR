import torch.nn as nn
import torch.nn.functional as F
from .helper import init, make_standard_block, adaptive_padding
import torch

NUM_JOINTS = 18
NUM_LIMBS = 38
HG_DEPTH=3
HG_NUM_BLOCKS=2
RESIDUAL_BN_CONV=False  # ugly but works

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


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        if C % G == 0:
            x = x.view(N, G, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            x = (x - mean) / (var + self.eps).sqrt()
        else:
            whole = x[:, : G * (C // G), :, :]
            mod = x[:, G * (C // G) + 1:, :, :]
            whole_mean = whole.mean(-1, keepdim=True)
            whole_var = whole.var(-1, keepdim=True)
            mod_mean = mod.mean(-1, keepdim=True)
            mod_var = mod.var(-1, keepdim=True)
            whole, mod = tuple(map(lambda x, mean, var: (x - mean) / (var + self.eps).sqrt(),
                                   [whole, mod], [whole_mean, mod_mean], [whole_var, mod_var]))
            x = torch.cat((whole, mod), axis=1)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class blockFactory:
    def __init__(self, inp_feats, output_feats, blocktype="standard", stage1=False, kernel_size=3):
        if stage1:
            self.block = make_paf_block_stage1(inp_feats, output_feats)
        else:
            if blocktype == "standard":
                self.block = make_paf_block_stage2(output_feats, output_feats)
            elif blocktype == "hg":
                layers = [Hourglass(Bottleneck,
                                    num_blocks=HG_NUM_BLOCKS,
                                    planes=inp_feats,
                                    depth=HG_DEPTH,
                                    kernel_size=kernel_size),
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

    def __init__(self, inplanes, outplanes, activation=False, kernel=3, stride=1):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, inplanes // self.expansion, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(inplanes // self.expansion)
        self.conv2 = nn.Conv2d(inplanes // self.expansion, inplanes // self.expansion, kernel_size=kernel, stride=stride,
                               padding=(kernel - stride) // self.expansion, bias=True)
        self.bn3 = nn.BatchNorm2d(inplanes // self.expansion)
        self.conv3 = nn.Conv2d(inplanes // self.expansion, outplanes, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if inplanes != outplanes:
            self.identical_mapping = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True, stride=1)
            )
        self.stride = stride
        self.activation = activation

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
            residual = self.identical_mapping(x)
        out += residual
        return self.relu(out) if self.activation else out

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth, kernel_size=3, activation=False): # planes : channels
        super(Hourglass, self).__init__()
        self.depth = depth
        self.kernel_size = kernel_size
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

        if activation:
            self.relu = nn.ReLU(inplace=True)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for _ in range(num_blocks + int(i == 0)):
                res.append(block(planes, planes, kernel=self.kernel_size))
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
        low1 = nn.MaxPool2d(2, stride=2)(up1)
        low1 = self.hg[n - 1][1](low1)

        low2 = self._hour_glass_forward(n - 1, low1) if n > 1 \
            else self.hg[n - 1][HG_NUM_BLOCKS](low1)

        low2 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low2, scale_factor=2, mode='bilinear')

        up2 = nn.ZeroPad2d(adaptive_padding(up1, up2))(up2)

        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class CPRmodel(nn.Module):
    def __init__(self, backend, backend_outp_feats, n_joints, n_paf, n_stages=7,
                 share=True, blocktype="standard", kernel_size=3):
        super(CPRmodel, self).__init__()
        assert (n_stages > 0)
        self.backend = backend
        self.n_stages = n_stages
        layers = [Stage(backend_outp_feats, n_joints, n_paf, True, blocktype="standard", kernel_size=kernel_size),
                  Stage(backend_outp_feats, n_joints, n_paf, False, blocktype=blocktype, kernel_size=kernel_size),
                  Stage(backend_outp_feats, n_joints, n_paf, False, blocktype=blocktype, kernel_size=kernel_size)]
        if not share:
            layers += [Stage(backend_outp_feats, n_joints, n_paf, False, blocktype=blocktype,
                             kernel_size=kernel_size)] * (n_stages - 3)
        self.source = layers[1]
        self.share = share

        '''for _ in range(n_stages - 2):
            stage = Stage(backend_outp_feats, n_joints, n_paf, False, blocktype=blocktype)
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

        for _ in range(self.n_stages - 1):
            for i in range(2):
                heatmap_out, paf_out = self.stages[1 + i](cur_feats)
                cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)
            heatmap_outs.append(heatmap_out); paf_outs.append(paf_out)

        return heatmap_outs, paf_outs


class Stage(nn.Module):
    def __init__(self, backend_outp_feats, n_joints, n_paf, stage1, blocktype="standard", kernel_size=3):
        super(Stage, self).__init__()
        if stage1:
            block_fact1 = blockFactory(backend_outp_feats, n_joints, blocktype, True, kernel_size=kernel_size)
            block_fact2 = blockFactory(backend_outp_feats, n_paf, blocktype, True, kernel_size=kernel_size)

            self.block1 = block_fact1.get_block()
            self.block2 = block_fact2.get_block()
        else:
            inp_feats = backend_outp_feats + n_joints + n_paf
            block_fact1 = blockFactory(inp_feats, n_joints, blocktype, kernel_size=kernel_size)
            block_fact2 = blockFactory(inp_feats, n_paf, blocktype, kernel_size=kernel_size)

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

