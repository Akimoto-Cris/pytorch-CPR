import torch.nn as nn
import torch.nn.functional as F
import math
import torch


def init(model, method="default"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if method == "xavier":
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            elif method == "default":
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif method == "kaiming":
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            else:
                raise ModuleNotFoundError("Does not support init method {}.".format(method))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def make_standard_block(inplanes, outplanes, kernel, stride=1, padding=0, config=None):
    layers = [nn.Conv2d(inplanes, outplanes, kernel, stride, padding=(kernel - stride) // 2)]

    if config != None:
        if config['norm'] == "bn":
            layers += [nn.BatchNorm2d(outplanes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        elif config['norm'] == "gn":
            if not config["nGroup"]:
                config["nGroup"] = 32
            layers += [nn.GroupNorm(config["nGroup"], outplanes, eps=1e-05, affine=True)]
    else:
        layers += [nn.BatchNorm2d(outplanes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
   
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

def make_conv_layer(inplanes, outplanes, kernel, stride=1, config=None):
    return nn.Conv2d(inplanes, outplanes, kernel, stride, padding=(kernel - stride) // 2)

class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, outplanes, kernel=3, stride=1, config=None):
        super(Bottleneck, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(inplanes, outplanes // self.expansion, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(outplanes // self.expansion, outplanes // self.expansion,
                               kernel_size=kernel, stride=stride,
                               padding=(kernel - stride) // self.expansion, bias=True)
        self.conv3 = nn.Conv2d(outplanes // self.expansion, outplanes, kernel_size=1, bias=True)
        self._make_norm_layers(inplanes, outplanes)

        self.relu = nn.ReLU()
        self.flag = inplanes != outplanes or stride != 1
        if self.flag:
            self.id_mapping = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True, stride=stride)
            )
        self.stride = stride
        self.activation = self.config["block"]["activation"]

    def _make_norm_layers(self, inplanes, outplanes):
        if self.config["norm"] == "bn":
            self.norm1 = nn.BatchNorm2d(inplanes, eps=1e-05,
                                        momentum=0.1, affine=True, track_running_stats=True)
            self.norm2 = nn.BatchNorm2d(outplanes // self.expansion, eps=1e-05,
                                        momentum=0.1, affine=True, track_running_stats=True)
            self.norm3 = nn.BatchNorm2d(outplanes // self.expansion, eps=1e-05,
                                        momentum=0.1, affine=True, track_running_stats=True)
        elif self.config["norm"] == "gn":
            if not self.config["nGroup"]:
                self.config["nGroup"] = 32
            self.norm1 = nn.GroupNorm(num_groups=self.config["nGroup"], num_channels=inplanes, eps=1e-05, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=self.config["nGroup"], num_channels=outplanes // self.expansion, eps=1e-05, affine=True)
            self.norm3 = nn.GroupNorm(num_groups=self.config["nGroup"], num_channels=outplanes // self.expansion, eps=1e-05, affine=True)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.flag:
            residual = self.id_mapping(x)
        out += residual
        return self.relu(out) if self.activation else out

class Hourglass(nn.Module):
    def __init__(self, inplanes, outplanes, kernel=3, stride=1, config=None): # planes : channels
        super(Hourglass, self).__init__()
        self.depth = config["hg"]["depth"]
        self.config = config
        self.hg = self._make_hour_glass(Bottleneck if config["hg"]["block"]["type"] =="bottleneck" else make_standard_block,
                                        config["hg"]["nBlock"],
                                        inplanes, outplanes, outplanes, kernel=kernel, stride=stride)
        init(self.hg, method=self.config["init"])

    def _make_residual(self, block, num_blocks, inplanes, outplanes, kernel=3, stride=1):
        layers = [block(inplanes, outplanes, kernel=kernel, stride=stride, config=self.config["hg"])] + \
            [block(outplanes, outplanes, kernel=kernel, stride=stride, config=self.config["hg"]) for _ in range(num_blocks - 1)]
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, inplanes, planes, outplanes, kernel=3, stride=1):
        layers = []
        for i in range(self.depth):
            l = [self._make_residual(block, num_blocks, inplanes if i == 0 else planes, planes,
                                     kernel=kernel, stride=stride),
                 self._make_residual(block, num_blocks, planes, planes,
                                     kernel=kernel, stride=stride),
                 self._make_residual(block, num_blocks, planes, outplanes if i == self.depth - 1 else planes,
                                     kernel=kernel, stride=stride)]
            if self.config["hg"]["skip"] == "bottleneck":
                l += [self._make_residual(block, num_blocks, planes, planes)]
            layers += [nn.ModuleList(l)]
        return nn.ModuleList(layers)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = nn.MaxPool2d(2, stride=2)(up1)
        low1 = self.hg[n - 1][1](low1)

        low2 = self._hour_glass_forward(n - 1, low1) if n > 1 \
            else self.hg[n - 1][3](low1)

        low2 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low2, scale_factor=2, mode='bilinear')
        up2 = nn.ZeroPad2d(adaptive_padding(up1, up2))(up2 )
        mapping = self.hg[n - 1][3](up1)
        out = mapping + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

def adaptive_padding(up1, up2):
    hw1, hw2 = tuple(up1.shape[2:]), list(up2.shape[2:])
    assert (hw1[0] >= hw2[0] and hw1[1] >= hw2[1])

    single_offset = list(map(lambda x, y: (x - y) // 2, tuple(hw1), tuple(hw2)))
    leftout = list(map(lambda x, y: (x - y) % 2, tuple(hw1), tuple(hw2)))

    return (single_offset[1] + leftout[1],      # padding left
            single_offset[1],                   # padding right
            single_offset[0] + leftout[0],      # padding top
            single_offset[0])                   # padding down

class supervision_weight(nn.Module):
    def __init__(self, n_stages):
        super(supervision_weight, self).__init__()
        self.n_stages = n_stages
        self.exp = 2.718
        self.block = self.make_fc()
        self._init()

    def make_fc(self):
        return nn.Sequential(
            nn.Linear(self.n_stages, self.n_stages),
            nn.Linear(self.n_stages, self.n_stages),
            nn.Softmax(self.n_stages)
        )

    def _init(self):
        for m in self.block.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_([self.exp ** i for i in range(len(m.weight.data))])
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)