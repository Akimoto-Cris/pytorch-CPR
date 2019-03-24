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
            else:
                raise ModuleNotFoundError("Does not support init method {}.".format(method))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def make_standard_block(inplanes, outplanes, kernel, stride=1, config=None):
    layers = [nn.Conv2d(inplanes, outplanes, kernel, stride, padding=(kernel - stride) // 2)]
    norm = config["norm"]
    if norm == "bn":
        layers += [nn.BatchNorm2d(outplanes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
    elif norm == "gn":
        layers += [GroupNorm(32, outplanes, eps=1e-05, affine=True)]
    else:
        raise NotImplementedError("norm type shoud be one of `bn`, `gn`, not {}".format(norm))
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

def make_conv_layer(inplanes, outplanes, kernel, stride=1, config=None):
    return nn.Conv2d(inplanes, outplanes, kernel, stride, padding=(kernel - stride) // 2)

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_features, num_groups=89, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__(num_groups, num_channels=num_features, eps=eps, affine=affine)
        self.weight = nn.parameter.Parameter(torch.Tensor(num_features))
        self.bias = nn.parameter.Parameter(torch.Tensor(num_features))

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

        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

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
            self.norm1 = GroupNorm(self.config["nGroup"], inplanes, eps=1e-05, affine=True)
            self.norm2 = GroupNorm(self.config["nGroup"], outplanes // self.expansion, eps=1e-05, affine=True)
            self.norm3 = GroupNorm(self.config["nGroup"], outplanes // self.expansion, eps=1e-05, affine=True)

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

        mapping = self.hg[n - 1][3](up1)
        out = mapping + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)