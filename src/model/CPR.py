from .helper import *
import torch


def blockFactory(block_type, i, stage):
    if block_type == "std":
        return make_standard_block
    elif block_type == "conv":
        return make_conv_layer
    elif block_type == "botn":
        return Bottleneck
    elif block_type == "hg":
        return Hourglass
    else:
        raise NotImplementedError("stage{} block #{} should be one of `std`, `conv`, 'botn', `hg`.".format(stage, i))


class CPRmodel(nn.Module):
    def __init__(self, backend, config):
        super(CPRmodel, self).__init__()
        self.backend = backend
        self.config = config["model"]
        assert (self.config["nStage"] > 0)
        self.n_stages = self.config["nStage"]
        self.share = self.config["share"]
        layers = [Stage("stage1", self.config),
                  Stage("stage2", self.config)]
        if not self.share:
            layers += [Stage("stage2", self.config) for _ in range(self.config["nStage"] - 2)]
        self.stages = nn.ModuleList(layers)

    def forward(self, x):
        img_feats = self.backend(x)
        cur_feats = img_feats
        heatmap_outs, paf_outs = [], []
        heatmap_out, paf_out = self.stages[0](cur_feats)
        heatmap_outs.append(heatmap_out)
        paf_outs.append(paf_out)
        cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)

        for j in range(self.n_stages - 1):
            for i in range(1):
                heatmap_out, paf_out = self.stages[j + i + 1](cur_feats)
                cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)
                heatmap_outs.append(heatmap_out)
                paf_outs.append(paf_out)

        return heatmap_outs, paf_outs


class Stage(nn.Module):
    def __init__(self, stage, config):
        super(Stage, self).__init__()
        self.stage = stage
        self.config = config
        assert (self.stage in ["stage1", "stage2"])

        self.block1, self.block2 = self._make_blocks(stage)
        init(self.block1, method=config["init"])
        init(self.block2, method=config["init"])

        self.sigmoid = nn.Sigmoid()

    def _make_blocks(self, stage):
        hm_layers, paf_layers = [], []
        # stage1 in plane is backend_out_feat
        inplane = self.config["backend_out_feats"]
        hm_outplane = self.config["nJoints"]
        paf_outplane = self.config["nLimbs"]

        if stage != "stage1":
            inplane += hm_outplane + paf_outplane
        self.stage = stage
        ks = self.config[stage]["k"]
        hm_chs = self.config[stage]["hm_ch"]
        paf_chs = self.config[stage]["paf_ch"]
        types = self.config[stage]["type"]
        assert (len(ks) == len(hm_chs) == len(paf_chs) == len(types))

        for i in range(len(hm_chs)):
            block = blockFactory(types[i], stage, i)
            hm_layers += [block(inplanes=inplane if i == 0 else hm_chs[i - 1],
                                outplanes=hm_outplane if i == len(hm_chs) - 1 else hm_chs[i],
                                kernel=ks[i], stride=1, config=self.config[stage])]
            paf_layers += [block(inplanes=inplane if i == 0 else paf_chs[i - 1],
                                 outplanes=paf_outplane if i == len(paf_chs) - 1 else paf_chs[i],
                                 kernel=ks[i], stride=1, config=self.config[stage])]
        if "hg" not in types:
            return nn.Sequential(*hm_layers), nn.Sequential(*paf_layers)
        else:
            return nn.ModuleList(hm_layers), nn.ModuleList(paf_layers)

    def forward(self, x):
        if "hg" not in self.config[self.stage]["type"]:
            y1 = self.block1(x)
            y2 = self.block2(x)
        else:
            hg_i = self.config[self.stage]["type"].index("hg")
            y1, y2 = x, x
            for i in range(hg_i):
                y1, y2 = self.block1[i](y1), self.block2[i](y2)
                m1, m2 = y1, y2
            y1 = self.block1[hg_i](y1) + m1
            y2 = self.block2[hg_i](y2) + m2
            for i in range(hg_i + 1, len(self.block1)):
                y1, y2 = self.block1[i](y1), self.block2[i](y2)
        return y1, y2
