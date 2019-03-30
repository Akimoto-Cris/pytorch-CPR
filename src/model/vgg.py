import torch.nn as nn
import torchvision.models as models
from .helper import init, make_standard_block


class Backend(nn.Module):
    def __init__(self, typ=models.vgg19, use_bn=True, config=None):  # Original implementation doesn't use BN
        super(Backend, self).__init__()
        backend = typ(pretrained=True)
        if typ == models.vgg19:
            backend = list(list(backend.children())[0].children())[:config['layer_to_use']]
            self.backend = nn.Sequential(*backend)
            self.feature_extractor = nn.Sequential(make_standard_block(512, 256, 3, config=config),
                                                   make_standard_block(256, 128, 3, config=config))
        elif typ == models.resnet34:
            backend = list(backend.children())[: config['layer_to_use']]
            self.backend = nn.Sequential(*backend)
            self.feature_extractor = nn.Sequential(make_standard_block(128, 128, 3, config=config))

        init(self.feature_extractor, method=config["init"])

    def forward(self, x):
        backend_out = self.backend(x)
        x = self.feature_extractor(backend_out)
        return x
