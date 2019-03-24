from .vgg import VGG
from .paf_model import PAFModel
from .CPR import CPRmodel
import torch.nn as nn
import torch

def parse_criterion(criterion):
    if criterion == 'l1':
        return nn.L1Loss(size_average = False)
    elif criterion == 'mse':
        return nn.MSELoss(size_average = False)
    elif criterion == 'smoooth_l1':
        return nn.SmoothL1Loss(size_average=False)
    else:
        raise ValueError('Criterion ' + criterion + ' not supported')


def create_model(opt):
    if opt["model"]["backend"] == 'vgg':
        backend = VGG(use_bn=True, config=opt["model"]["vgg"])
        backend_feats = 128
    else:
        raise ValueError('Model ' + opt["model"]["backend"] + ' not available.')
    model = PAFModel(backend, backend_feats, n_joints=18, n_paf=32, n_stages=7) if opt["typ"] == 'paf' else \
                CPRmodel(backend, config=opt)
    if opt["typ"] == "paf" and opt["env"]["loadModel"]:
        model = torch.load(opt["env"]["loadModel"])
    criterion_hm = parse_criterion(opt["train"]["criterionHm"])
    criterion_paf = parse_criterion(opt["train"]["criterionPaf"])
    return model, criterion_hm, criterion_paf


def create_optimizer(opt, model):
    if opt["train"]["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), opt["train"]["LR"])
    elif opt["train"]["optimizer"] == "rmsp":
        return torch.optim.RMSprop(model.parameters(),opt["train"]["LR"])