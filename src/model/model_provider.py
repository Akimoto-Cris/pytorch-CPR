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
    if opt.backend == 'vgg':
        backend = VGG(use_bn=True)
        backend_feats = 128
    else:
        raise ValueError('Model ' + opt.backend + ' not available.')
    model = PAFModel(backend, backend_feats, n_joints=18, n_paf=32, n_stages=7) if opt.model == 'paf' else \
                CPRmodel(backend, backend_feats, n_joints=18, n_paf=32, n_stages=6, blocktype=opt.blocktype,
                         share=False, kernel_size=3, activation=opt.activation)
    if opt.model == "paf" and opt.loadModel:
        model = torch.load(opt.loadModel)
    criterion_hm = parse_criterion(opt.criterionHm)
    criterion_paf = parse_criterion(opt.criterionPaf)
    return model, criterion_hm, criterion_paf


def create_optimizer(opt, model):
    return torch.optim.Adam(model.parameters(), opt.LR)
