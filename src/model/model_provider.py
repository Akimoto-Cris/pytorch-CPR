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
    else:
        raise ValueError('Criterion ' + criterion + ' not supported')


def create_model(opt):
    if opt.backend == 'vgg':
        backend = VGG()
        backend_feats = 128
    else:
        raise ValueError('Model ' + opt.backend + ' not available.')
    model = PAFModel(backend, backend_feats, n_joints=18, n_paf=32, n_stages=7) if opt.model == 'paf' else \
                CPRmodel(backend, backend_feats, n_joints=18, n_paf=32, n_stages=4, blocktype=opt.blocktype, activation=opt.activation)
    if not opt.loadModel=='none':
        model = torch.load(opt.loadModel)
        print('Loaded model from '+opt.loadModel)
    criterion_hm = parse_criterion(opt.criterionHm)
    criterion_paf = parse_criterion(opt.criterionPaf)
    return model, criterion_hm, criterion_paf


def create_optimizer(opt, model):
    return torch.optim.Adam(model.parameters(), opt.LR)
