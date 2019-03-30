from .vgg import Backend
from .paf_model import PAFModel
from .CPR import CPRmodel
import torch.nn as nn
import torchvision.models as models
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
    if opt["model"]["backend"] == "vgg":
        backend = Backend(models.vgg19, use_bn=True, config=opt["model"][opt["model"]["backend"]])
        backend_feats = 128
    elif opt["model"]["backend"] == "resnet":
        backend = Backend(models.resnet101, use_bn=True, config=opt["model"][opt["model"]["backend"]])
    else:
        raise ValueError('Model ' + opt["model"]["backend"] + ' not available.')
    model = PAFModel(backend, backend_feats, n_joints=18, n_paf=32, n_stages=7) if opt["typ"] == 'paf' else \
                CPRmodel(backend, config=opt)
    if opt["env"]["loadModel"] is not None:
        model.load_state_dict(torch.load(opt["env"]["loadModel"]))
        print('Loaded model from ' + opt["env"]["loadModel"])
    criterion_hm = parse_criterion(opt["train"]["criterionHm"])
    criterion_paf = parse_criterion(opt["train"]["criterionPaf"])
    return model, criterion_hm, criterion_paf


def create_optimizer(opt, model):
    if opt["train"]["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), opt["train"]["LR"])
    elif opt["train"]["optimizer"] == "rmsp":
        return torch.optim.RMSprop(model.parameters(),opt["train"]["LR"])
