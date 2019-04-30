from .vgg import Backend
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
    elif criterion == "cross_entropy":
        return nn.CrossEntropyLoss(size_average=False)
    else:
        raise ValueError(f'Criterion{criterion}not supported')

def load_dict(model, load_path):
    dict_new = model.state_dict().copy()
    dict_trained = torch.load(load_path)
    buggey_prefix = 'module.'
    dict_trained_keys = [k.split(buggey_prefix)[-1] if buggey_prefix in k else k for k in dict_trained.keys()]
    interset_keys = list(set(dict_trained_keys) & set(dict_new.keys()))
    for k in interset_keys:
        try:
            dict_new[k] = dict_trained[k]
        except:
            dict_new[k] = dict_trained[buggey_prefix + k]
    model.load_state_dict(dict_new)
    print(f"Loaded {len(interset_keys)} layers from {load_path}, omit {len(dict_trained_keys) - len(interset_keys)} layers.")
    return model

def create_model(opt):
    backend = Backend(opt["model"]["backend"], use_bn=True, config=opt["model"][opt["model"]["backend"]])
    backend_feats = 128
    model = PAFModel(backend, backend_feats, n_joints=18, n_paf=32, n_stages=7) if opt["typ"] == 'paf' else \
                CPRmodel(backend, config=opt)
    if opt["env"]["loadModel"]:
        if opt["typ"] == "cpr":
            model = load_dict(model, opt["env"]["loadModel"])
        else:
            model = torch.load(opt["env"]["loadModel"])
    criterion_hm = parse_criterion(opt["train"]["criterionHm"])
    criterion_paf = parse_criterion(opt["train"]["criterionPaf"])
    return model, criterion_hm, criterion_paf


def create_optimizer(opt, model):
    if opt["train"]["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), opt["train"]["LR"])
    elif opt["train"]["optimizer"] == "rmsp":
        return torch.optim.RMSprop(model.parameters(),opt["train"]["LR"])
