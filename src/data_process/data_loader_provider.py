import torch
from .coco import CocoDataSet


def create_data_loaders(opt):
    tr_dataset, te_dataset = create_data_sets(opt)
    train_loader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=opt["train"]["batchSize"],
        shuffle=opt["debug"],
        drop_last=True,
        num_workers=opt["train"]["nThreads"]
    )
    test_loader = torch.utils.data.DataLoader(
        te_dataset,
        batch_size=opt["train"]["batchSize"],
        shuffle=False,
        drop_last=True,
        num_workers=opt["train"]["nThreads"],
    )
    return train_loader, test_loader


def create_data_sets(opt):
    if opt["train"]["dataset"] == 'coco':
        tr_dataset = CocoDataSet(opt["env"]["data"], opt, 'train')
        te_dataset = CocoDataSet(opt["env"]["data"], opt, 'val')
    else:
        raise ValueError('Data set ' + opt["train"]["dataset"] + ' not available.')
    return tr_dataset, te_dataset
