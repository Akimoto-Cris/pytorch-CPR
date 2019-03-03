import numpy as np
import torch
import random
from opts.base_opts import Opts
from data_process.data_loader_provider import create_data_loaders
from model.model_provider import create_model, create_optimizer
from training.train_net import process
import os


def main():
    # Seed all sources of randomness to 0 for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    opt = Opts().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    print("Using GPU: {}".format(opt.device))

    # Create data loaders
    train_loader, test_loader = create_data_loaders(opt)

    # Create nn
    model, criterion_hm, criterion_paf = create_model(opt)

    model = torch.nn.DataParallel(model, device_ids=[int(index) for index in opt.device.split(",")]).cuda() \
        if "," in opt.device else model.cuda()

    if opt.vizModel:
        from model.helper import visualize_net
        visualize_net(model, opt.saveDir)

    criterion_hm = criterion_hm.cuda()
    criterion_paf = criterion_paf.cuda()

    # Create optimizer
    optimizer = create_optimizer(opt, model)

    # Other params
    n_epochs = opt.nEpoch
    to_train = opt.train
    drop_lr = opt.dropLR
    val_interval = opt.valInterval
    learn_rate = opt.LR
    visualize_out = opt.vizOut

    # train/ test
    Processer = process(model)
    if to_train:
        Processer.train_net(train_loader, test_loader, criterion_hm, criterion_paf, optimizer, n_epochs,
                  val_interval, learn_rate, drop_lr, opt.saveDir, visualize_out)
    else:
        Processer.validate_net(test_loader, criterion_hm, criterion_paf, viz_output=visualize_out)


if __name__ == '__main__':
    main()
