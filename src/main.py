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
    os.environ["CUDA_VISIBLE_DEVICES"] = opt["env"]["device"]
    print("Using GPU: {}".format(opt["env"]["device"]))

    # Create data loaders
    # Create data loaders
    train_loader, test_loader = create_data_loaders(opt)
    # Create nn
    model, criterion_hm, criterion_paf = create_model(opt)
    model = torch.nn.DataParallel(model, device_ids=[int(index) for index in opt["env"]["device"].split(",")]).cuda() \
        if "," in opt["env"]["device"] else model.cuda()
    if opt["env"]["loadModel"] is not None and opt["typ"] == 'cpr':
        model.load_state_dict(torch.load(opt["env"]["loadModel"]))
        print('Loaded model from ' + opt["env"]["loadModel"])
    criterion_hm = criterion_hm.cuda()
    criterion_paf = criterion_paf.cuda()

    # Create optimizer
    optimizer = create_optimizer(opt, model)

    # Other params
    to_train = opt["to_train"]
    visualize_out = opt["viz"]["vizOut"]

    # train/ test
    Processer = process(model)
    if to_train:
        Processer.train_net(train_loader, test_loader, criterion_hm, criterion_paf, optimizer, opt, viz_output=visualize_out)
    else:
        Processer.validate_net(test_loader, criterion_hm, criterion_paf, save_dir=opt["env"]["saveDir"], viz_output=visualize_out)


if __name__ == '__main__':
    main()
