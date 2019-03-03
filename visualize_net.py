import torch
import torchvision.models
import hiddenlayer as hl

def visualize_net(model, save_dir):
    transforms = [
        hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
        hl.transforms.Fold("Conv > BatchNorm", "ConvBn"),
        hl.transforms.FoldDuplicates()
    ]
    try:
        hl.build_graph(model, torch.zeros([1, 3, 224, 224]).cuda().float(), transforms=transforms)
        hl.save(save_dir + "/model.pdf")
        print("Net Visualization saved to {}".format(save_dir + "/model.pdf"))
        return True
    except:
        print("Failed to save net visualization")