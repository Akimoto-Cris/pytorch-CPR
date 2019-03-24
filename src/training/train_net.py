import torch
from torch.autograd import Variable
import os
from tqdm import tqdm
from visualization.visualize import visualize_output
import threading
from queue import Queue


class process:
    def __init__(self, model):
        self.model = model

    def step(self, data_loader, criterion_hm, criterion_paf, to_train=False,
             optimizer=None, viz_output=False, regularization=False, lamda=1.0):
        if to_train:
            self.model.train()
        else:
            self.model.eval()

        if viz_output:
            self.queue = Queue()
            self.costumer = visualize_thread(self.queue)
            self.costumer.start()

        nIters = len(data_loader)
        hm_loss_meter, paf_loss_meter = AverageMeter(), AverageMeter()
        with tqdm(total=nIters) as t:
            for i, (input_, heatmap, paf, ignore_mask, indices) in enumerate(data_loader):
                input_cuda = Variable(input_).float().cuda()
                heatmap_t_cuda = Variable(heatmap).float().cuda()
                paf_t_cuda = Variable(paf).float().cuda()
                ignore_mask_cuda = Variable(ignore_mask.reshape(ignore_mask.shape[0], 1,
                                                       ignore_mask.shape[1], ignore_mask.shape[2])).float().cuda()
                allow_mask = 1 - ignore_mask_cuda
                heatmap_outputs, paf_outputs = self.model(input_cuda)

                loss_hm_total = 0.0
                loss_paf_total = 0.0
                for i in range(len(heatmap_outputs)):
                    heatmap_out = heatmap_outputs[i]
                    paf_out = paf_outputs[i]
                    loss_hm_total += criterion_hm(heatmap_out * allow_mask, heatmap_t_cuda * allow_mask) \
                                     / allow_mask.sum().detach()/heatmap.shape[0]/heatmap.shape[1]
                    loss_paf_total += criterion_paf(paf_out * allow_mask, paf_t_cuda * allow_mask) \
                                      / allow_mask.sum().detach()/heatmap.shape[0]/paf.shape[1]
                loss = loss_hm_total + loss_paf_total
                if regularization:
                    loss_reg = 0.0
                    for param in self.model.parameters():
                        loss_reg += lamda * torch.norm(param)
                    loss += loss_reg
                output = (heatmap_outputs[1].data.cpu().numpy(), paf_outputs[1].data.cpu().numpy(), indices.numpy())
                if to_train:
                    loss.backward()
                    optimizer.step()
                if viz_output:
                    self.queue.put((input_.numpy(), heatmap.numpy(), paf.numpy(), ignore_mask.numpy(), output))

                hm_loss_meter.update(loss_hm_total.data.cpu().numpy())
                paf_loss_meter.update(loss_paf_total.data.cpu().numpy())
                t.set_postfix(loss_hm='{:05.3f}'.format(hm_loss_meter.avg), loss_paf='{:05.3f}'.format(paf_loss_meter.avg))
                t.update()

        return hm_loss_meter.avg, paf_loss_meter.avg

    def train_net(self, train_loader, test_loader, criterion_hm, criterion_paf, optimizer,
                  opt, viz_output=False):

        heatmap_loss_avg, paf_loss_avg = 0.0, 0.0
        for epoch in range(1, opt["train"]["nEpoch"] + 1):
            epoch += opt["env"]["checkpoint"]
            self.step(train_loader, criterion_hm, criterion_paf, True, optimizer,
                      regularization=opt["train"]["regularization"], lamda=opt["train"]["lambda"],
                      viz_output=viz_output)
            opt["env"]["checkpoint"] = epoch
            if epoch % opt["train"]["valInterval"] == 0:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    self.validate_net(test_loader, criterion_hm, criterion_paf, opt["env"]["saveDir"], epoch, viz_output=viz_output)
            adjust_learning_rate(optimizer, epoch, opt["train"]["dropLR"], opt["train"]["LR"])

        self.costumer.join()
        return heatmap_loss_avg, paf_loss_avg

    def validate_net(self, test_loader, criterion_hm, criterion_paf, save_dir=None, epoch=0, viz_output=False):
        heatmap_loss_avg, paf_loss_avg = self.step(test_loader, criterion_hm, criterion_paf, viz_output=viz_output)
        if save_dir:
            torch.save(self.model.state_dict(), os.path.join(save_dir, 'model_{}.pth'.format(epoch)))
        return heatmap_loss_avg, paf_loss_avg


class visualize_thread(threading.Thread):
    def __init__(self, queue):
        super(visualize_thread, self).__init__()
        self.data = queue

    def run(self):
        input_, heatmap, paf, ignore_mask, output = self.data.get()
        visualize_output(input_, heatmap, paf, ignore_mask, output)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    lr = LR * (0.1 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
