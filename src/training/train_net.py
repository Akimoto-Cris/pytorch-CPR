import torch, random
from torch.autograd import Variable
import os, cv2
from tqdm import tqdm
from visualization.visualize import visualize_output
import threading
from queue import Queue
import hiddenlayer as hl
from model.helper import root_loss, limb_aware_loss

torch.multiprocessing.set_sharing_strategy('file_system')

class process:
    def __init__(self, model):
        self.model = model
        self.hist_loss = hl.History()
        self.hist_fig = hl.History()
        self.canvas_hm = hl.Canvas()
        self.canvas_paf = hl.Canvas()
        self._step = (0, 0)
        # self.model.stages[-1].block1[-1].register_forward_hook(self.activations_hook)

    def activations_hook(self, input, heatmaps, pafs):
        """Intercepts the forward pass and logs activations.
        """
        batch_ix = self._step[1]
        if batch_ix % 40 == 0:
            # The output of this layer is of shape [batch_size, 16, 32, 32]
            # Take a slice that represents one feature map
            print("hook!!!")
            print(heatmaps[-1].data[0, 0])
            self.hist.log(self._step, backend_output=heatmaps[-1].data[0, 0])

    def step(self, data_loader, criterion_hm, criterion_paf, to_train=False,
             optimizer=None, viz_output=False, regularization=False, lamda=1.0, opts=None, epoch=0, criterion_sw=None):
        if to_train:
            self.model.train()
        else:
            self.model.eval()

        if viz_output:
            self.queue = Queue()
            self.costumer = visualize_thread(self.queue, opts)
            self.costumer.start()

        nIters = len(data_loader)
        hm_loss_meter, paf_loss_meter = AverageMeter(), AverageMeter()
        if os.path.exists(opts["env"]["saveDir"] + '/experiment.pkl'):
            self.hist_loss.load(opts["env"]["saveDir"] + '/experiment.pkl')
        with tqdm(total=nIters) as t:
            for i, (input_, heatmap, paf, ignore_mask, indices) in enumerate(iter(data_loader)):
                self._step = (epoch, i)
                input_cuda = input_.float().cuda()
                heatmap_t_cuda = heatmap.float().cuda()
                paf_t_cuda = paf.float().cuda()
                ignore_mask_cuda = ignore_mask.reshape(ignore_mask.shape[0], 1,
                                                       ignore_mask.shape[1], ignore_mask.shape[2]).float().cuda()
                allow_mask = 1 - ignore_mask_cuda
                heatmap_outputs, paf_outputs = self.model(input_cuda)
                stage_weight = [1.] * len(heatmap_outputs)

                loss_hm_total = []
                loss_paf_total = []
                loss_la_total = []
                for i in range(len(heatmap_outputs)):
                    heatmap_out = heatmap_outputs[i]
                    paf_out = paf_outputs[i]
                    loss_hm_total += [criterion_hm(heatmap_out * allow_mask, heatmap_t_cuda * allow_mask) /
                                      allow_mask.sum().detach()/heatmap.shape[0]/heatmap.shape[1]]
                    loss_paf_total += [criterion_paf(paf_out * allow_mask, paf_t_cuda * allow_mask) \
                                      / allow_mask.sum().detach()/heatmap.shape[0]/paf.shape[1]]
                    if opts["model"]["limb_aware"]:
                        self.hist_fig.log(self._step, f_t=limb_aware_loss(heatmap_t_cuda * allow_mask, paf_t_cuda * allow_mask).data[0, 0])
                        # self.hist_fig.log(self._step, f_t=paf_t_cuda.data[0, 0])
                        loss_la_total += [criterion_paf(limb_aware_loss(heatmap_out * allow_mask, paf_out * allow_mask),
                                                      limb_aware_loss(heatmap_t_cuda * allow_mask, paf_t_cuda * allow_mask))/
                                                      allow_mask.sum().detach() / heatmap[0].shape[0] / heatmap[0].shape[1]]
                if opts["model"]["stage_weight"]:
                    stage_weight_hm = root_loss(loss_hm_total)
                    stage_weight_paf = root_loss(loss_paf_total)
                    stage_weight = torch.Tensor([0.5 + sw_hm + sw_paf for sw_hm, sw_paf in zip(stage_weight_hm, stage_weight_paf)]).cuda()

                loss = sum(
                    [(loss_hm + loss_paf) * sw for loss_hm, loss_paf, sw in zip(loss_hm_total, loss_paf_total, stage_weight)])
                if opts["model"]["limb_aware"]:
                    loss += sum(loss_la_total)
                if regularization:
                    loss_reg = 0.0
                    for param in self.model.parameters():
                        loss_reg += lamda * torch.norm(param)
                    loss += loss_reg
                output = (heatmap_outputs[-1].data.cpu().numpy(), paf_outputs[-1].data.cpu().numpy(), indices.numpy())
                # print(type(heatmap_outputs[0]))

                if to_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if viz_output:
                        self.hist_fig.log(self._step, heatmap_o=heatmap_outputs[-1].data[0, 0])
                        self.canvas_hm.draw_image(self.hist_fig["f_t"])
                        # self.canvas_paf.draw_image(self.hist["paf_o"])
                        self.queue.put((i, input_.numpy(), heatmap.numpy(), paf.numpy(), ignore_mask.numpy(), output))

                hm_loss_meter.update(sum(loss_hm_total).data.cpu().numpy())
                paf_loss_meter.update(sum(loss_paf_total).data.cpu().numpy())

                self.hist_loss.log(self._step, loss_hm=hm_loss_meter.avg, loss_paf=paf_loss_meter.avg, mu=stage_weight)
                self.hist_loss.save(opts["env"]["saveDir"] + "/experiment.pkl")

                t.set_postfix(loss_hm='{:05.4f}'.format(hm_loss_meter.avg),
                              loss_paf='{:05.4f}'.format(paf_loss_meter.avg))
                t.update()
        return hm_loss_meter.avg, paf_loss_meter.avg

    def train_net(self, train_loader, test_loader, criterion_hm, criterion_paf, optimizer,
                  opt, viz_output=False, criterion_sw=None):
        for epoch in range(1 + opt["env"]["checkpoint"], opt["train"]["nEpoch"] + 1):
            _ = self.step(train_loader, criterion_hm, criterion_paf, True, optimizer,
                      regularization=opt["train"]["regularization"], lamda=opt["train"]["lambda"], epoch=epoch,
                      viz_output=viz_output, opts=opt, criterion_sw=criterion_sw)
            opt["env"]["checkpoint"] = epoch
            if epoch % opt["train"]["valInterval"] == 0:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    self.validate_net(test_loader, criterion_hm, criterion_paf, opt["env"]["saveDir"], epoch=epoch,
                                      viz_output=viz_output, opts=opt)
            adjust_learning_rate(optimizer, epoch, opt["train"]["dropLR"], opt["train"]["LR"])

        self.costumer.join()

    def validate_net(self, test_loader, criterion_hm, criterion_paf, save_dir=None,
                     epoch=0, viz_output=False, opts=None):
        if save_dir:
            torch.save(self.model.state_dict(), os.path.join(save_dir, 'model_{}.pth'.format(epoch)))
        with torch.no_grad():
            losses_avg = self.step(test_loader, criterion_hm, criterion_paf,
                                                       viz_output=viz_output, opts=opts, epoch=epoch)
        return losses_avg


class visualize_thread(threading.Thread):
    def __init__(self, queue, opts):
        super(visualize_thread, self).__init__()
        self.data = queue
        self.opts = opts

    def run(self):
        i, input_, heatmap, paf, ignore_mask, output = self.data.get()
        heatmap_o, paf_o, _= output
        if self.opts['viz']["show_image"]:
            visualize_output(input_, heatmap, paf, ignore_mask, output)
        vis_path = os.path.join(self.opts["env"]["saveDir"], 'train_viz')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        if len(heatmap_o)>3:
            heatmap_o = heatmap_o[0, :3, :, :].transpose(1, 2, 0)
            paf_o = paf_o[0, :3, :, :].transpose(1, 2, 0)
        cv2.imwrite(os.path.join(vis_path, "heatmap_{}.png".format(i)), heatmap_o)
        cv2.imwrite(os.path.join(vis_path, "paf_{}.png".format(i)), paf_o)


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


