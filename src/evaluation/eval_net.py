import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
from data_process.process_utils import resize_hm, denormalize
from visualization.visualize import visualize_output_single
from .post import decode_pose, append_result
import torch.multiprocessing as mp


# IO operations
class Consumer(mp.Process):
    def __init__(self, queue, outputs, indices, dataloader, opts):
        super(Consumer, self).__init__()
        self.data = queue
        self.dataset = dataloader.dataset
        self.opts = opts
        self.outputs = outputs
        self.indices = indices
        self.dataset_len = 100  # len(dataset)

    def do_some_works(self, query):
        i, img_basic, heatmap_t, heatmap_avg, paf_t, paf_avg, ignore_mask_t, heights, widths = query
        if self.opts.vizOut:
            '''history.log(i, heatmap=heatmaps[-1].data.cpu().numpy()[0, 0, :heights[j]//8, :widths[j]//8], 
                        paf=pafs[-1].data.cpu().numpy()[0, 0, :heights[j]//8, :widths[j]//8])
            canvas.draw_image(history['heatmap'])
            canvas.draw_image(history['paf'])'''
            visualize_output_single(img_basic, heatmap_t, paf_t, ignore_mask_t, heatmap_avg, paf_avg)
        img_basic = denormalize(img_basic)
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        canvas, to_plot, candidate, subset = decode_pose(img_basic, param, heatmap_avg, paf_avg)
        append_result(self.dataset.indices[i], subset, candidate, self.outputs)
        vis_path = os.path.join(self.opts.saveDir, 'viz')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        cv2.imwrite(vis_path + '/{}.png'.format(i), to_plot)

        self.indices.put(self.dataset.indices[:self.dataset_len])

    def run(self):
        while True:
            query = self.data.get()
            if not query:
                self.data.task_done()
                break
            self.do_some_works(query)
            self.data.task_done()
        return


class Results():
    def __init__(self):
        self.outputs = []
        self.indices = []

    def append_output(self, one_result):
        self.outputs.append(one_result)

    def append_index(self, ind):
        self.indices.append(ind)


def eval_net(dataloader, model, opts):
    torch.manual_seed(23)

    queue = mp.JoinableQueue()
    _outputs, _indices = mp.Queue(), mp.Queue()

    # decoder = Decoder(queue, dataloader, model)
    consumer = Consumer(queue, _outputs, _indices, dataloader, opts)

    dataset = dataloader.dataset
    model.eval()
    scales = [1., 0.75, 1.25]
    assert (scales[0] == 1)
    n_scales = len(scales)
    dataset_len = 100  # len(dataset)
    # history = hl.History()
    # canvas = hl.Canvas()

    consumer.start()

    with tqdm(total=dataset_len) as t:
        for i in range(dataset_len):
            imgs, heatmap_t, paf_t, ignore_mask_t = dataset.get_imgs_multiscale(i, scales, flip=False)
            n_imgs = len(imgs)
            assert (n_imgs == n_scales)
            heights = list(map(lambda x: x.shape[1], imgs))
            widths = list(map(lambda x: x.shape[2], imgs))
            max_h, max_w = max(heights), max(widths)
            imgs_np = np.zeros((n_imgs, 3, max_h, max_w))
            for j in range(n_imgs):
                img = imgs[j]
                h, w = img.shape[1], img.shape[2]
                imgs_np[j, :, :h, :w] = img
            img_basic = imgs[0]
            heatmap_avg = np.zeros(heatmap_t.shape)
            paf_avg = np.zeros(paf_t.shape)
            for j in range(0, n_imgs):
                with torch.no_grad():
                    imgs_torch = torch.from_numpy(imgs_np[j:j + 1]).float().cuda()
                    heatmaps, pafs = model(imgs_torch)
                    heatmap = heatmaps[-1].data.cpu().numpy()[0, :, :heights[j] // 8, :widths[j] // 8]
                    paf = pafs[-1].data.cpu().numpy()[0, :, :heights[j] // 8, :widths[j] // 8]
                    heatmap = resize_hm(heatmap, (widths[0], heights[0]))
                    paf = resize_hm(paf, (widths[0], heights[0]))
                    heatmap_avg += heatmap / n_imgs
                    paf_avg += paf / n_imgs
            queue.put((i, img_basic, heatmap_t, heatmap_avg, paf_t, paf_avg, ignore_mask_t, heights, widths))
            t.update()

    consumer.join()

    outputs, indices = [], []
    for _ in range(100):
        outputs.append(_outputs.get())
        indices.append(_indices.get())
    print(outputs)
    return outputs, indices
