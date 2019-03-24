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
        i, img_basic, heatmap_t, heatmap_avg, paf_t, paf_avg, ignore_mask_t = query
        if self.opts.vizOut:
            '''
            history.log(i, heatmap=heatmaps[-1].data.cpu().numpy()[0, 0, :heights[j]//8, :widths[j]//8], 
                        paf=pafs[-1].data.cpu().numpy()[0, 0, :heights[j]//8, :widths[j]//8])
            canvas.draw_image(history['heatmap'])
            canvas.draw_image(history['paf'])
            '''
            visualize_output_single(img_basic, heatmap_t, paf_t, ignore_mask_t, heatmap_avg, paf_avg)
        img_basic = denormalize(img_basic)
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        canvas, to_plot, candidate, subset = decode_pose(img_basic, param, heatmap_avg, paf_avg)
        append_result(self.dataset.indices[i], subset, candidate, self.outputs)
        vis_path = os.path.join(self.opts["env"]["saveDir"], 'viz')
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

    consumer = Consumer(queue, _outputs, _indices, dataloader, opts)
    consumer.start()
    dataset = dataloader.dataset
    model.eval()

    dataset_len = 100

    with tqdm(total=dataset_len) as t:
        for i in range(dataset_len):
            img, heatmap_t, paf_t, ignore_mask_t = dataset.get_img(i, flip=False)
            img_batch = np.zeros((1, 3, opts["train"]["imgSize"], opts["train"]["imgSize"]))
            img_batch[0, :, :, :] = img
            with torch.no_grad():
                imgs_torch = torch.from_numpy(img_batch).float().cuda()
                heatmaps, pafs = model(imgs_torch)
                heatmap = heatmaps[-1][0].data.cpu().numpy()
                paf = pafs[-1][0].data.cpu().numpy()
            # print(heatmap_t.shape)
            queue.put((i, img, heatmap_t, heatmap, paf_t, paf, ignore_mask_t[0]))
            t.update()
    consumer.join()
    outputs, indices = [], []
    for _ in range(100):
        outputs.append(_outputs.get())
        indices.append(_indices.get())
    print(outputs)
    return outputs, indices
