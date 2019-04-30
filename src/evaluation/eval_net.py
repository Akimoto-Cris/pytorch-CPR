import os
import time
import numpy as np
import torch, json
from data_process.process_utils import resize_hm, denormalize
from visualization.visualize import visualize_output_single
from .post import plot_pose_pdf, decode_pose, append_result
from openpose_plus.inference.post_process import decode_pose as openpose_decode_pose
import torch.multiprocessing as mp

# Typical evaluation is done on multi-scale and average across all evals is taken as output
# These reduce the quantization error in the model
def eval_net(data_loader, model, opts, ids_in_ckpt=[], fraction=[0, 0.3]):
    model.eval()
    dataset = data_loader.dataset
    scales = [1., 0.5, 1.25, 1.5, 1.75, 2.0]
    # scales = [1.]
    assert (scales[0]==1)
    n_scales = len(scales)
    dataset_len = len(dataset)
    # keypoints_list = []
    runtimes = []

    with torch.no_grad():
        for i in range(int(dataset_len * fraction[0]), int(dataset_len * fraction[1])):
            if dataset.indices[i] in ids_in_ckpt:
                print(f"skip {i}th image of image_id {dataset.indices[i]}.")
                continue

            print(i)
            start = time.time()
            # imgs, heatmap_t, paf_t, ignore_mask_t = dataset.get_imgs_multiscale(i, scales, flip=False, to_resize=False)
            if opts["to_test"]:
                imgs, orig_shape = dataset.get_imgs_multiscale_resize(i, scales, flip=False, resize_factor=1)
                heatmap_t = np.zeros((opts["model"]["nJoints"], opts["test"]["hmSize"], opts["test"]["hmSize"]))
                paf_t = np.zeros((opts["model"]["nLimbs"], opts["test"]["hmSize"], opts["test"]["hmSize"]))
                ignore_mask_t = np.zeros((opts["test"]["hmSize"], opts["test"]["hmSize"]))
            else:
                imgs, heatmap_t, paf_t, ignore_mask_t, orig_shape = dataset.get_imgs_multiscale_resize(i, scales,
                                                                                                       flip=False,
                                                                                                       resize_factor=1)
            resize_factors = tuple([orig_shape[i] / imgs[0].shape[i + 1] for i in [1, 0]])
            # resize_factors = (1, 1)
            print(f"input_size:{imgs[0].shape}, origin_size:{orig_shape}")
            n_imgs = len(imgs)
            assert(n_imgs == n_scales)
            heights = list(map(lambda x: x.shape[1], imgs))
            widths = list(map(lambda x: x.shape[2], imgs))
            max_h, max_w = max(heights), max(widths)
            imgs_np = np.zeros((n_imgs, 3, max_h, max_w))
            for j in range(n_imgs):
                img = imgs[j]
                h, w = img.shape[1], img.shape[2]
                imgs_np[j,:,:h,:w] = img
            img_basic = imgs[0]

            heatmap_avg_lst = []
            paf_avg_lst = []
            print("first loop", time.time() - start)
            for j in range(0, n_imgs):
                imgs_torch = torch.from_numpy(imgs_np[j:j+1]).float().cuda()
                heatmaps, pafs = model(imgs_torch)
                heatmap = heatmaps[-1].data.cpu().numpy()[0, :, :heights[j]//8, :widths[j]//8]
                paf = pafs[-1].data.cpu().numpy()[0, :, :heights[j]//8, :widths[j]//8]
                heatmap = resize_hm(heatmap, (widths[0], heights[0]))
                paf = resize_hm(paf, (widths[0], heights[0]))
                heatmap_avg_lst += [heatmap]
                paf_avg_lst += [paf]
            heatmap_avg = sum(heatmap_avg_lst)/n_imgs
            paf_avg = sum(paf_avg_lst)/n_imgs
            print("second loop", time.time() - start)
            if opts["viz"]["vizOut"]:
                visualize_output_single(img_basic, heatmap_t, paf_t, ignore_mask_t, heatmap_avg, paf_avg)
            img_basic = denormalize(img_basic)
            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            vis_path = os.path.join(opts["env"]["saveDir"], "val" if not opts["to_test"] else "test" + '_viz')

            def native():
                candidates, subset = decode_pose(img_basic, param, heatmap_avg, paf_avg)
                plot_pose_pdf(img_basic, candidates, subset, os.path.join(vis_path, f"{i}.pdf"))
                outputs = append_result(dataset.indices[i], subset, candidates)
                return outputs

            def openpose_inf():
                outputs = openpose_decode_pose(img_basic, param, heatmap_avg, paf_avg, opts,
                                               dataset.indices[i], os.path.join(vis_path, f"{i}.pdf"),
                                               resize_shape=None)
                return outputs

            if not os.path.exists(vis_path):
                os.makedirs(vis_path)

            outputs = native() # native()
            for out in outputs:
                out['keypoints'][::3] = [i * resize_factors[0] for i in out['keypoints'][::3]]
                out['keypoints'][1::3] = [i * resize_factors[1] for i in out['keypoints'][1::3]]
            final = time.time()-start
            print("both loops took ", final)

            yield outputs, dataset.indices[i]
