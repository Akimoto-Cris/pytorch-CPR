import numpy as np
import cv2
from data_process.process_utils import denormalize, resize_hm
WAIT_TIME = None

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def apply_heatmap(img, heatmap):
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    return img


def visualize_masks(img, ignore_mask):
    if ignore_mask.sum() > 0:
        # cv2.namedWindow('masked_img')
        # cv2.startWindowThread()
        cv2.imshow('masked_img', apply_mask(img.copy(), ignore_mask, color=(0, 0, 1)))
        cv2.waitKey()


def visualize_heatmap(img, heat_maps, displayname = 'heatmaps'):
    heat_maps = heat_maps.max(axis=0)
    heat_maps = (heat_maps/heat_maps.max() * 255.).astype('uint8')
    img = img.copy()
    colored = cv2.applyColorMap(heat_maps, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    # cv2.namedWindow(displayname)
    # cv2.startWindowThread()
    cv2.imshow(displayname, img)
    cv2.waitKey()


def visualize_keypoints(img, keypoints, body_part_map):
    img = img.copy()
    keypoints = keypoints.astype('int32')
    for person in range(keypoints.shape[0]):
        for i in range(keypoints.shape[0]):
            x = keypoints[person, i, 0]
            y = keypoints[person, i, 1]
            if keypoints[person, i, 2] > 0:
                cv2.circle(img, (x, y), 3, (0, 1, 0), -1)
        for part in body_part_map:
            keypoint_1 = keypoints[person, part[0]]
            x_1 = keypoint_1[0]
            y_1 = keypoint_1[1]
            keypoint_2 = keypoints[person, part[1]]
            x_2 = keypoint_2[0]
            y_2 = keypoint_2[1]
            if keypoint_1[2] > 0 and keypoint_2[2] > 0:
                cv2.line(img, (x_1, y_1), (x_2, y_2), (1, 0, 0), 2)
    # cv2.namedWindow('keypoints')
    # cv2.startWindowThread()
    cv2.imshow('keypoints', img)
    cv2.waitKey()


def visualize_paf(img, pafs):
    img = img.copy()
    paf = (pafs[:,0,:,:] > 1e-8).astype('bool') | (pafs[:,0,:,:] < -1e-8).astype('bool')
    paf = (np.abs(paf).max(axis=0)*255).astype('uint8')
    #paf = (pafs[:,:,:].sum(axis=0)*255).astype('uint8')
    colored = cv2.applyColorMap(paf, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    # cv2.namedWindow('pafs')
    # cv2.startWindowThread()
    cv2.imshow('pafs', img)
    cv2.waitKey()


def visualize_output_single(img, heatmap_t, paf_t, ignore_mask_t, heatmap_o, paf_o):
    img = (denormalize(img) * 255).astype('uint8')
    heatmap_o = resize_hm(heatmap_o, (img.shape[1], img.shape[0]))
    paf_o = resize_hm(paf_o, (img.shape[1], img.shape[0]))
    heatmap_t = resize_hm(heatmap_t, (img.shape[1], img.shape[0]))
    paf_t = resize_hm(paf_t, (img.shape[1], img.shape[0]))
    ignore_mask = cv2.resize(ignore_mask_t, (img.shape[1], img.shape[0]))
    cv2.namedWindow('heatmap_out')
    cv2.namedWindow('heatmap_target')
    cv2.namedWindow('paf_out')
    cv2.namedWindow('paf_target')
    cv2.namedWindow('masked_img')
    cv2.startWindowThread()
    visualize_heatmap(img, heatmap_o, 'heatmap_out')
    visualize_heatmap(img, heatmap_t, 'heatmap_target')
    visualize_heatmap(img, paf_o, 'paf_out')
    visualize_heatmap(img, paf_t, 'paf_target')
    visualize_masks(img, ignore_mask)


def visualize_output(input_, heatmaps_t, pafs_t, ignore_masks_t, outputs):
    n_images = input_.shape[0]
    cv2.namedWindow('heatmap_out')
    cv2.namedWindow('heatmap_target')
    cv2.namedWindow('paf_out')
    cv2.namedWindow('paf_target')
    cv2.namedWindow('masked_img')
    cv2.startWindowThread()
    for i in range(n_images):
        img = input_[i].copy()
        heatmap_o = outputs[0][i].copy()
        paf_o = outputs[1][i].copy()
        heatmap_t = heatmaps_t[i].copy()
        paf_t = pafs_t[i].copy()
        ignore_mask_t = ignore_masks_t[i].copy()
        visualize_output_single(img, heatmap_t, paf_t, ignore_mask_t, heatmap_o, paf_o)