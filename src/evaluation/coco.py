import os

import cv2
import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np

from .coco_process_utils import clean_annot, get_ignore_mask, get_heatmap, get_paf, get_keypoints, FLIP_INDICES
from .process_utils import flip, resize, color_augment, resize_hm_paf, normalize, affine_augment


class CocoDataSet(data.Dataset):
    def __init__(self, data_path, opt, split='train'):
        self.coco_year = 2017
        self.coco = COCO(
            os.path.join(data_path, 'annotations/person_keypoints_{}{}.json'.format(split, self.coco_year)))
        self.split = split
        self.data_path = data_path
        self.do_augment = split == 'train'
        # self.do_augment = split == False

        # load annotations that meet specific standards
        self.indices = clean_annot(self.coco, data_path, split)
        self.img_dir = os.path.join(data_path, split + str(self.coco_year))
        self.opt = opt
        if split + "_data" in self.opt[split].keys():   
            self.indices = self.indices[: min(self.opt[self.split][self.split + "_data"], len(self.indices))]
        print('Loaded {} images for {}'.format(len(self.indices), split))


    def get_item_raw(self, index, to_resize=True):
        index = self.indices[index]
        anno_ids = self.coco.getAnnIds(index)
        annots = self.coco.loadAnns(anno_ids)
        img_path = os.path.join(self.img_dir, self.coco.loadImgs([index])[0]['file_name'])
        img = self.load_image(img_path)
        ignore_mask = get_ignore_mask(self.coco, img, annots)
        keypoints = get_keypoints(self.coco, img, annots)
        if self.do_augment:
            img, ignore_mask, keypoints = self.augment(img, ignore_mask, keypoints, self.opt)
        if to_resize:
            img, ignore_mask, keypoints = resize(img, ignore_mask, keypoints, self.opt[self.split]["imgSize"])
        heat_map = get_heatmap(self.coco, img, keypoints, self.opt[self.split]["sigmaHM"])
        paf = get_paf(self.coco, img, keypoints, self.opt[self.split]["sigmaPAF"], self.opt[self.split]["variableWidthPAF"])

        return img, heat_map, paf, ignore_mask, keypoints

    # global
    def augment(self, img, ignore_mask, keypoints, opts):
        if np.random.random() < opts[self.split]["flipAugProb"]:
            img, ignore_mask, keypoints = flip(img, ignore_mask, keypoints, FLIP_INDICES)
        img, ignore_mask, keypoints = color_augment(img, ignore_mask, keypoints, opts["train"]["colorAugFactor"])
        rot_angle = 0
        if np.random.random() < opts[self.split]["rotAugProb"]:
            rot_angle = np.clip(np.random.randn(),-2.0,2.0) * opts["train"]["rotAugFactor"]
        img, ignore_mask, keypoints = affine_augment(img, ignore_mask, keypoints, rot_angle, opts["train"]["scaleAugFactor"])
        return img, ignore_mask, keypoints

    def __getitem__(self, index):
        img, heat_map, paf, ignore_mask, _ = self.get_item_raw(index)
        img = normalize(img)
        heat_map, paf, ignore_mask = resize_hm_paf(heat_map, paf, ignore_mask, self.opt[self.split]["hmSize"])
        return img, heat_map, paf, ignore_mask, index

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = img.astype('float32') / 255.
        return img

    def get_imgs_multiscale(self, index, scales, flip = False):
        img, heat_map, paf, ignore_mask, _ = self.get_item_raw(index, False)
        imgs = []
        for scale in scales:
            width, height = img.shape[1], img.shape[0]
            new_width, new_height = int(scale* width), int(scale*height)
            scaled_img = cv2.resize(img.copy(), (new_width, new_height))
            flip_img = cv2.flip(scaled_img, 1)
            scaled_img = normalize(scaled_img)
            imgs.append(scaled_img)
            if flip:
                imgs.append(normalize(flip_img))
        paf = paf.transpose(2, 3, 0, 1)
        paf = paf.reshape(paf.shape[0], paf.shape[1], paf.shape[2] * paf.shape[3])
        paf = paf.transpose(2, 0, 1)
        return imgs, heat_map, paf, ignore_mask

    def get_img(self, index, flip = False):
        img, heat_map, paf, ignore_mask, _ = self.get_item_raw(index, True)
        flip_img = cv2.flip(img, 1)
        img = normalize(img)
        if flip:
            img = normalize(flip_img)
        heat_map, paf, ignore_mask = resize_hm_paf(heat_map, paf, ignore_mask, self.opt[self.split]["hmSize"])
        return img, heat_map, paf, ignore_mask

    def __len__(self):
        return len(self.indices)
