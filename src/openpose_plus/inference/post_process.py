from enum import Enum
import numpy as np
from .common import CocoPart, draw_humans
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from cv2 import resize
import matplotlib.pyplot as plt
from data_process.coco_process_utils import BODY_PARTS

class PARTS_wo_NECK(Enum):
    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16

    def decode_from_openpose(self, part_name):
        pass
"""
def get_peak_map(origin, name):
    smoothed = _gauss_smooth(origin)
    max_pooled = tf.nn.pool(smoothed, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    return tf.where(tf.equal(smoothed, max_pooled), smoothed, tf.zeros_like(origin), name)"""

def get_peak_map(param, img):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image and one-shot peak map with the same shape of input image
    """

    '''
    footprint = array([[False,  True, False],
                       [ True,  True,  True],
                       [False,  True, False]], dtype=bool)
    '''
    ret = np.zeros(img.shape)
    for i in range(img.shape[-1]):
        peaks_binary = (maximum_filter(img[:, :, i],
                                       footprint=generate_binary_structure(2, 1)       # mask within the filter
                                       ) == img[:, :, i])
        peaks = np.array(np.nonzero(peaks_binary)).T       # nonzero return array-like of INDICES of nonzero entries
        for peak in peaks:
            # print(tuple(peak) + (i,))
            ret[tuple(peak) + (i,)] = img[tuple(peak) + (i,)]
    return ret

def upsample(img, upsample_size):
    if isinstance(upsample_size, list):
        upsample_size = tuple(upsample_size)
    elif isinstance(upsample_size, int):
        upsample_size = tuple([upsample_size] * 2)
    elif not isinstance(upsample_size, tuple):
        raise TypeError("Upsample_size should be tuple")
    return resize(img, upsample_size)

class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()

class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()

def estimate_paf(peaks, heat_mat, paf_mat):
    from .pafprocess import pafprocess  # TODO: don't depend on it
    pafprocess.process_paf(peaks.astype(np.float32), heat_mat.astype(np.float32), paf_mat.astype(np.float32))

    humans = []
    for human_id in range(pafprocess.get_num_humans()):
        human = Human([])
        is_added = False

        for part_idx in range(18):
            c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
            if c_idx < 0:
                continue

            is_added = True
            human.body_parts[part_idx] = BodyPart('%d-%d' % (human_id, part_idx), part_idx,
                                                  float(pafprocess.get_part_x(c_idx)),
                                                  float(pafprocess.get_part_y(c_idx)),
                                                  pafprocess.get_part_score(c_idx))

        if is_added:
            print(human)
            human.score = pafprocess.get_score(human_id)
            humans.append(human)
    return humans

class PostProcessor(object):
    __slots__ = ('n_joins', 'n_connections', 'f_height', 'f_width', 'param', 'origin_size')

    def __init__(self, opt, param={'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}):
        """Create the PostProcessor.

        Parameters:
            origin_size : (height, width) of the input image
            feature_size : (height', width') of the the feature maps
        """
        assert isinstance(opt, dict)
        self.n_joins = opt["model"]["nJoints"]
        self.n_connections = opt["model"]["nLimbs"]
        self.param = param
        self.f_height = self.f_width = opt["val" if not opt["to_test"] else "test"]["hmSize"]
        self.origin_size = opt["val" if not opt["to_test"] else "test"]["imgSize"]

    def __call__(self, heatmap_input, pafmap_input):
        p = [1, 2, 0]
        heatmap_input = heatmap_input.transpose(p)
        pafmap_input = pafmap_input.transpose(p)

        heatmap = upsample(heatmap_input, self.origin_size)
        peaks = get_peak_map(self.param, heatmap)
        pafmap= upsample(pafmap_input, self.origin_size)

        humans = estimate_paf(peaks, heatmap, pafmap)
        return humans, heatmap, pafmap


def decode_human(image_id, humans, img_size):
    """
    Convert Internal Structured Human to COCO evaluation format
    :param humans:
    :param img_size:
    :return:
    """
    local2openplus = {member.value: CocoPart.__members__[k].value
                      for k, member in PARTS_wo_NECK.__members__.items()}
    n_parts = len(PARTS_wo_NECK.__members__)

    results = []
    for human in humans:
        one_result = {
            "image_id": image_id,
            "category_id": 1,
            "keypoints": [],
            "score": 0
        }
        keypoints = np.zeros((n_parts, 3))
        for i in range(n_parts):
            ii = local2openplus[i]
            if ii not in human.body_parts.keys():
                continue
            body_part = human.body_parts[ii]
            keypoints[i, 0] = int(body_part.x * img_size[1] + 0.5)
            keypoints[i, 1] = int(body_part.y * img_size[0] + 0.5)
            keypoints[i, 2] = 1
        one_result['keypoints'] = list(keypoints.reshape(n_parts * 3))
        one_result['score'] = human.score
        results += [one_result]
    return results

def decode_pose(img_orig, param, heatmaps, pafs, opt, image_id, save_path=None):
    humans, _, _ = PostProcessor(opt, param=param)(heatmaps, pafs)
    outputs = decode_human(image_id, humans, img_orig.shape[:2])
    if save_path:
        draw_humans(img_orig, humans, save_path)
    return outputs