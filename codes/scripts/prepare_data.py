import os
import sys
import glob
import cv2
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from shutil import copy, copytree

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import data.util as data_util
    import utils.util as util
except ImportError:
    pass


def rgb2ycbcr(src_root, dst_root, only_y=True):
    util.mkdir(dst_root)
    src_img_paths = sorted(glob.glob(os.path.join(src_root, '*.png')))

    for src_img_path in src_img_paths:
        print(src_img_path)
        src_img = cv2.imread(src_img_path)
        dst_img = data_util.bgr2ycbcr(src_img, only_y=only_y)
        cv2.imwrite(os.path.join(dst_root, '{}'.format(os.path.basename(src_img_path))), dst_img[:, :, [2, 1, 0]])


def realvsr(src_root, dst_root, only_y):
    seq_paths = sorted(glob.glob(os.path.join(src_root, '*')))
    seqs = [os.path.basename(seq_path) for seq_path in seq_paths]

    for seq in seqs:
        print('Processing {}'.format(seq))
        src_img_paths = sorted(glob.glob(os.path.join(src_root, seq, '*.png')))

        for src_img_path in src_img_paths:
            src_img = cv2.imread(src_img_path)
            dst_img = data_util.bgr2ycbcr(src_img, only_y=only_y)
            util.mkdir(os.path.join(dst_root, seq))
            cv2.imwrite(os.path.join(dst_root, seq, '{}'.format(os.path.basename(src_img_path))), dst_img[:, :, [2, 1, 0]])


def vimeo90k(src_root, dst_root):
    seq_paths = sorted(glob.glob(os.path.join(src_root, '*', '*', '*.png')))

    for src_img_path in seq_paths:
        print(src_img_path)
        tmp_list = src_img_path.split('/')
        name_a, name_b, img_name = tmp_list[-3], tmp_list[-2], tmp_list[-1]
        src_img = cv2.imread(src_img_path)
        dst_img = data_util.bgr2ycbcr(src_img, only_y=False)
        util.mkdir(os.path.join(dst_root, name_a, name_b))
        cv2.imwrite(os.path.join(dst_root, name_a, name_b, img_name), dst_img[:, :, [2, 1, 0]])


def save_keys_realvsr(save_path):
    key_list = []
    for seq_idx in range(500):
        for img_idx in range(50):
            key_list.append('{:03d}_{:05d}'.format(seq_idx, img_idx))
    with open(save_path, 'wb') as f:
        pickle.dump({'keys': key_list}, f)


if __name__ == '__main__':
    pass
