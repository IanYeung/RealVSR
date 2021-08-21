import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util

logger = logging.getLogger('base')


class RealVSRDataset(data.Dataset):
    """
    Reading the training REDS dataset
    key example: 000_00000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    """

    def __init__(self, opt):
        super(RealVSRDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info(
            'Temporal augmentation interval list: [{}], with random reverse is {}.'.
            format(','.join(str(x) for x in opt['interval_list']), self.random_reverse)
        )

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        
        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError('Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')

        # remove some sequences for testing
        self.paths_GT = [
            v for v in self.paths_GT if v.split('_')[0] not in
            ['008', '026', '029', '031', '042', '055', '058', '077', '105', '113',
             '132', '135', '146', '155', '161', '167', '173', '175', '180', '181',
             '189', '194', '195', '226', '232', '237', '241', '242', '247', '256',
             '268', '275', '293', '309', '358', '371', '372', '379', '383', '401',
             '409', '413', '426', '438', '448', '471', '478', '484', '490', '498']
        ]
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False, meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()

        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 49:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval)
                )
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval)
                )
            name_b = '{:05d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval > 49) or \
                  (center_frame_idx - self.half_N_frames * interval < 0):
                center_frame_idx = random.randint(0, 49)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval)
            )
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:05d}'.format(neighbor_list[self.half_N_frames])

        assert len(neighbor_list) == self.opt['N_frames'], \
            'Wrong length of neighbor list: {}'.format(len(neighbor_list))

        #### get the GT image (as the center frame)
        GT_size_tuple = (3, 1024, 512)
        if self.data_type == 'lmdb':
            img_GT = util.read_img(self.GT_env, key, GT_size_tuple)
        else:
            img_GT = util.read_img(None, osp.join(self.GT_root, name_a, name_b + '.png'))
        if self.opt['color']:  # change color space if necessary
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        #### get LQ images
        LQ_size_tuple = (3, 1024, 512)
        img_LQ_l = []
        for v in neighbor_list:
            img_LQ_path = osp.join(self.LQ_root, name_a, '{:05d}.png'.format(v))
            if self.data_type == 'lmdb':
                img_LQ = util.read_img(self.LQ_env, '{}_{:05d}'.format(name_a, v), LQ_size_tuple)
            else:
                img_LQ = util.read_img(None, img_LQ_path)
            if self.opt['color']:  # change color space if necessary
                img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR = int(rnd_h * scale)
                rnd_w_HR = int(rnd_w * scale)
                img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (0, 3, 1, 2)))).float()

        return {'LQs': img_LQs, 'GT': img_GT, 'key': key}

    def __len__(self):
        return len(self.paths_GT)


class RealVSRAllPairDataset(data.Dataset):
    """
    Reading the training REDS dataset
    key example: 000_00000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    """

    def __init__(self, opt):
        super(RealVSRAllPairDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info(
            'Temporal augmentation interval list: [{}], with random reverse is {}.'.
                format(','.join(str(x) for x in opt['interval_list']), self.random_reverse)
        )

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs

        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError('Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')

        # remove some sequences for testing
        if opt['remove_list']:
            self.remove_list = pickle.load(open(opt['remove_list'], 'rb'))
            self.paths_GT = [v for v in self.paths_GT if v.split('_')[0] not in self.remove_list]
            logger.info('Remove sequences: {}'.format(self.remove_list))
        else:
            logger.info('Using all sequences for training.')
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False, meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()

        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 49:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval)
                )
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval)
                )
            name_b = '{:05d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval > 49) or \
                    (center_frame_idx - self.half_N_frames * interval < 0):
                center_frame_idx = random.randint(0, 49)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval)
            )
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:05d}'.format(neighbor_list[self.half_N_frames])

        assert len(neighbor_list) == self.opt['N_frames'], \
            'Wrong length of neighbor list: {}'.format(len(neighbor_list))

        #### get the GT image (as the center frame)
        GT_size_tuple = (3, 1024, 512)
        img_GT_l = []
        for v in neighbor_list:
            img_GT_path = osp.join(self.GT_root, name_a, '{:05d}.png'.format(v))
            if self.data_type == 'lmdb':
                img_GT = util.read_img(self.GT_env, '{}_{:05d}'.format(name_a, v), GT_size_tuple)
            else:
                img_GT = util.read_img(None, img_GT_path)
            if self.opt['color']:  # change color space if necessary
                img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
            img_GT_l.append(img_GT)

        #### get LQ images
        LQ_size_tuple = (3, 1024, 512)
        img_LQ_l = []
        for v in neighbor_list:
            img_LQ_path = osp.join(self.LQ_root, name_a, '{:05d}.png'.format(v))
            if self.data_type == 'lmdb':
                img_LQ = util.read_img(self.LQ_env, '{}_{:05d}'.format(name_a, v), LQ_size_tuple)
            else:
                img_LQ = util.read_img(None, img_LQ_path)
            if self.opt['color']:  # change color space if necessary
                img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale
                rnd_h_LQ = random.randint(0, max(0, H - LQ_size))
                rnd_w_LQ = random.randint(0, max(0, W - LQ_size))
                img_LQ_l = [v[rnd_h_LQ:rnd_h_LQ + LQ_size, rnd_w_LQ:rnd_w_LQ + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR = int(rnd_h_LQ * scale)
                rnd_w_HR = int(rnd_w_LQ * scale)
                img_GT_l = [v[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :] for v in img_GT_l]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_GT_l]

            # augmentation - flip, rotate
            rlt = [*img_LQ_l, *img_GT_l]
            rlt = util.augment(rlt, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[:len(neighbor_list)]
            img_GT_l = rlt[len(neighbor_list):]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        img_GTs = np.stack(img_GT_l, axis=0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GTs = img_GTs[:, :, :, [2, 1, 0]]
            img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (0, 3, 1, 2)))).float()

        return {'LQs': img_LQs, 'GT': img_GTs, 'key': key}

    def __len__(self):
        return len(self.paths_GT)


if __name__ == '__main__':
    pass