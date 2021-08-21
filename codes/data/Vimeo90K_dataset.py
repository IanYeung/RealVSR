'''
Vimeo90K dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import utils.util as util
import data.util as data_util
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


class Vimeo90kDataset(data.Dataset):
    '''
    Reading the training Vimeo90K dataset: LQ is precomputed from GT
    key example: 00001_0001 (_1, ..., _7)
    '''

    def __init__(self, opt):
        super(Vimeo90kDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info(
            'Temporal augmentation interval list: [{}], with random reverse is {}.'
            .format(','.join(str(x) for x in opt['interval_list']), self.random_reverse)
        )

        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs

        #### determine the LQ frame list
        '''
        N | frames
        1 |       4
        3 |     3,4,5
        5 |   2,3,4,5,6
        7 | 1,2,3,4,5,6,7
        '''
        self.frame_list = []
        self.center = opt['N_frames'] // 2
        for i in range(opt['N_frames']):
            self.frame_list.append(i + (9 - opt['N_frames']) // 2)

        if self.data_type == 'lmdb':
            self.paths_GT, _ = data_util.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError('Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False, meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()

        scale = self.opt['scale']
        GT_size, LQ_size = self.opt['GT_size'], self.opt['LQ_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')

        #### temporal augmentation: random reverse
        if self.random_reverse and random.random() < 0.5:
            self.frame_list.reverse()

        #### get GT image
        GT_size_tuple = (3, 256, 448)
        if self.data_type == 'lmdb':
            img_GT = data_util.read_img(self.GT_env, key + '_4', GT_size_tuple)
        else:
            img_GT = data_util.read_img(None, osp.join(self.GT_root, name_a, name_b, 'im4.png'))

        #### get LQ images
        LQ_size_tuple = (3, 256 // scale, 448 // scale) if self.LR_input else (3, 256, 448)
        img_LQ_l = []
        for v in self.frame_list:
            if self.data_type == 'lmdb':
                img_LQ = data_util.read_img(self.LQ_env, key + '_{}'.format(v), LQ_size_tuple)
            else:
                img_LQ = data_util.read_img(None, osp.join(self.LQ_root, name_a, name_b, 'im{}.png'.format(v)))
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale
                rnd_h_LQ = random.randint(0, max(0, H - LQ_size))
                rnd_w_LQ = random.randint(0, max(0, W - LQ_size))
                img_LQ_l = [v[rnd_h_LQ:rnd_h_LQ + LQ_size, rnd_w_LQ:rnd_w_LQ + LQ_size, :] for v in img_LQ_l]
                rnd_h_GT = int(rnd_h_LQ * scale)
                rnd_w_GT = int(rnd_w_LQ * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            else:
                rnd_h_LQ = random.randint(0, max(0, H - GT_size))
                rnd_w_LQ = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h_LQ:rnd_h_LQ + GT_size, rnd_w_LQ:rnd_w_LQ + GT_size, :] for v in img_LQ_l]
                rnd_h_GT = rnd_h_LQ
                rnd_w_GT = rnd_w_LQ
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = data_util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        img_LQs = np.stack(img_LQ_l, axis=0)  # stack LQ images to NHWC, N is the frame number

        #### BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (0, 3, 1, 2)))).float()

        return {'LQs': img_LQs, 'GT': img_GT, 'key': key}

    def __len__(self):
        return len(self.paths_GT)


class Vimeo90kAllPairDataset(data.Dataset):
    '''
    Reading the training Vimeo90K dataset: LQ is precomputed from GT
    key example: 00001_0001 (_1, ..., _7)
    '''

    def __init__(self, opt):
        super(Vimeo90kAllPairDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info(
            'Temporal augmentation interval list: [{}], with random reverse is {}.'
            .format(','.join(str(x) for x in opt['interval_list']), self.random_reverse)
        )

        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs

        #### determine the LQ frame list
        '''
        N | frames
        1 |       4
        3 |     3,4,5
        5 |   2,3,4,5,6
        7 | 1,2,3,4,5,6,7
        '''
        self.frame_list = []
        self.center = opt['N_frames'] // 2
        for i in range(opt['N_frames']):
            self.frame_list.append(i + (9 - opt['N_frames']) // 2)

        if self.data_type == 'lmdb':
            self.paths_GT, _ = data_util.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))
        else:
            raise ValueError('Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False, meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()

        scale = self.opt['scale']
        GT_size, LQ_size = self.opt['GT_size'], self.opt['LQ_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')

        #### temporal augmentation: random reverse
        if self.random_reverse and random.random() < 0.5:
            self.frame_list.reverse()

        #### get GT image
        GT_size_tuple = (3, 256, 448)
        img_GT_l = []
        for v in self.frame_list:
            if self.data_type == 'lmdb':
                img_GT = data_util.read_img(self.GT_env, key + '_{}'.format(v), GT_size_tuple)
            else:
                img_GT = data_util.read_img(None, osp.join(self.GT_root, name_a, name_b, 'im{}.png'.format(v)))
            img_GT_l.append(img_GT)

        #### get LQ images
        LQ_size_tuple = (3, 256 // scale, 448 // scale) if self.LR_input else (3, 256, 448)
        img_LQ_l = []
        for v in self.frame_list:
            if self.data_type == 'lmdb':
                img_LQ = data_util.read_img(self.LQ_env, key + '_{}'.format(v), LQ_size_tuple)
            else:
                img_LQ = data_util.read_img(None, osp.join(self.LQ_root, name_a, name_b, 'im{}.png'.format(v)))
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale
                rnd_h_LQ = random.randint(0, max(0, H - LQ_size))
                rnd_w_LQ = random.randint(0, max(0, W - LQ_size))
                img_LQ_l = [v[rnd_h_LQ:rnd_h_LQ + LQ_size, rnd_w_LQ:rnd_w_LQ + LQ_size, :] for v in img_LQ_l]
                rnd_h_GT = int(rnd_h_LQ * scale)
                rnd_w_GT = int(rnd_w_LQ * scale)
                img_GT_l = [v[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :] for v in img_GT_l]
            else:
                rnd_h_LQ = random.randint(0, max(0, H - GT_size))
                rnd_w_LQ = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h_LQ:rnd_h_LQ + GT_size, rnd_w_LQ:rnd_w_LQ + GT_size, :] for v in img_LQ_l]
                rnd_h_GT = rnd_h_LQ
                rnd_w_GT = rnd_w_LQ
                img_GT_l = [v[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :] for v in img_GT_l]

            # augmentation - flip, rotate
            rlt = data_util.augment([*img_LQ_l, *img_GT_l], self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[:len(self.frame_list)]
            img_GT_l = rlt[len(self.frame_list):]

        img_LQs = np.stack(img_LQ_l, axis=0)  # stack LQ images to NHWC, N is the frame number
        img_GTs = np.stack(img_GT_l, axis=0)  # stack GT images to NHWC, N is the frame number

        #### BGR to RGB, HWC to CHW, numpy to tensor
        img_GTs = img_GTs[:, :, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (0, 3, 1, 2)))).float()

        return {'LQs': img_LQs, 'GT': img_GTs, 'key': key}

    def __len__(self):
        return len(self.paths_GT)


if __name__ == '__main__':
    pass