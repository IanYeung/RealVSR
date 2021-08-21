import os
import glob
import time
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.archs.TOF_arch as TOF_arch
import models.archs.RCAN_arch as RCAN_arch
import models.archs.EDVR_arch as EDVR_arch
import models.archs.TDAN_arch as TDAN_arch
import models.archs.FSTRN_arch as FSTRN_arch


def center_crop(img, tar_h, tar_w):
    N, C, inp_h, inp_w = img.shape  # LQ size
    # center crop
    start_h = int((inp_h - tar_h) / 2)
    start_w = int((inp_w - tar_w) / 2)
    img = img[:, :, start_h:start_h + tar_h, start_w:start_w + tar_w]
    return img


def main():
    #################
    # configurations
    #################
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_mode = 'RealSeq'

    # TODO: Modify the configurations here
    # model
    N_ch = 3
    N_in = 3
    model = 'EDVR'
    model_name = '001_EDVR_NoUp_woTSA_scratch_lr1e-4_150k_RealVSR_3frame_WiCutBlur_YCbCr_LapPyr+GW'
    model_path = '../experiments/pretrained_models/{}.pth'.format(model_name)
    # dataset
    read_folder = '/home/yangxi/datasets/RealVSR/test_frames_YCbCr'
    save_folder = '/home/yangxi/datasets/RealVSR/results/{}/{}'.format(data_mode, model_name)
    # color mode
    color = 'YCbCr'
    # device
    device = torch.device('cuda')

    if model == 'RCAN':
        model = RCAN_arch.RCAN(num_in_ch=N_ch, num_out_ch=N_ch, num_frames=N_in, num_feat=64,
                               num_group=5, num_block=2, squeeze_factor=16, upscale=1, res_scale=1)
    elif model == 'FSTRN':
        model = FSTRN_arch.FSTRN(k=3, nf=64, scale=1, nframes=N_in)
    elif model == 'TOF':
        model = TOF_arch.TOF(nframes=3, K=3, in_nc=N_ch, out_nc=N_ch, nf=64, nb=10, upscale=1)
    elif model == 'TDAN':
        model = TDAN_arch.TDAN(channel=N_ch, nf=64, nframes=N_in, groups=8, scale=1)
    elif model == 'EDVR':
        model = EDVR_arch.EDVR_NoUp(nf=64, nc=N_ch, nframes=N_in, groups=8, front_RBs=5, back_RBs=10,
                                    predeblur=False, HR_in=False, w_TSA=False)
    else:
        raise ValueError()

    # temporal padding mode
    padding = 'new_info'  # different from the official setting
    save_imgs = True
    
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, read_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))

    subfolder_l = sorted(glob.glob(os.path.join(read_folder, '*')))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    subfolder_name_l = []

    # for each sub-folder
    for subfolder in subfolder_l:
        subfolder_name = subfolder.split('/')[-1]
        subfolder_name_l.append(subfolder_name)
        save_subfolder = os.path.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(os.path.join(subfolder, '*')))
        max_idx = len(img_path_l)

        if save_imgs:
            util.mkdirs(save_subfolder)
        logger.info('Folder {} '.format(subfolder_name))
        time_list = []

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            img_l = []
            for idx in select_idx:
                inp_path = os.path.join(subfolder, '{:05d}.png'.format(idx + 1))
                img = data_util.read_img(None, inp_path)[:, :, [2, 1, 0]]
                img_l.append(img)
            imgs = np.stack(img_l, axis=0)
            imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
            imgs = center_crop(imgs, tar_h=imgs.shape[2], tar_w=imgs.shape[3])
            imgs = imgs.unsqueeze(0).to(device)
            t_1 = time.time()
            output = util.single_forward(model, imgs)
            t_2 = time.time()
            logger.info('Processing: {}, Time: {:.4f} s'.format(img_name, t_2 - t_1))
            time_list.append(t_2 - t_1)
            output = output.squeeze(0).cpu().numpy()
            output = np.transpose(output, (1, 2, 0))
            output = data_util.ycbcr2bgr(output)
            output = (np.clip(output, 0, 1) * 255.).round().astype(np.uint8)
            # save imgs
            if save_imgs:
                cv2.imwrite(os.path.join(save_subfolder, '{}.png'.format(img_name)), output)
        logger.info('Average inference time: {:.4f} s'.format(sum(time_list) / len(time_list)))


if __name__ == '__main__':
    main()
