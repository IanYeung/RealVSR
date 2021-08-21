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


def main():
    #################
    # configurations
    #################
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_mode = 'RealVSR'

    # TODO: Modify the configurations here
    # model
    N_ch = 3
    N_in = 3
    model = 'EDVR'
    model_name = '001_EDVR_NoUp_woTSA_scratch_lr1e-4_150k_RealVSR_3frame_WiCutBlur_YCbCr_LapPyr+GW'
    model_path = '../experiments/pretrained_models/{}.pth'.format(model_name)
    # dataset
    read_folder = '/home/xiyang/Datasets/RealVSR/release_v2/LQ_YCbCr_test'
    save_folder = '/home/xiyang/Datasets/RealVSR/results/{}/{}'.format(data_mode, model_name)
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

    #### evaluation
    flip_test = False
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate

    # temporal padding mode
    padding = 'replicate'  # different from the official setting
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

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    avg_ssim_l, avg_ssim_center_l, avg_ssim_border_l = [], [], []
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

        #### read LR images
        imgs = data_util.read_img_seq(subfolder, color=color)
        #### read GT images
        img_GT_l = []
        subfolder_GT = os.path.join(subfolder.replace('/LQ_YCbCr_test/', '/GT_YCbCr_test/'), '*')
        for img_GT_path in sorted(glob.glob(subfolder_GT)):
            if color == 'YCbCr':
                tmp_img = data_util.read_img(None, img_GT_path)[:, :, [2, 1, 0]]
            else:
                tmp_img = data_util.read_img(None, img_GT_path)
            img_GT_l.append(tmp_img)

        avg_psnr, avg_psnr_border, avg_psnr_center = 0, 0, 0
        avg_ssim, avg_ssim_border, avg_ssim_center = 0, 0, 0
        N_border, N_center = 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            # get input images
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
            output = util.single_forward(model, imgs_in)

            if color == 'YCbCr':
                output = util.tensor2img(output.squeeze(0), out_type=np.float32, reverse_channel=False)
                img = (np.clip(data_util.ycbcr2bgr(output), 0, 1) * 255.).round().astype(np.uint8)
                # save imgs
                if save_imgs:
                    cv2.imwrite(os.path.join(save_subfolder, '{}.png'.format(img_name)), img)
            else:
                output = util.tensor2img(output.squeeze(0), out_type=np.uint8, reverse_channel=True)
                img = output
                # save imgs
                if save_imgs:
                    cv2.imwrite(os.path.join(save_subfolder, '{}.png'.format(img_name)), img)

            #### calculate PSNR and SSIM
            if color == 'YCbCr':
                output = output / 255.
                GT = np.copy(img_GT_l[img_idx])
                GT = np.squeeze(GT)
                output, GT = util.crop_border([output, GT], crop_border)
                output = (output * 255.0).round().astype(np.uint8)
                GT = (GT * 255.0).round().astype(np.uint8)
                crt_psnr = util.calculate_psnr(output[:, :, 0], GT[:, :, 0])
                crt_ssim = util.calculate_ssim(output[:, :, 0], GT[:, :, 0])
            else:
                output = output / 255.
                GT = np.copy(img_GT_l[img_idx])
                GT = np.squeeze(GT)
                output, GT = util.crop_border([output, GT], crop_border)
                crt_psnr = util.calculate_psnr(output * 255, GT * 255)
                crt_ssim = util.calculate_ssim(output * 255, GT * 255)
            logger.info('{:3d} - {:25} \tPSNR: {:.2f} dB \tSSIM: {:.4f}'.
                        format(img_idx + 1, img_name, crt_psnr, crt_ssim))

            if border_frame <= img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                avg_ssim_center += crt_ssim
                N_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                avg_ssim_border += crt_ssim
                N_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_ssim = (avg_ssim_center + avg_ssim_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_ssim_center = avg_ssim_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_ssim_border = 0 if N_border == 0 else avg_ssim_border / N_border

        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)
        avg_ssim_l.append(avg_ssim)
        avg_ssim_center_l.append(avg_ssim_center)
        avg_ssim_border_l.append(avg_ssim_border)

        logger.info('Folder {} - Average PSNR: {:.2f} dB for {} frames; '
                    'Center PSNR: {:.2f} dB for {} frames; '
                    'Border PSNR: {:.2f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_psnr_border, N_border))
        logger.info('Folder {} - Average SSIM: {:.4f} for {} frames; '
                    'Center SSIM: {:.4f} for {} frames; '
                    'Border SSIM: {:.4f} for {} frames.'.format(subfolder_name, avg_ssim,
                                                                (N_center + N_border),
                                                                avg_ssim_center, N_center,
                                                                avg_ssim_border, N_border))

    logger.info('################ Tidy Outputs ################')
    for name, psnr, psnr_center, psnr_border in zip(subfolder_name_l, avg_psnr_l,
                                                    avg_psnr_center_l, avg_psnr_border_l):
        logger.info('Folder {} - Average PSNR: {:.2f} dB. '
                    'Center PSNR: {:.2f} dB. '
                    'Border PSNR: {:.2f} dB.'.format(name, psnr, psnr_center, psnr_border))
    for name, ssim, ssim_center, ssim_border in zip(subfolder_name_l, avg_ssim_l,
                                                    avg_ssim_center_l, avg_ssim_border_l):
        logger.info('Folder {} - Average SSIM: {:.4f}. '
                    'Center SSIM: {:.4f}. '
                    'Border SSIM: {:.4f}.'.format(name, ssim, ssim_center, ssim_border))
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, read_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.2f} dB for {} clips. '
                'Center PSNR: {:.2f} dB. Border PSNR: {:.2f} dB.'.format(
        sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
        sum(avg_psnr_center_l) / len(avg_psnr_center_l),
        sum(avg_psnr_border_l) / len(avg_psnr_border_l)))
    logger.info('Total Average SSIM: {:.4f} for {} clips. '
                'Center SSIM: {:.4f}. Border SSIM: {:.4f}.'.format(
        sum(avg_ssim_l) / len(avg_ssim_l), len(subfolder_l),
        sum(avg_ssim_center_l) / len(avg_ssim_center_l),
        sum(avg_ssim_border_l) / len(avg_ssim_border_l)))


if __name__ == '__main__':
    main()
