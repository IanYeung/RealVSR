import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

import utils.util as util
import data.util as data_util
from metrics import calculate_psnr, calculate_ssim, calculate_niqe
from IQA_pytorch import LPIPSvgg, DISTS


def setup_logger(logger_name, log_file, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def prepare_image(image, resize=False, repeatNum=1):
    if resize and min(image.size) > 256:
        image = transforms.functional.resize(image, 256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum, 1, 1, 1)


def evaluate_psnr(model_name, GT_folder, log_file=None, color='y'):

    if log_file:
        setup_logger('base', log_file, level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
    else:
        log_file = '/home/xiyang/Results/RealVSR/PSNR_{}.log'.format(model_name)
        setup_logger('base', log_file, level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')

    subfolder_l = sorted(glob.glob(GT_folder))
    avg_psnr_l = []

    # for each sub-folder
    for subfolder in subfolder_l:
        subfolder_name = subfolder.split('/')[-1]
        logger.info(subfolder_name)
        avg_psnr = 0
        for img_idx, img_GT_path in enumerate(sorted(glob.glob(osp.join(subfolder, '[0-9]*')))):
            if color == 'y':
                img_GT = data_util.read_img(None, img_GT_path)
                img_GT = data_util.bgr2ycbcr(img_GT, only_y=True)
            else:
                img_GT = data_util.read_img(None, img_GT_path)
            # TODO: modify accordingly
            img_LQ_path = img_GT_path.replace('/GT_test/', '/test_results/{}/'.format(model_name))
            assert img_LQ_path != img_GT_path
            if color == 'y':
                img_LQ = data_util.read_img(None, img_LQ_path)
                img_LQ = data_util.bgr2ycbcr(img_LQ, only_y=True)
            else:
                img_LQ = data_util.read_img(None, img_LQ_path)

            if color == 'y':
                psnr = util.calculate_psnr(img_LQ * 255, img_GT * 255)
            else:
                psnr = util.calculate_psnr(img_LQ * 255, img_GT * 255)
            logger.info('{:3d} - {:25} \tPSNR: {:.2f} dB'.format(img_idx + 1, os.path.basename(img_LQ_path), psnr))
            avg_psnr += psnr

        avg_psnr = avg_psnr / len(subfolder_l)
        avg_psnr_l.append(avg_psnr)
    logger.info(model_name)
    logger.info('PSNR: {:.2f} dB'.format(sum(avg_psnr_l) / len(avg_psnr_l)))


def evaluate_ssim(model_name, GT_folder, log_file=None, color='y'):

    if log_file:
        setup_logger('base', log_file, level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
    else:
        log_file = '/home/xiyang/Results/RealVSR/SSIM_{}.log'.format(model_name)
        setup_logger('base', log_file, level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')

    subfolder_l = sorted(glob.glob(GT_folder))
    avg_ssim_l = []

    # for each sub-folder
    for subfolder in subfolder_l:
        subfolder_name = subfolder.split('/')[-1]
        logger.info(subfolder_name)
        avg_ssim = 0
        for img_idx, img_GT_path in enumerate(sorted(glob.glob(osp.join(subfolder, '[0-9]*')))):
            if color == 'y':
                img_GT = data_util.read_img(None, img_GT_path)
                img_GT = data_util.bgr2ycbcr(img_GT, only_y=True)
            else:
                img_GT = data_util.read_img(None, img_GT_path)
            # TODO: modify accordingly
            img_LQ_path = img_GT_path.replace('/GT_test/', '/test_results/{}/'.format(model_name))
            assert img_LQ_path != img_GT_path
            if color == 'y':
                img_LQ = data_util.read_img(None, img_LQ_path)
                img_LQ = data_util.bgr2ycbcr(img_LQ, only_y=True)
            else:
                img_LQ = data_util.read_img(None, img_LQ_path)

            if color == 'y':
                ssim = util.calculate_ssim(img_LQ * 255, img_GT * 255)
            else:
                ssim = util.calculate_ssim(img_LQ * 255, img_GT * 255)
            logger.info('{:3d} - {:25} \tSSIM: {:.4f}'.format(img_idx + 1, os.path.basename(img_LQ_path), ssim))
            avg_ssim += ssim

        avg_ssim = avg_ssim / len(subfolder_l)
        avg_ssim_l.append(avg_ssim)
    logger.info(model_name)
    logger.info('SSIM: {:.4f}'.format(sum(avg_ssim_l) / len(avg_ssim_l)))


def evaluate_lpips(model_name, GT_folder, device='cuda:0', log_file=None):

    if log_file:
        setup_logger('base', log_file, level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
    else:
        log_file = '/home/xiyang/Results/RealVSR/LPIPS_{}.log'.format(model_name)
        setup_logger('base', log_file, level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')

    avg_lpips_l = []
    subfolder_l = sorted(glob.glob(GT_folder))

    # for each sub-folder
    for subfolder in subfolder_l:
        subfolder_name = subfolder.split('/')[-1]
        logger.info(subfolder_name)
        avg_lpips = 0
        for img_idx, img_GT_path in enumerate(sorted(glob.glob(osp.join(subfolder, '[0-9]*')))):
            img_GT = Image.open(img_GT_path).convert("RGB")
            # TODO: modify accordingly
            img_LQ_path = img_GT_path.replace('/GT_test/', '/test_results/{}/'.format(model_name))
            assert img_LQ_path != img_GT_path
            img_LQ = Image.open(img_LQ_path).convert("RGB")

            lq = prepare_image(img_LQ, resize=False).to(device)
            gt = prepare_image(img_GT, resize=False).to(device)

            img_name = os.path.basename(img_LQ_path)
            metric = LPIPSvgg().to(device)
            score = metric(lq, gt, as_loss=False)
            logger.info('{:3d} - {:25} \t LPIPS: {:.4f}'.format(img_idx + 1, img_name, score.item()))
            avg_lpips += score.item()

        avg_lpips = avg_lpips / 50
        avg_lpips_l.append(avg_lpips)
    logger.info('{}'.format(model_name))
    logger.info('LPIPS: {:.4f}'.format(sum(avg_lpips_l) / len(avg_lpips_l)))


def evaluate_dists(model_name, GT_folder, device='cuda:0', log_file=None):

    if log_file:
        setup_logger('base', log_file, level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
    else:
        log_file = '/home/xiyang/Results/RealVSR/DISTS_{}.log'.format(model_name)
        setup_logger('base', log_file, level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')

    avg_dists_l = []
    subfolder_l = sorted(glob.glob(GT_folder))

    # for each sub-folder
    for subfolder in subfolder_l:
        subfolder_name = subfolder.split('/')[-1]
        logger.info(subfolder_name)
        avg_dists = 0
        for img_idx, img_GT_path in enumerate(sorted(glob.glob(osp.join(subfolder, '[0-9]*')))):
            img_GT = Image.open(img_GT_path).convert("RGB")
            # TODO: modify accordingly
            img_LQ_path = img_GT_path.replace('/GT_test/', '/test_results/{}/'.format(model_name))
            assert img_LQ_path != img_GT_path
            img_LQ = Image.open(img_LQ_path).convert("RGB")

            lq = prepare_image(img_LQ, resize=False).to(device)
            gt = prepare_image(img_GT, resize=False).to(device)

            img_name = os.path.basename(img_LQ_path)
            metric = DISTS().to(device)
            score = metric(lq, gt, as_loss=False)
            logger.info('{:3d} - {:25} \t DISTS: {:.4f}'.format(img_idx + 1, img_name, score.item()))
            avg_dists += score.item()

        avg_dists = avg_dists / 50
        avg_dists_l.append(avg_dists)
    logger.info('{}'.format(model_name))
    logger.info('DISTS: {:.4f}'.format(sum(avg_dists_l) / len(avg_dists_l)))


if __name__ == '__main__':
    pass
