import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.VideoSR_archs as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
import data.augments_video_allpair as augments
from models.loss import CharbonnierLoss, HuberLoss, GANLoss, GWLoss
from IQA_pytorch import SSIM, MS_SSIM

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import utils.util as util


logger = logging.getLogger('base')


class VideoSRGANModel(BaseModel):
    def __init__(self, opt):
        super(VideoSRGANModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if opt['dist']:
                self.netD = DistributedDataParallel(self.netD, device_ids=[torch.cuda.current_device()])
            else:
                self.netD = DataParallel(self.netD)

            self.netG.train()
            self.netD.train()

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight_s'] > 0:
                loss_type = train_opt['pixel_criterion_s']
                if loss_type == 'l1':
                    self.cri_pix_s = nn.L1Loss(reduction='mean').to(self.device)
                elif loss_type == 'l2':
                    self.cri_pix_s = nn.MSELoss(reduction='mean').to(self.device)
                elif loss_type == 'cb':
                    self.cri_pix_s = CharbonnierLoss(reduction='mean').to(self.device)
                elif loss_type == 'hb':
                    self.cri_pix_s = HuberLoss(reduction='mean').to(self.device)
                elif loss_type == 'gw':
                    self.cri_pix_s = GWLoss(w=4, reduction='mean').to(self.device)
                elif loss_type == 'ssim':
                    self.cri_pix_s = SSIM(channels=1).to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
                self.l_pix_w_s = train_opt['pixel_weight_s']
            else:
                self.cri_pix_s = None

            if train_opt['pixel_weight_d'] > 0:
                loss_type = train_opt['pixel_criterion_d']
                if loss_type == 'l1':
                    self.cri_pix_d = nn.L1Loss(reduction='mean').to(self.device)
                elif loss_type == 'l2':
                    self.cri_pix_d = nn.MSELoss(reduction='mean').to(self.device)
                elif loss_type == 'cb':
                    self.cri_pix_d = CharbonnierLoss(reduction='mean').to(self.device)
                elif loss_type == 'hb':
                    self.cri_pix_d = HuberLoss(reduction='mean').to(self.device)
                elif loss_type == 'gw':
                    self.cri_pix_d = GWLoss(w=4, reduction='mean').to(self.device)
                elif loss_type == 'ssim':
                    self.cri_pix_d = SSIM(channels=1).to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
                self.l_pix_w_d = train_opt['pixel_weight_d']
            else:
                self.cri_pix_d = None

            if train_opt['pixel_weight_c'] > 0:
                loss_type = train_opt['pixel_criterion_c']
                if loss_type == 'l1':
                    self.cri_pix_c = nn.L1Loss(reduction='mean').to(self.device)
                elif loss_type == 'l2':
                    self.cri_pix_c = nn.MSELoss(reduction='mean').to(self.device)
                elif loss_type == 'cb':
                    self.cri_pix_c = CharbonnierLoss(reduction='mean').to(self.device)
                elif loss_type == 'hb':
                    self.cri_pix_c = HuberLoss(reduction='mean').to(self.device)
                elif loss_type == 'gw':
                    self.cri_pix_c = GWLoss(w=4, reduction='mean').to(self.device)
                elif loss_type == 'ssim':
                    self.cri_pix_c = SSIM(channels=2).to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
                self.l_pix_w_c = train_opt['pixel_weight_c']
            else:
                self.cri_pix_c = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif l_fea_type == 'cb':
                    self.cri_fea = CharbonnierLoss(reduction='mean').to(self.device)
                elif l_fea_type == 'hb':
                    self.cri_fea = HuberLoss(reduction='mean').to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    pass  # do not need to use DistributedDataParallel for netF
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        if self.opt.get('augment'):
            opt = self.opt['augment']
            self.var_H, self.var_L = augments.apply_augment(
                self.var_H, self.var_L,
                opt['augs'], opt['probs'], opt['alphas'], opt['mix_p']
            )
            self.fake_H = self.netG(self.var_L)
        else:
            self.fake_H = self.netG(self.var_L)

        # calculate loss and back propagate
        center_idx = self.var_L.size(1) // 2

        fake_y = self.fake_H[:, 0:1, :, :]
        fake_c = self.fake_H[:, 1:3, :, :]

        real_y = self.var_H[:, center_idx, 0:1, :, :].contiguous()
        real_c = self.var_H[:, center_idx, 1:3, :, :].contiguous()

        real_ref_y = self.var_ref[:, center_idx, 0:1, :, :].contiguous()
        real_ref_c = self.var_ref[:, center_idx, 1:3, :, :].contiguous()

        gauss_kernel = util.gauss_kernel(size=5, device=self.device, channels=1)
        fake_y_pyr = util.laplacian_pyramid(img=fake_y, kernel=gauss_kernel, max_levels=3)
        real_y_pyr = util.laplacian_pyramid(img=real_y, kernel=gauss_kernel, max_levels=3)
        real_ref_y_pyr = util.laplacian_pyramid(img=real_ref_y, kernel=gauss_kernel, max_levels=3)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # pixel loss
            if self.cri_pix_s:
                l_g_pix_s = self.l_pix_w_s * self.cri_pix_s(fake_y_pyr[-1], real_y_pyr[-1])
                l_g_total += l_g_pix_s
            if self.cri_pix_d:
                l_g_pix_d = self.l_pix_w_d * self.cri_pix_d(fake_y_pyr[0], real_y_pyr[0]) + \
                            self.l_pix_w_d * self.cri_pix_d(fake_y_pyr[1], real_y_pyr[1])
                l_g_total += l_g_pix_d
            if self.cri_pix_c:
                l_g_pix_c = self.l_pix_w_c * self.cri_pix_c(fake_c, real_c)
                l_g_total += l_g_pix_c
            # feature loss
            if self.cri_fea:
                real_fea = self.netF(real_y_pyr[-1]).detach()
                fake_fea = self.netF(fake_y_pyr[-1])
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            # gan loss
            if self.opt['train']['gan_type'] == 'gan':
                pred_g_fake = self.netD(fake_y_pyr[:-1])
                l_g_gan = 0
                for i in range(len(pred_g_fake)):
                    l_g_gan += self.l_gan_w * self.cri_gan(pred_g_fake[i], True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real = [out.detach() for out in self.netD(real_ref_y_pyr[:-1])]
                pred_g_fake = self.netD(fake_y_pyr[:-1])
                l_g_gan = 0
                for i in range(len(pred_d_real)):
                    l_g_gan += self.l_gan_w * (
                        self.cri_gan(pred_d_real[i] - torch.mean(pred_g_fake[i]), False) +
                        self.cri_gan(pred_g_fake[i] - torch.mean(pred_d_real[i]), True)) / 2
            l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        if self.opt['train']['gan_type'] == 'gan':
            # need to forward and backward separately, since batch norm statistics differ
            # real
            pred_d_real = self.netD(real_ref_y_pyr[:-1])
            l_d_real = 0
            for i in range(len(pred_d_real)):
                l_d_real += self.cri_gan(pred_d_real[i], True)
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD([x.detach() for x in fake_y_pyr[:-1]])  # detach to avoid BP to G
            l_d_fake = 0
            for i in range(len(pred_d_fake)):
                l_d_fake += self.cri_gan(pred_d_fake[i], False)
            l_d_fake.backward()
        elif self.opt['train']['gan_type'] == 'ragan':
            # need to forward and backward separately, since batch norm statistics differ
            # real
            pred_d_fake = [out.detach() for out in self.netD([x.detach() for x in fake_y_pyr[:-1]])]
            pred_d_real = self.netD(real_ref_y_pyr[:-1])
            l_d_real = 0
            for i in range(len(pred_d_fake)):
                l_d_real += self.cri_gan(pred_d_real[i] - torch.mean(pred_d_fake[i]), True) * 0.5
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD([x.detach() for x in fake_y_pyr[:-1]])  # detach to avoid BP to G
            l_d_fake = 0
            for i in range(len(pred_d_fake)):
                l_d_fake += self.cri_gan(pred_d_fake[i] - torch.mean(pred_d_real[i].detach()), False) * 0.5
            l_d_fake.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix_s:
                self.log_dict['l_g_pix_s'] = l_g_pix_s.item()
            if self.cri_pix_d:
                self.log_dict['l_g_pix_d'] = l_g_pix_d.item()
            if self.cri_pix_c:
                self.log_dict['l_g_pix_c'] = l_g_pix_c.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
            self.log_dict['l_g_total'] = l_g_total.item()
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        # self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        # self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQs'] = self.var_L.detach()[0].float().cpu()
        out_dict['HQ'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
