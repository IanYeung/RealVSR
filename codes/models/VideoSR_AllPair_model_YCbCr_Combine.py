import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.VideoSR_archs as networks
import models.lr_scheduler as lr_scheduler
import data.augments_video_allpair as augments
from .base_model import BaseModel
from models.loss import CharbonnierLoss, HuberLoss, GWLoss, PyramidLoss, LapPyrLoss
from IQA_pytorch import SSIM, MS_SSIM, DISTS


logger = logging.getLogger('base')


class VideoSRModel(BaseModel):

    def __init__(self, opt):
        super(VideoSRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='mean').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='mean').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss(reduction='mean').to(self.device)
            elif loss_type == 'hb':
                self.cri_pix = HuberLoss(reduction='mean').to(self.device)
            elif loss_type == 'pyr':
                self.cri_pix = PyramidLoss(num_levels=3, pyr_mode='gau', loss_mode='cb', reduction='mean')
            elif loss_type == 'lappyr':
                self.cri_pix = LapPyrLoss(num_levels=3, lf_mode='ssim', hf_mode='cb', reduction='mean')
            elif loss_type == 'msssim':
                channels = opt['network_G']['nc']
                self.cri_pix = MS_SSIM(channels=channels).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            if train_opt.get('edge_criterion') and train_opt.get('edge_weight'):
                loss_type = train_opt['edge_criterion']
                if loss_type == 'l1':
                    self.cri_edg = nn.L1Loss(reduction='mean').to(self.device)
                elif loss_type == 'l2':
                    self.cri_edg = nn.MSELoss(reduction='mean').to(self.device)
                elif loss_type == 'cb':
                    self.cri_edg = CharbonnierLoss(reduction='mean').to(self.device)
                elif loss_type == 'hb':
                    self.cri_edg = HuberLoss(reduction='mean').to(self.device)
                elif loss_type == 'pyr':
                    self.cri_edg = PyramidLoss(num_levels=3, pyr_mode='lap', loss_mode='cb', reduction='mean')
                elif loss_type == 'lappyr':
                    self.cri_edg = LapPyrLoss(num_levels=3, lf_mode='ssim', hf_mode='cb', reduction='mean')
                elif loss_type == 'msssim':
                    channels = opt['network_G']['nc']
                    self.cri_edg = MS_SSIM(channels=channels).to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
                self.l_edg_w = train_opt['edge_weight']
            else:
                logger.info('Remove edge loss.')
                self.cri_edg = None

            # G feature loss
            if train_opt.get('feature_criterion') and train_opt.get('feature_weight'):
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif loss_type == 'cb':
                    self.cri_fea = CharbonnierLoss().to(self.device)
                elif loss_type == 'hb':
                    self.cri_fea = HuberLoss().to(self.device)
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

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer, train_opt['lr_steps'],
                            restarts=train_opt['restarts'],
                            weights=train_opt['restart_weights'],
                            gamma=train_opt['lr_gamma'],
                            clear_state=train_opt['clear_state']
                        )
                    )
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'],
                            eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'],
                            weights=train_opt['restart_weights']
                        )
                    )
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.var_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        # clear gradient and forward propagate
        self.optimizer_G.zero_grad()
        if self.opt['augment']:
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
        l_tot = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H[:, center_idx, :, :, :].contiguous())
        if self.cri_edg:
            l_edg = self.l_edg_w * self.cri_edg(self.fake_H, self.var_H[:, center_idx, :, :, :].contiguous())
            l_tot += l_edg
        if self.cri_fea:
            real_fea = self.netF(self.var_H[:, center_idx, :, :, :]).detach()
            fake_fea = self.netF(self.fake_H)
            l_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
            l_tot += l_fea
        l_tot.backward()
        self.optimizer_G.step()
        # set log
        self.log_dict['l_tot'] = l_tot.item()
        if self.cri_fea:
            self.log_dict['l_fea'] = l_fea.item()
        if self.cri_edg:
            self.log_dict['l_edg'] = l_edg.item()

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
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def load_separately(self):
        load_path_G_a = self.opt['path']['pretrain_model_G_a']
        load_path_G_b = self.opt['path']['pretrain_model_G_b']
        name_a = self.opt['path']['name_a']
        name_b = self.opt['path']['name_b']
        if load_path_G_a is not None and load_path_G_b is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G_a))
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G_b))
            self.load_network_separately(load_path_G_a, load_path_G_b, name_a, name_b,
                                         self.netG, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
