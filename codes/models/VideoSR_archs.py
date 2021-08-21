import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.VGG_arch as VGG_arch
import models.archs.TOF_arch as TOF_arch
import models.archs.RCAN_arch as RCAN_arch
import models.archs.EDVR_arch as EDVR_arch
import models.archs.TDAN_arch as TDAN_arch
import models.archs.FSTRN_arch as FSTRN_arch
import models.archs.discriminator_arch as discriminator_arch


####################
# define network
####################


def define_G(opt):
    """Discriminator"""
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'EDVR':
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nc=opt_net['nc'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'])

    elif which_model == 'EDVR_NoUp':
        netG = EDVR_arch.EDVR_NoUp(nf=opt_net['nf'], nc=opt_net['nc'], nframes=opt_net['nframes'],
                                   groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                   back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                   predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                                   w_TSA=opt_net['w_TSA'])

    elif which_model == 'TDAN':
        netG = TDAN_arch.TDAN(nf=opt_net['nf'], channel=opt_net['nc'], nframes=opt_net['nframes'],
                              nb_f=opt_net['nb_f'], nb_b=opt_net['nb_b'], groups=opt_net['groups'],
                              scale=opt['scale'])

    elif which_model == 'TOF':
        netG = TOF_arch.TOF(nframes=opt_net['nframes'], K=opt_net['K'], in_nc=opt_net['nc'],
                            out_nc=opt_net['nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                            upscale=opt['scale'])

    elif which_model == 'FSTRN':
        netG = FSTRN_arch.FSTRN(k=opt_net['k'], nf=opt_net['nf'], scale=opt['scale'],
                                nframes=opt_net['nframes'])

    elif which_model == 'RCAN':
        netG = RCAN_arch.RCAN(num_in_ch=opt_net['num_in_ch'], num_out_ch=opt_net['num_out_ch'],
                              num_frames=opt_net['num_frames'], num_feat=opt_net['num_feat'],
                              num_group=opt_net['num_group'], num_block=opt_net['num_block'],
                              squeeze_factor=opt_net['squeeze_factor'], upscale=opt['scale'],
                              res_scale=opt_net['res_scale'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_192':
        netD = VGG_arch.Discriminator_VGG_192(in_nc=opt_net['in_nc'], nf=opt_net['nf'])

    elif which_model == 'PatchDiscriminator':
        netD = discriminator_arch.PatchDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'],
                                                     norm_layer=nn.BatchNorm2d)
    elif which_model == 'PixelDiscriminator':
        netD = discriminator_arch.PixelDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'],
                                                     norm_layer=nn.BatchNorm2d)
    elif which_model == 'UNetDiscriminator':
        netD = discriminator_arch.UNetDiscriminator(in_nc=opt_net['in_nc'], nf=opt_net['nf'])

    elif which_model == 'MultiscaleDiscriminator_v1':
        netD = discriminator_arch.MultiscaleDiscriminator_v1(input_nc=opt_net['in_nc'],
                                                             ndf=opt_net['nf'],
                                                             num_D=opt_net['num_D'],
                                                             norm_layer=nn.BatchNorm2d,
                                                             gan_type=opt_net['gan_type'])
    elif which_model == 'MultiscaleDiscriminator_v2':
        netD = discriminator_arch.MultiscaleDiscriminator_v2(input_nc=opt_net['in_nc'],
                                                             ndf=opt_net['nf'],
                                                             num_D=opt_net['num_D'],
                                                             norm_layer=nn.BatchNorm2d,
                                                             gan_type=opt_net['gan_type'])
    elif which_model == 'MultiscaleDiscriminator_v3':
        netD = discriminator_arch.MultiscaleDiscriminator_v3(input_nc=opt_net['in_nc'],
                                                             ndf=opt_net['nf'],
                                                             num_D=opt_net['num_D'],
                                                             norm_layer=nn.BatchNorm2d,
                                                             gan_type=opt_net['gan_type'])
    elif which_model == 'MultiscaleDiscriminator_v4':
        netD = discriminator_arch.MultiscaleDiscriminator_v4(input_nc=opt_net['in_nc'],
                                                             ndf=opt_net['nf'],
                                                             num_D=opt_net['num_D'],
                                                             norm_layer=nn.BatchNorm2d,
                                                             gan_type=opt_net['gan_type'])
    elif which_model == 'LaplacePyramidDiscriminator':
        netD = discriminator_arch.LaplacePyramidDiscriminator(input_nc=opt_net['in_nc'],
                                                              ndf=opt_net['nf'],
                                                              num_D=opt_net['num_D'],
                                                              norm_layer=nn.BatchNorm2d,
                                                              gan_type=opt_net['gan_type'])
    elif which_model == 'GaussianPyramidDiscriminator':
        netD = discriminator_arch.GaussianPyramidDiscriminator(input_nc=opt_net['in_nc'],
                                                               ndf=opt_net['nf'],
                                                               num_D=opt_net['num_D'],
                                                               norm_layer=nn.BatchNorm2d,
                                                               gan_type=opt_net['gan_type'])
    elif which_model == 'ImageGradientPyramidDiscriminator_v1':
        netD = discriminator_arch.ImageGradientPyramidDiscriminator_v1(input_nc=opt_net['in_nc'],
                                                                       ndf=opt_net['nf'],
                                                                       num_D=opt_net['num_D'],
                                                                       norm_layer=nn.BatchNorm2d,
                                                                       gan_type=opt_net['gan_type'])
    elif which_model == 'ImageGradientPyramidDiscriminator_v2':
        netD = discriminator_arch.ImageGradientPyramidDiscriminator_v2(input_nc=opt_net['in_nc'],
                                                                       ndf=opt_net['nf'],
                                                                       num_D=opt_net['num_D'],
                                                                       norm_layer=nn.BatchNorm2d,
                                                                       gan_type=opt_net['gan_type'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


def define_F(opt, use_bn=False):
    """Network for Perceptual Loss"""
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGG_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device)
    netF.eval()  # No need to train

    return netF