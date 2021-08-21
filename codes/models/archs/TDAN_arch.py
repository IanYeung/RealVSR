'''
Network architecture for TDAN:
TDAN: Temporally Deformable Alignment Network for Video Super-Resolution
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
try:
    from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


class Align(nn.Module):

    def __init__(self, channel=1, nf=64, nb=5, groups=8):
        super(Align, self).__init__()

        self.initial_conv = nn.Conv2d(channel, nf, 3, padding=1, bias=True)
        self.residual_layers = arch_util.make_layer(arch_util.ResidualBlock_noBN, nb)

        self.bottle_neck = nn.Conv2d(nf * 2, nf, 3, padding=1, bias=True)

        self.offset_conv_1 = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.deform_conv_1 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                                 extra_offset_mask=True)
        self.offset_conv_2 = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.deform_conv_2 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                                 extra_offset_mask=True)
        self.offset_conv_3 = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.deform_conv_3 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                                 extra_offset_mask=True)

        self.offset_conv = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.deform_conv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)
        self.reconstruction = nn.Conv2d(nf, channel, 3, padding=1, bias=True)

    def forward(self, x):
        B, N, C, W, H = x.size()

        # extract features
        y = x.view(-1, C, W, H)
        out = F.relu(self.initial_conv(y), inplace=True)
        out = self.residual_layers(out)
        out = out.view(B, N, -1, W, H)

        # reference frame
        ref_index = N // 2
        ref_frame = out[:, ref_index, :, :, :].clone().contiguous()
        # neighbor frames
        y = []
        for i in range(N):
            nei_frame = out[:, i, :, :, :].contiguous()
            fea = torch.cat([ref_frame, nei_frame], dim=1)
            fea = self.bottle_neck(fea)
            # feature transformation
            offset1 = self.offset_conv_1(fea)
            fea = self.deform_conv_1([fea, offset1])
            offset2 = self.offset_conv_2(fea)
            fea = self.deform_conv_2([fea, offset2])
            offset3 = self.offset_conv_3(fea)
            fea = self.deform_conv_3([nei_frame, offset3])
            offset = self.offset_conv(fea)
            aligned_fea = (self.deform_conv([fea, offset]))
            im = self.reconstruction(aligned_fea)
            y.append(im)
        y = torch.cat(y, dim=1)
        return y


class Trunk(nn.Module):

    def __init__(self, channel=1, nframes=5, scale=4, nf=64, nb=10):
        super(Trunk, self).__init__()
        self.feature_extractor = nn.Sequential(nn.Conv2d(nframes * channel, 64, 3, padding=1, bias=True),
                                               nn.ReLU(inplace=True))
        self.residual_layers = arch_util.make_layer(arch_util.ResidualBlock_noBN, nb)
        self.upsampler = nn.Sequential(arch_util.Upsampler(arch_util.default_conv, scale, 64, act=False),
                                       nn.Conv2d(64, 3, 3, padding=1, bias=False))

    def forward(self, x):
        '''
        :param x: (B, C*T, H, W)
        :return: (B, C, s*H, s*W)
        '''
        out = self.feature_extractor(x)
        out = self.residual_layers(out)
        out = self.upsampler(out)
        return out


class TDAN(nn.Module):
    '''Temporally Deformable Alignment Network'''
    def __init__(self, channel=1, nframes=5, scale=4, nf=64, nb_f=5, nb_b=10, groups=8):
        super(TDAN, self).__init__()

        self.align = Align(channel=channel, nf=nf, nb=nb_f, groups=groups)
        self.trunk = Trunk(channel=channel, nframes=nframes, scale=scale, nf=nf, nb=nb_b)

    def forward(self, x):
        '''
        :param x: (B, T, C, H, W)
        :return: (B, C, s*H, s*W)
        '''
        out = self.align(x)
        out = self.trunk(out)
        return out


if __name__ == '__main__':
    B, N, C, W, H = 1, 7, 3, 64, 64
    model = TDAN(channel=C, nf=64, nframes=N, groups=8, scale=1).to(device=torch.device('cuda'))
    x = torch.randn(B, N, C, W, H).to(device=torch.device('cuda'))
    out = model(x)
    print(out.shape)
