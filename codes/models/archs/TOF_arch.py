import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

from models.archs.SRResNet_arch import MSRResNet


class SpyNet_Block(nn.Module):
    """
    A submodule of SpyNet.
    """

    def __init__(self, ic=8):
        super(SpyNet_Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

        # initialization
        arch_util.initialize_weights(self.block, 0.1)

    def forward(self, x):
        """
        input: x: [ref im, nbr im, initial flow] - (B, 8, H, W)
        output: estimated flow - (B, 2, H, W)
        """
        return self.block(x)


class SpyNet(nn.Module):
    """
    SpyNet for estimating optical flow
    Ranjan et al., Optical Flow Estimation using a Spatial Pyramid Network, 2016
    """

    def __init__(self, K=3):
        super(SpyNet, self).__init__()

        self.K = K
        ## modify input block
        self.block0 = SpyNet_Block(ic=6)
        self.blocks = nn.ModuleList([SpyNet_Block(ic=8) for _ in range(K)])

    def forward(self, ref, nbr):
        """Estimating optical flow in coarse level, upsample, and estimate in fine level
        Note: the size of input should be divisible by 8, if not, pad them before input
        input: ref: reference image - [B, 3, H, W]
               nbr: the neighboring image to be warped - [B, 3, H, W]
        output: warpped nbr by estimated optical flow - [B, 3, H, W]
          flow: estimated optical flow (absolute displacement) - [B, 2, H, W]
        """
        B, C, H, W = ref.size()
        ref = [ref]
        nbr = [nbr]

        for _ in range(self.K):
            ref.insert(
                0,
                nn.functional.avg_pool2d(input=ref[0], kernel_size=2, stride=2,
                                         count_include_pad=False)
            )
            nbr.insert(
                0,
                nn.functional.avg_pool2d(input=nbr[0], kernel_size=2, stride=2,
                                         count_include_pad=False)
            )

        flow = self.block0(torch.cat([ref[0], nbr[0]], 1))  # [H//2^K, W//2^K]

        for i in range(self.K):
            flow_up = nn.functional.interpolate(input=flow, scale_factor=2, mode='bilinear',
                                                align_corners=True) * 2.0
            flow = flow_up + self.blocks[i](
                torch.cat([ref[i+1], arch_util.flow_warp(nbr[i+1], flow_up.permute(0, 2, 3, 1)), flow_up], 1)
            )

        output = arch_util.flow_warp(nbr[-1], flow.permute(0, 2, 3, 1))
        return output, flow


class MSRResNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        if self.upscale == 2 or self.upscale == 3 or self.upscale == 4:
            arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        C = x.size(1)
        if C > 3:
            ## for video sr with multi-frames input
            x_base = x[:, C//2-1:C//2+2, :, :]
        else:
            x_base = x

        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x_base, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


class TOF(nn.Module):
    """
    Video sr based on SpyNet and MEDSR
    Args [in_nc] is number of input channels of a single frame!
    """
    def __init__(self, nframes=3, K=3, in_nc=3, out_nc=3, nf=32, nb=12, upscale=2):
        super(TOF, self).__init__()

        self.nframes = nframes

        self.align_arch = SpyNet(K=K)
        self.sr_arch = MSRResNet(in_nc=nframes * in_nc, out_nc=out_nc, nf=nf, nb=nb, upscale=upscale)

    def forward(self, x):
        """
        x: [B, T, C, H, W], T = nframes.
        """
        B, T, C, H, W = x.size()
        assert T == self.nframes

        ## warp neighbour frames to reference frame
        # reference frame
        ref_index = T // 2
        ref_frame = x[:, ref_index, :, :, :]
        # neighbour frames
        y = []
        nbrs = []
        flows = []
        for i in range(T):
            if i == ref_index:
                y.append(ref_frame)
            else:
                warp_nbr_frame, flow = self.align_arch(ref_frame, x[:, i, :, :, :])
                y.append(warp_nbr_frame)
                nbrs.append(warp_nbr_frame)
                flows.append(flow)

        ## cat frames as input of sr module
        y = torch.cat(y, dim=1)
        out = self.sr_arch(y)

        return out


if __name__ == '__main__':
    device = torch.device('cuda')
    model = TOF(nframes=1, K=3, in_nc=3, out_nc=3, nf=32, nb=10, upscale=1).to(device)
    x = torch.randn(4, 1, 3, 128, 128).to(device)
    out = model(x)
    print(out[0].shape)

