import sys
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import functools

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import utils.util as util


class SimpleBlock(nn.Module):

    def __init__(self, depth=3, n_channels=64, in_nc=3, out_nc=64, kernel_size=3, padding=1, bias=True):
        super(SimpleBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_nc, n_channels, kernel_size=kernel_size, padding=padding, bias=bias))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding, bias=bias))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.append(nn.Conv2d(n_channels, out_nc, kernel_size=kernel_size, padding=padding, bias=bias))
        self.simple_block = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.simple_block(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN (NxN PatchGAN) discriminator"""

    def __init__(self, input_nc, ndf=64, n_block=2, norm_layer=nn.BatchNorm2d, kw=5, padw=2):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_block (int)   -- the number of blocks in the body of discriminator
            norm_layer      -- normalization layer
        """
        super(PatchDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(n_block):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_block, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a PatchGAN (NxN PatchGAN) discriminator"""

    def __init__(self, input_nc, ndf=64, n_block=2, norm_layer=nn.BatchNorm2d, kw=5, padw=2):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_block (int)   -- the number of blocks in the body of discriminator
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(n_block):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_block, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class UNetDiscriminator(nn.Module):

    def __init__(self, in_nc=3, nf=64, depth=2):
        super(UNetDiscriminator, self).__init__()

        # encoder
        self.conv_block_s1 = SimpleBlock(depth=depth, n_channels=1 * nf, in_nc=in_nc,
                                         out_nc=nf, kernel_size=3)
        self.pool1 = nn.Conv2d(nf, 2 * nf, 3, 2, 1, bias=True)

        self.conv_block_s2 = SimpleBlock(depth=depth, n_channels=2 * nf, in_nc=2 * nf,
                                         out_nc=2 * nf, kernel_size=3)
        self.pool2 = nn.Conv2d(2 * nf, 4 * nf, 3, 2, 1, bias=True)

        self.conv_block_s3 = SimpleBlock(depth=depth, n_channels=4 * nf, in_nc=4 * nf,
                                         out_nc=4 * nf, kernel_size=3)

        # decoder
        self.up1 = nn.ConvTranspose2d(4 * nf, 2 * nf, kernel_size=2, stride=2, padding=0, bias=True)
        # cat with conv_block_s4 (256, H/2, W/2)
        self.conv_block_s4 = SimpleBlock(depth=depth, n_channels=2 * nf, in_nc=4 * nf,
                                         out_nc=2 * nf, kernel_size=3)

        self.up2 = nn.ConvTranspose2d(2 * nf, 1 * nf, kernel_size=2, stride=2, padding=0, bias=True)
        # cat with conv_block_s3 (128, H/1, W/1)
        self.conv_block_s5 = SimpleBlock(depth=depth, n_channels=nf, in_nc=2 * nf,
                                         out_nc=1, kernel_size=3)

    def forward(self, x):

        # encoder
        x_s1 = self.conv_block_s1(x)     # 064, H/1, W/1
        x_s2 = self.pool1(x_s1)          # 128, H/2, W/2
        x_s2 = self.conv_block_s2(x_s2)  # 128, H/2, W/2
        x_s3 = self.pool2(x_s2)          # 256, H/4, W/4
        x_s3 = self.conv_block_s3(x_s3)  # 256, H/4, W/4

        # decoder
        out = self.up1(x_s3)             # 128, H/2, W/2
        out = torch.cat((out, x_s2), 1)  # 256, H/2, W/2
        out = self.conv_block_s4(out)    # 128, H/2, W/2
        out = self.up2(out)              # 064, H/1, W/1
        out = torch.cat((out, x_s1), 1)  # 128, H/1, W/1
        out = self.conv_block_s5(out)    # out, H/1, W/1

        return out


class MultiscaleDiscriminator_v1(nn.Module):
    """Multi-scale Discriminator (discriminators are of different architectures)"""
    def __init__(self, input_nc, ndf=64, n_block=3, norm_layer=nn.BatchNorm2d, num_D=3, gan_type='patch'):
        super(MultiscaleDiscriminator_v1, self).__init__()
        self.num_D = num_D
        self.n_block = n_block

        for i in range(num_D):
            if gan_type == 'patch':
                netD = PatchDiscriminator(input_nc, ndf, n_block - i, norm_layer)
            else:
                netD = PixelDiscriminator(input_nc, ndf, n_block - i, norm_layer)
            setattr(self, 'D_{}'.format(str(i)), netD.model)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        num_D = self.num_D
        result = []
        for i in range(num_D):
            model = getattr(self, 'D_{}'.format(str(num_D - 1 - i)))
            result.append(self.singleD_forward(model, input))
        return result


class MultiscaleDiscriminator_v2(nn.Module):
    """Multi-scale Discriminator (discriminators are of same architectures)"""
    def __init__(self, input_nc, ndf=64, n_block=2, norm_layer=nn.BatchNorm2d, num_D=3, gan_type='patch'):
        super(MultiscaleDiscriminator_v2, self).__init__()
        self.num_D = num_D
        self.n_block = n_block

        for i in range(num_D):
            if gan_type == 'patch':
                netD = PatchDiscriminator(input_nc, ndf, n_block, norm_layer)
            else:
                netD = PixelDiscriminator(input_nc, ndf, n_block, norm_layer)
            setattr(self, 'D_{}'.format(str(i)), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'D_{}'.format(str(num_D - 1 - i)))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class MultiscaleDiscriminator_v3(nn.Module):
    """Multi-scale Discriminator (discriminators are of different architectures)"""
    def __init__(self, input_nc, ndf=64, n_block=3, norm_layer=nn.BatchNorm2d, num_D=3, gan_type='patch'):
        super(MultiscaleDiscriminator_v3, self).__init__()
        self.num_D = num_D
        self.n_block = n_block

        for i in range(num_D):
            if gan_type == 'patch':
                netD = PatchDiscriminator(input_nc, ndf, n_block - i, norm_layer)
            else:
                netD = PixelDiscriminator(input_nc, ndf, n_block - i, norm_layer)
            setattr(self, 'D_{}'.format(str(i)), netD.model)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        num_D = self.num_D
        assert len(input) == num_D
        result = []
        for i in range(num_D):
            model = getattr(self, 'D_{}'.format(str(num_D - 1 - i)))
            result.append(self.singleD_forward(model, input[i]))
        return result


class MultiscaleDiscriminator_v4(nn.Module):
    """Multi-scale Discriminator (discriminators are of same architectures)"""
    def __init__(self, input_nc, ndf=64, n_block=2, norm_layer=nn.BatchNorm2d, num_D=3, gan_type='patch'):
        super(MultiscaleDiscriminator_v4, self).__init__()
        self.num_D = num_D
        self.n_block = n_block

        for i in range(num_D):
            if gan_type == 'patch':
                netD = PatchDiscriminator(input_nc, ndf, n_block, norm_layer)
            else:
                netD = PixelDiscriminator(input_nc, ndf, n_block, norm_layer)
            setattr(self, 'D_{}'.format(str(i)), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        num_D = self.num_D
        assert len(input) == num_D
        result = []
        for i in range(num_D):
            model = getattr(self, 'D_{}'.format(str(num_D - 1 - i)))
            result.append(self.singleD_forward(model, input[i]))
        return result


class LaplacePyramidDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_block=2, norm_layer=nn.BatchNorm2d, num_D=3, gan_type='patch'):
        super(LaplacePyramidDiscriminator, self).__init__()
        self.num_D = num_D
        self.input_nc = input_nc
        self.n_block = n_block

        for i in range(num_D):
            if gan_type == 'patch':
                netD = PatchDiscriminator(input_nc, ndf, n_block, norm_layer)
            else:
                netD = PixelDiscriminator(input_nc, ndf, n_block, norm_layer)
            setattr(self, 'D_{}'.format(str(i)), netD.model)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        gauss_kernel = torch.tensor([[1.,  4.,  6.,  4., 1.],
                                     [4., 16., 24., 16., 4.],
                                     [6., 24., 36., 24., 6.],
                                     [4., 16., 24., 16., 4.],
                                     [1.,  4.,  6.,  4., 1.]])
        gauss_kernel /= 256.
        gauss_kernel = gauss_kernel.repeat(self.input_nc, 1, 1, 1).to(input.device)
        lap_pyr = util.laplacian_pyramid(img=input, kernel=gauss_kernel, max_levels=self.num_D)
        num_D = self.num_D
        result = []
        for i in range(num_D):
            model = getattr(self, 'D_{}'.format(str(num_D - 1 - i)))
            result.append(self.singleD_forward(model, lap_pyr[i]))
        return result


class GaussianPyramidDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_block=2, norm_layer=nn.BatchNorm2d, num_D=3, gan_type='patch'):
        super(GaussianPyramidDiscriminator, self).__init__()
        self.num_D = num_D
        self.input_nc = input_nc
        self.n_block = n_block

        for i in range(num_D):
            if gan_type == 'patch':
                netD = PatchDiscriminator(input_nc, ndf, n_block, norm_layer)
            else:
                netD = PixelDiscriminator(input_nc, ndf, n_block, norm_layer)
            setattr(self, 'D_{}'.format(str(i)), netD.model)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        gauss_kernel = torch.tensor([[1.,  4.,  6.,  4., 1.],
                                     [4., 16., 24., 16., 4.],
                                     [6., 24., 36., 24., 6.],
                                     [4., 16., 24., 16., 4.],
                                     [1.,  4.,  6.,  4., 1.]])
        gauss_kernel /= 256.
        gauss_kernel = gauss_kernel.repeat(self.input_nc, 1, 1, 1).to(input.device)
        gau_pyr = util.gaussian_pyramid(img=input, kernel=gauss_kernel, max_levels=self.num_D)
        num_D = self.num_D
        result = []
        for i in range(num_D):
            model = getattr(self, 'D_{}'.format(str(num_D - 1 - i)))
            result.append(self.singleD_forward(model, gau_pyr[i]))
        return result


class ImageGradientPyramidDiscriminator_v1(nn.Module):
    def __init__(self, input_nc, ndf=64, n_block=2, norm_layer=nn.BatchNorm2d, num_D=3, gan_type='patch'):
        super(ImageGradientPyramidDiscriminator_v1, self).__init__()
        self.num_D = num_D
        self.input_nc = input_nc
        self.n_block = n_block

        for i in range(num_D):
            if gan_type == 'patch':
                netD = PatchDiscriminator(input_nc * 2, ndf, n_block, norm_layer)
            else:
                netD = PixelDiscriminator(input_nc * 2, ndf, n_block, norm_layer)
            setattr(self, 'D_{}'.format(str(i)), netD.model)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        gauss_kernel = torch.tensor([[1.,  4.,  6.,  4., 1.],
                                     [4., 16., 24., 16., 4.],
                                     [6., 24., 36., 24., 6.],
                                     [4., 16., 24., 16., 4.],
                                     [1.,  4.,  6.,  4., 1.]])
        gauss_kernel /= 256.
        gauss_kernel = gauss_kernel.repeat(self.input_nc, 1, 1, 1).to(input.device)
        gau_pyr = util.gaussian_pyramid(img=input, kernel=gauss_kernel, max_levels=self.num_D)
        lap_pyr = util.laplacian_pyramid(img=input, kernel=gauss_kernel, max_levels=self.num_D)
        num_D = self.num_D
        result = []
        for i in range(num_D):
            model = getattr(self, 'D_{}'.format(str(num_D - 1 - i)))
            input = torch.cat((gau_pyr[i], lap_pyr[i]), dim=1)  # concat along channel dimension
            result.append(self.singleD_forward(model, input))
        return result


class ImageGradientPyramidDiscriminator_v2(nn.Module):
    def __init__(self, input_nc, ndf=64, n_block=2, norm_layer=nn.BatchNorm2d, num_D=3, gan_type='patch'):
        super(ImageGradientPyramidDiscriminator_v2, self).__init__()
        self.num_D = num_D
        self.input_nc = input_nc
        self.n_block = n_block

        for i in range(num_D):
            if gan_type == 'patch':
                netD = PatchDiscriminator(input_nc * 2, ndf, n_block, norm_layer)
            else:
                netD = PixelDiscriminator(input_nc * 2, ndf, n_block, norm_layer)
            setattr(self, 'D_{}'.format(str(i)), netD.model)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, gau_pyr, lap_pyr):
        assert len(gau_pyr) == self.num_D
        assert len(lap_pyr) == self.num_D
        num_D = self.num_D
        result = []
        for i in range(num_D):
            model = getattr(self, 'D_{}'.format(str(num_D - 1 - i)))
            input = torch.cat((gau_pyr[i], lap_pyr[i]), dim=1)  # concat along channel dimension
            result.append(self.singleD_forward(model, input))
        return result


if __name__ == '__main__':
    with torch.no_grad():
        device = torch.device('cpu')
        x = torch.randn(1, 3, 128, 128).to(device)
        m = torch.randn(1, 3, 128, 128).to(device)

        # patch_gan = PatchDiscriminator(input_nc=3)
        # pixel_gan = PixelDiscriminator(input_nc=3)

        # patch_gan = SPADEPatchDiscriminator(3).to(device)
        # pixel_gan = SPADEPixelDiscriminator(3).to(device)
        #
        # out_patch = patch_gan(x, m)
        # out_pixel = pixel_gan(x, m)
        # print(out_patch.shape)
        # print(out_pixel.shape)

        # lap_pyr = LaplacePyramidDiscriminator(input_nc=3, gan_type='pixel').to(device)
        # gau_pyr = GaussianPyramidDiscriminator(input_nc=3, gan_type='pixel').to(device)
        # lap_gau = ImageGradientPyramidDiscriminator(input_nc=3, gan_type='pixel').to(device)
        # out = lap_pyr(x)
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)
        # out = gau_pyr(x)
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)
        # out = lap_gau(x)
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)

        # # x1 = torch.randn(1, 3, 128, 128).to(device)
        # # x2 = torch.randn(1, 3, 64, 64).to(device)
        # # x3 = torch.randn(1, 3, 32, 32).to(device)
        # kernel = util.gauss_kernel(device=torch.device('cpu'))
        # x1, x2, x3 = util.gaussian_pyramid(x, kernel=kernel, max_levels=3)
        # # x1, x2, x3 = util.laplacian_pyramid(x, kernel=kernel, max_levels=3)
        #
        # multi_scale_v3 = MultiscaleDiscriminator_v3(input_nc=3, gan_type='pixel').to(device)
        # multi_scale_v4 = MultiscaleDiscriminator_v4(input_nc=3, gan_type='pixel').to(device)
        #
        # out = multi_scale_v3([x3, x2, x1])
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)
        # out = multi_scale_v4([x3, x2, x1])
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)
        #
        # unet = UNetDiscriminator(in_nc=3, nf=64)
        # out = unet(x)
        # print(out.shape)
