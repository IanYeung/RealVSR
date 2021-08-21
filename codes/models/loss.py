import torch
import torch.nn as nn
import torch.nn.functional as F
import data.util as data_util
import utils.util as util

from IQA_pytorch import SSIM, MS_SSIM


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        diff = x - y
        if self.reduction == 'mean':
            loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        else:
            loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


class HuberLoss(nn.Module):
    """Huber Loss (L1)"""
    def __init__(self, delta=1e-2, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, x, y):
        abs_diff = torch.abs(x - y)
        q_term = torch.min(abs_diff, torch.full_like(abs_diff, self.delta))
        l_term = abs_diff - q_term
        if self.reduction == 'mean':
            loss = torch.mean(0.5 * q_term ** 2 + self.delta * l_term)
        else:
            loss = torch.sum(0.5 * q_term ** 2 + self.delta * l_term)
        return loss


class TVLoss(nn.Module):
    """Total Variation Loss"""
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
               torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


class GWLoss(nn.Module):
    """Gradient Weighted Loss"""
    def __init__(self, w=4, reduction='mean'):
        super(GWLoss, self).__init__()
        self.w = w
        self.reduction = reduction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)
        self.weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=sobel_y, requires_grad=False)

    def forward(self, x1, x2):
        b, c, w, h = x1.shape
        weight_x = self.weight_x.expand(c, 1, 3, 3).type_as(x1)
        weight_y = self.weight_y.expand(c, 1, 3, 3).type_as(x1)
        Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        # loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = (1 + self.w * dx) * (1 + self.w * dy) * torch.abs(x1 - x2)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class StyleLoss(nn.Module):
    """Style Loss"""
    def __init__(self):
        super(StyleLoss, self).__init__()

    @staticmethod
    def gram_matrix(self, x):
        B, C, H, W = x.size()
        features = x.view(B * C, H * W)
        G = torch.mm(features, features.t())
        return G.div(B * C * H * W)

    def forward(self, input, target):
        G_i = self.gram_matrix(input)
        G_t = self.gram_matrix(target).detach()
        loss = F.mse_loss(G_i, G_t)
        return loss


class GANLoss(nn.Module):
    """GAN loss (vanilla | lsgan | wgan-gp)"""
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    """Gradient Penalty Loss"""
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


class PyramidLoss(nn.Module):
    """Pyramid Loss"""
    def __init__(self, num_levels=3, pyr_mode='gau', loss_mode='l1', reduction='mean'):
        super(PyramidLoss, self).__init__()
        self.num_levels = num_levels
        self.pyr_mode = pyr_mode
        self.loss_mode = loss_mode
        assert self.pyr_mode == 'gau' or self.pyr_mode == 'lap'
        if self.loss_mode == 'l1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif self.loss_mode == 'l2':
            self.loss = nn.MSELoss(reduction=reduction)
        elif self.loss_mode == 'hb':
            self.loss = HuberLoss(reduction=reduction)
        elif self.loss_mode == 'cb':
            self.loss = CharbonnierLoss(reduction=reduction)
        else:
            raise ValueError()

    def forward(self, x, y):
        B, C, H, W = x.shape
        device = x.device
        gauss_kernel = util.gauss_kernel(size=5, device=device, channels=C)
        if self.pyr_mode == 'gau':
            pyr_x = util.gau_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
            pyr_y = util.gau_pyramid(img=y, kernel=gauss_kernel, max_levels=self.num_levels)
        else:
            pyr_x = util.lap_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
            pyr_y = util.lap_pyramid(img=y, kernel=gauss_kernel, max_levels=self.num_levels)
        loss = 0
        for i in range(self.num_levels):
            loss += self.loss(pyr_x[i], pyr_y[i])
        return loss


class LapPyrLoss(nn.Module):
    """Pyramid Loss"""
    def __init__(self, num_levels=3, lf_mode='ssim', hf_mode='cb', reduction='mean'):
        super(LapPyrLoss, self).__init__()
        self.num_levels = num_levels
        self.lf_mode = lf_mode
        self.hf_mode = hf_mode
        if lf_mode == 'ssim':
            self.lf_loss = SSIM(channels=1)
        elif lf_mode == 'cb':
            self.lf_loss = CharbonnierLoss(reduction=reduction)
        else:
            raise ValueError()
        if hf_mode == 'ssim':
            self.hf_loss = SSIM(channels=1)
        elif hf_mode == 'cb':
            self.hf_loss = CharbonnierLoss(reduction=reduction)
        else:
            raise ValueError()

    def forward(self, x, y):
        B, C, H, W = x.shape
        device = x.device
        gauss_kernel = util.gauss_kernel(size=5, device=device, channels=C)
        pyr_x = util.laplacian_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
        pyr_y = util.laplacian_pyramid(img=y, kernel=gauss_kernel, max_levels=self.num_levels)
        loss = self.lf_loss(pyr_x[-1], pyr_y[-1])
        for i in range(self.num_levels - 1):
            loss += self.hf_loss(pyr_x[i], pyr_y[i])
        return loss


if __name__ == '__main__':
    device = torch.device('cuda')
    x1 = torch.randn(4, 3, 64, 64).to(device)
    x1.requires_grad = True
    x2 = torch.randn(4, 3, 64, 64).to(device)
    x2.requires_grad = True
    loss = GWLoss().to(device)
    l = loss(x1, x2)
    print(l)
    l.backward()

