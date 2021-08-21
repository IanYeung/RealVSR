"""
Network architecture for FSTRN:
Fast Spatio-Temporal Residual Network for Video Super-Resolution (CVPR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FRB(nn.Module):
    """Fast spatial-temporal residual block"""
    def __init__(self, k=3, nf=64):
        super(FRB, self).__init__()
        self.prelu = nn.PReLU()
        self.conv3d_1 = nn.Conv3d(nf, nf, (1, k, k), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.conv3d_2 = nn.Conv3d(nf, nf, (k, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=True)

    def forward(self, x):
        res = x
        out = self.conv3d_2(self.conv3d_1(self.prelu(x)))
        return res + out


class FSTRN(nn.Module):
    """Fast spatial-temporal residual network"""
    def __init__(self, k=3, nf=64, scale=4, nframes=5):
        super(FSTRN, self).__init__()
        self.k = k
        self.nf = nf
        self.scale = scale
        self.center = nframes // 2
        #### LFENet
        self.conv3d_fe = nn.Conv3d(3, nf, (k, k, k), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        #### FRBs
        self.frb_1 = FRB(k=k, nf=nf)
        self.frb_2 = FRB(k=k, nf=nf)
        self.frb_3 = FRB(k=k, nf=nf)
        self.frb_4 = FRB(k=k, nf=nf)
        self.frb_5 = FRB(k=k, nf=nf)
        #### LSRNet
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.3, inplace=False)
        self.conv3d_1 = nn.Conv3d(nf, nf, (k, k, k), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.upsample = nn.ConvTranspose3d(nf, nf, (1, self.scale, self.scale),
                                           stride=(1, self.scale, self.scale), bias=True)
        self.conv3d_2 = nn.Conv3d(nf, 3, (k, k, k), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)

    def forward(self, x):
        """
        x: [B, T, C, H, W], reshape to [B, C, T, H, W] for Conv3D
        """
        x = x.permute(0, 2, 1, 3, 4)
        #### LFENet
        cs_res = x
        out = self.conv3d_fe(x)
        #### FRBs (with LR residual connection)
        lr_res = out
        out = self.frb_5(self.frb_4(self.frb_3(self.frb_2(self.frb_1(out)))))
        out = lr_res + out
        #### LSRNet
        out = self.dropout(self.prelu(out))
        out = self.conv3d_1(out)
        out = self.upsample(out)
        out = self.conv3d_2(out)
        #### Cross-space residual connection
        cs_out = F.interpolate(cs_res, scale_factor=(1, self.scale, self.scale), mode='trilinear', align_corners=False)
        out = cs_out + out
        return out[:, :, self.center, :, :]


if __name__ == '__main__':
    device = torch.device('cuda')
    model = FSTRN(k=3, nf=64, nframes=5, scale=1).to(device)
    x = torch.randn(32, 5, 3, 48, 48).to(device)
    out = model(x)
    print(out.shape)
