#!/usr/bin/env python
"""Different Network Structures"""

__author__ = "Qiaoying Huang"
__date__ = "04/08/2019"
__institute__ = "Rutgers University"


import torch.nn as nn
import torch
from torch.nn.init import normal_
from utils import sigtoimage
import math
import torch.nn.functional as F


# Cascaded Convolutional Blocks
class Cascade_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Cascade_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channel, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):
        x1 = self.layer(x)
        return x1


# DC layer
class DC_layer(nn.Module):
    def __init__(self):
        super(DC_layer, self).__init__()


    def forward(self, mask, x_rec, x_under):
        x_rec_per = x_rec.permute(0, 2, 3, 1)
        x_tran_per = torch.fft(x_rec_per, 2)
        x_tran = x_tran_per.permute(0, 3, 1, 2)
        masks = torch.cat((mask, mask), 1)
        matrixones = torch.ones_like(masks.data)
        output = (matrixones - masks) * x_tran + x_under
        output_per = output.permute(0, 2, 3, 1)
        output_tran_per = torch.ifft(output_per, 2)
        output_tran = output_tran_per.permute(0, 3, 1, 2)
        return output_tran


# MICCAN without long residual, reconstruction block could be cascaded blocks, UNet, UCA(UNet with attention)
class MICCAN(nn.Module):
    def __init__(self, in_channel, out_channel, n_layer, block='CSE'):
        super(MICCAN, self).__init__()
        if block == 'UCA':
            self.layer = nn.ModuleList([UNetCSE(in_channel, out_channel) for _ in range(n_layer)])
        if block == 'UNet':
            self.layer = nn.ModuleList([UNet(in_channel, out_channel) for _ in range(n_layer)])
        if block == 'Cas':
            self.layer = nn.ModuleList([Cascade_Block(in_channel, out_channel) for _ in range(n_layer)])
        self.dc = DC_layer()
        self.nlayer = n_layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                normal_(m.weight.data, 0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x_under, mask):
        x_under_per = x_under.permute(0, 2, 3, 1)
        x_zf_per = torch.ifft(x_under_per, 2)
        x_zf = x_zf_per.permute(0, 3, 1, 2)
        x_rec_dc = x_zf
        recimg = list()
        recimg.append(sigtoimage(x_zf))
        for i, l in enumerate(self.layer):
            x_rec = self.layer[i](x_rec_dc)
            x_res = x_rec_dc + x_rec
            x_rec_dc = self.dc(mask, x_res, x_under)
            recimg.append(sigtoimage(x_rec_dc))
        return recimg


# MICCAN without long residual, reconstruction block could be cascaded blocks, UNet, UCA(UNet with attention)
class MICCANlong(nn.Module):
    def __init__(self, in_channel, out_channel, n_layer, block='CSE'):
        super(MICCANlong, self).__init__()
        if block == 'UCA':
            self.layer = nn.ModuleList([UNetCSE(in_channel, out_channel) for _ in range(n_layer)])
        if block == 'UNet':
            self.layer = nn.ModuleList([UNet(in_channel, out_channel) for _ in range(n_layer)])
        if block == 'Cas':
            self.layer = nn.ModuleList([Cascade_Block(in_channel, out_channel) for _ in range(n_layer)])
        self.dc = DC_layer()
        self.nlayer = n_layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                normal_(m.weight.data, 0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x_under, mask):
        x_under_per = x_under.permute(0, 2, 3, 1)
        x_zf_per = torch.ifft(x_under_per, 2)
        x_zf = x_zf_per.permute(0, 3, 1, 2)
        x_rec_dc = x_zf
        recimg = list()
        recimg.append(sigtoimage(x_zf))
        for i in range(self.nlayer-1):
            x_rec = self.layer[i](x_rec_dc)
            x_res = x_rec_dc + x_rec
            x_rec_dc = self.dc(mask, x_res, x_under)
            recimg.append(sigtoimage(x_rec_dc))
        x_rec = self.layer[i+1](x_rec_dc)
        x_res = x_zf + x_rec
        x_rec_dc = self.dc(mask, x_res, x_under)
        recimg.append(sigtoimage(x_rec_dc))
        return recimg


### modified from https://github.com/milesial/Pytorch-UNet ###

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False):
        super(double_conv, self).__init__()
        if dropout:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )


    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, padding):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, padding)


    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, padding, dropout)
        )


    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, padding, dropout)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (math.ceil(diffY / 2), int(diffY / 2),
                        math.ceil(diffX / 2), int(diffX / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)


    def forward(self, x):
        x = self.conv(x)
        return x


# UNet
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32, 1)
        self.down1 = down(32, 64, 1)
        self.down2 = down(64, 128, 1)
        self.up2 = up(128, 64, 1)
        self.up3 = up(64, 32, 1)
        self.outc = outconv(32, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x

################################################################

# Channel-wise attention
class CSE_Block(nn.Module):
    def __init__(self, in_channel, r, w, h):
        super(CSE_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.AvgPool2d((w, h)),
            nn.Conv2d(in_channel, int(in_channel/r), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(in_channel/r), in_channel, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        s = self.layer(x)
        return s*x


# UNet with channel-wise attention, input arguments of CSE_block should change according to image size
class UNetCSE(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetCSE, self).__init__()
        self.inc = inconv(n_channels, 32, 1)
        self.down1 = down(32, 64, 1)
        self.down2 = down(64, 128, 1)
        self.se3 = CSE_Block(128, 8, 40, 36)
        self.up2 = up(128, 64, 1)
        self.se4 = CSE_Block(64, 8, 80, 72)
        self.up1 = up(64, 32, 1)
        self.se5 = CSE_Block(32, 8, 160, 144)
        self.outc = outconv(32, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.se3(x3)
        x = self.up2(x3, x2)
        x = self.se4(x)
        x = self.up1(x, x1)
        x = self.se5(x)
        x = self.outc(x)
        return x