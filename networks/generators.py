import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F



""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):
        """U-Net  #https://github.com/milesial/Pytorch-UNet
        """
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out


class GatedFeatureForwardUnit(nn.Module):
    def __init__(self, negative_slope=0.01, bn=True, attention=False):
        super(GatedFeatureForwardUnit, self).__init__()
        self.conv1 = GatedConv2dWithActivation(2, 32, 3, padding=1)
        self.conv2 = GatedConv2dWithActivation(32, 32, 3, padding=1)
        self.conv3 = GatedConv2dWithActivation(32, 32, 3, padding=1)
        self.conv4 = GatedConv2dWithActivation(32, 32, 3, padding=1)
        # self.conv5 = Sequential
        #     nn.Conv2d(32, 32, 3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv6 = nn.Conv2d(32, 2, 3, padding=1)
        self.ac6 = nn.LeakyReLU(negative_slope=negative_slope)
        if attention:
            self.attention = AttentionWeightChannel(channel_num=32)
        else:
            self.attention = None
    
    def forward(self, x):
        if self.attention is not None:
            out1 = self.conv1(x)
            out2 = self.attention(self.conv2(out1))
            out3 = self.attention(self.conv3(out2))
            out4 = self.attention(self.conv4(out3))
            # out5 = self.conv5(out4)
            out6 = self.conv6(out4)
            output = self.ac6(out6 + x)
        else:
            out1 = self.conv1(x)
            out2 = self.conv2(out1)
            out3 = self.conv3(out2)
            out4 = self.conv4(out3)
            # out5 = self.conv5(out4)
            out6 = self.conv6(out4)
            output = self.ac6(out6 + x)

        return output

# MD-net的卷积网络 用于图像域
class FeatureForwardUnit(nn.Module):
    def __init__(self, in_channels, out_channels,negative_slope=0.01, bn=True, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False), attention=False):
        super(FeatureForwardUnit, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            norm_layer(32),
            nn.LeakyReLU(negative_slope=negative_slope))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            norm_layer(32),
            nn.LeakyReLU(negative_slope=negative_slope))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            norm_layer(32),
            nn.LeakyReLU(negative_slope=negative_slope))
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            norm_layer(32),
            nn.LeakyReLU(negative_slope=negative_slope))
        # self.conv4 = SpadeBlock(in_channels = 32,out_channels = 32, ks=3, pw=1, norm_nc=32, label_nc=in_channels, negative_slope=negative_slope)
        # self.conv5 = Sequential(
        #     nn.Conv2d(32, 32, 3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
    
        self.conv6 = nn.Conv2d(32, out_channels, 3, padding=1)
        self.ac6 = nn.LeakyReLU(negative_slope=negative_slope)
        if attention:
            self.attention = AttentionWeightChannel(32)
        else:
            self.attention = None
    
    def forward(self, x):
        if self.attention is not None:
            out1 = self.conv1(x)
            out2 = self.attention(self.conv2(out1))
            out3 = self.attention(self.conv3(out2))
            out4 = self.attention(self.conv4(out3))
            # out5 = self.conv5(out4)
            out6 = self.conv6(out4)
            output = self.ac6(out6 + x)
        else:
            out1 = self.conv1(x)
            out2 = self.conv2(out1)
            out3 = self.conv3(out2)
            out4 = self.conv4(out3)
            # out5 = self.conv5(out4)
            out6 = self.conv6(out4)
            output = self.ac6(out6 + x)
        
        return output


class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False), activation=nn.LeakyReLU(0.2, inplace=True), norm_nc=32, label_nc=2):
        super(GatedConv2dWithActivation, self).__init__()
        
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.norm_layer = torch.nn.BatchNorm2d(out_channels)
        # self.spade = SPADE(norm_nc=norm_nc, label_nc=label_nc)
        self.norm_layer = norm_layer(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        return self.norm_layer(x)
