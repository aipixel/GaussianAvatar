import numbers
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class CBatchNorm2d(nn.Module):
    ''' Conditional batch normalization layer class.
        Borrowed from Occupancy Network repo: https://github.com/autonomousvision/occupancy_networks
    Args:
        c_dim (int): dimension of latent conditioned code c
        f_channels (int): number of channels of the feature maps
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_channels, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_channels = f_channels
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_channels, 1) # match the cond dim to num of feature channels
        self.conv_beta = nn.Conv1d(c_dim, f_channels, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm2d(f_channels, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm2d(f_channels, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm2d(f_channels, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x 1 (conv1d needs the 3rd dim)
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c).unsqueeze(-1) # make gamma be of shape [batch, f_dim, 1, 1]
        beta = self.conv_beta(c).unsqueeze(-1)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class Conv2DBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1, use_bias=False, use_bn=True, use_relu=True):
        super(Conv2DBlock, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        if use_bn:
            self.bn = nn.BatchNorm2d(output_nc, affine=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.use_relu:
            x = self.relu(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class UpConv2DBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1,
                 use_bias=False, use_bn=True, up_mode='upconv', use_dropout=False):
        super(UpConv2DBlock, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.relu = nn.ReLU()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=use_bias)
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, stride=1),
            )
        if use_bn:
            self.bn = nn.BatchNorm2d(output_nc, affine=False)
        if use_dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x, skip_input=None):
        x = self.relu(x)
        x = self.up(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_dropout:
            x = self.drop(x)
        if skip_input is not None:
            x = torch.cat([x, skip_input], 1)
        return x


class GeomConvLayers(nn.Module):
    '''
    A few convolutional layers to smooth the geometric feature tensor
    '''
    def __init__(self, input_nc=16, hidden_nc=16, output_nc=16, use_relu=False):
        super().__init__()
        self.use_relu = use_relu

        self.conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(hidden_nc, hidden_nc, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(hidden_nc, output_nc, kernel_size=5, stride=1, padding=2, bias=False)
        if use_relu:
            self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.conv2(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.conv3(x)

        return x


class GeomConvBottleneckLayers(nn.Module):
    '''
    A u-net-like small bottleneck network for smoothing the geometric feature tensor
    '''
    def __init__(self, input_nc=16, hidden_nc=16, output_nc=16, use_relu=False):
        super().__init__()
        self.use_relu = use_relu

        self.conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(hidden_nc, hidden_nc*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(hidden_nc*2, hidden_nc*4, kernel_size=4, stride=2, padding=1, bias=False)

        self.up1 = nn.ConvTranspose2d(hidden_nc*4, hidden_nc*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up2 = nn.ConvTranspose2d(hidden_nc*2, hidden_nc, kernel_size=4, stride=2, padding=1, bias=False)
        self.up3 = nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        return x


# class GaussianSmoothingLayers(nn.Module):
#     '''
#     use a fixed, not-trainable gaussian smoother layers for smoothing the geometric feature tensor
#     '''
#     def __init__(self, channels=16, kernel_size=5, sigma=1.0):
#         super().__init__()
#         self.conv1 = GaussianSmoothing(channels, kernel_size=kernel_size, sigma=1.0, dim=2)
#         self.conv2 = GaussianSmoothing(channels, kernel_size=kernel_size, sigma=1.0, dim=2)
#         self.conv3 = GaussianSmoothing(channels, kernel_size=kernel_size, sigma=1.0, dim=2)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         return x


class UnetNoCond5DS(nn.Module):
    # 5DS: downsample 5 times, for posmap size=32
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, 
                return_lowres=False, return_2branches=False):
        super().__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        self.upconv3 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512

        # Coord regressor
        self.upconv4 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode=up_mode) # 16
        self.upconv5 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode=up_mode) # 32

        if return_2branches:
            self.upconvN4 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode=up_mode) # 16
            self.upconvN5 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 32

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)

        u1 = self.upconv1(d5, d4)
        u2 = self.upconv2(u1, d3)
        u3 = self.upconv3(u2, d2)

        u4 = self.upconv4(u3, d1)
        u5 = self.upconv5(u4)

        if self.return_2branches:
            uN4 = self.upconvN4(u3, d1)
            uN5 = self.upconvN5(uN4)
            return u5, uN5

        return u5


class UnetNoCond6DS(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, return_lowres=False, return_2branches=False):
        super(UnetNoCond6DS, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv6 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        self.upconv3 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512
        self.upconv4 = UpConv2DBlock(4 * nf * 3, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512

        # Coord regressor
        self.upconvC5 = UpConv2DBlock(2 * nf * 3, 2 * nf, 4, 2, 1, up_mode='upsample') # 16
        self.upconvC6 = UpConv2DBlock(1 * nf * 3, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upsample') # 64x64, 128

        if return_2branches:
            # Normal regressor
            self.upconvN5 = UpConv2DBlock(2 * nf * 3, 2 * nf, 4, 2, 1, up_mode='upconv') # 32x32, 256
            self.upconvN6 = UpConv2DBlock(1 * nf * 3, 3, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 64x64, 128

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)

        # shared decoder layers
        u1 = self.upconv1(d6, d5)
        u2 = self.upconv2(u1, d4)
        u3 = self.upconv3(u2, d3)
        u4 = self.upconv4(u3, d2)

        # coord regressor
        uc5 = self.upconvC5(u4, d1)
        uc6 = self.upconvC6(uc5)

        if self.return_2branches:
            # normal regressor
            un5 = self.upconvN5(u4, d1)
            un6 = self.upconvN6(un5)

            return uc6, un6

        return uc6

class Unet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, return_lowres=False, return_2branches=False):
        super(Unet, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv6 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv7 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        self.upconv3 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512
        self.upconv4 = UpConv2DBlock(8 * nf * 2, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512

        # Coord regressor
        self.upconvC5 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode='upsample') # 16
        self.upconvC6 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode='upsample') # 32
        self.upconvC7 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upsample') # 64x64, 128

        if return_2branches:
            # Normal regressor
            self.upconvN5 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode='upconv') # 32x32, 256
            self.upconvN6 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode='upconv') # 64x64, 128
            self.upconvN7 = UpConv2DBlock(1 * nf * 2, 3, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 64x64, 128
        
        self.head_xyz_conv1 = torch.nn.Conv1d(output_nc, output_nc, 1)
        self.head_xyz_conv2 = torch.nn.Conv1d(output_nc, int(output_nc/2), 1)
        self.head_xyz_conv3 = torch.nn.Conv1d(int(output_nc/2), 3, 1)

        self.head_color_conv1 = torch.nn.Conv1d(output_nc, output_nc, 1)
        self.head_color_conv2 = torch.nn.Conv1d(output_nc, int(output_nc/2), 1)
        self.head_color_conv3 = torch.nn.Conv1d(int(output_nc/2), 3, 1)

        self.head_scale_conv1 = torch.nn.Conv1d(output_nc, output_nc, 1)
        self.head_scale_conv2 = torch.nn.Conv1d(output_nc, int(output_nc/2), 1)
        self.head_scale_conv3 = torch.nn.Conv1d(int(output_nc/2), 1, 1)

        self.bn1 = torch.nn.BatchNorm1d(output_nc)
        self.bn2 = torch.nn.BatchNorm1d(int(output_nc/2))

        self.bn1N = torch.nn.BatchNorm1d(output_nc)
        self.bn2N = torch.nn.BatchNorm1d(int(output_nc/2))  

        self.bn1SH = torch.nn.BatchNorm1d(output_nc)
        self.bn2SH = torch.nn.BatchNorm1d(int(output_nc/2))

        self.tan = nn.Tanh()
        self.actv_fn = nn.Softplus()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)
        d7 = self.conv7(d6)

        # shared decoder layers
        u1 = self.upconv1(d7, d6)
        u2 = self.upconv2(u1, d5)
        u3 = self.upconv3(u2, d4)
        u4 = self.upconv4(u3, d3)

        # coord regressor
        uc5 = self.upconvC5(u4, d2)
        uc6 = self.upconvC6(uc5, d1)
        uc7 = self.upconvC7(uc6)

        if self.return_2branches:
            # normal regressor
            un5 = self.upconvN5(u4, d2)
            un6 = self.upconvN6(un5, d1)
            un7 = self.upconvN7(un6)

            return uc7, un7
        
        B, C, H, W = uc7.shape
        feature = uc7.view(B, C, -1)
        
        x1 = self.actv_fn(self.bn1(self.head_xyz_conv1(feature)))
        x2 = self.actv_fn(self.bn2(self.head_xyz_conv2(x1)))
        x3 = self.head_xyz_conv3(x2)

        # scales pred
        x1N = self.actv_fn(self.bn1N(self.head_scale_conv1(feature)))
        x2N = self.actv_fn(self.bn2N(self.head_scale_conv2(x1N)))
        x3N = self.head_scale_conv3(x2N)

        # color pred
        x1SH = self.actv_fn(self.bn1SH(self.head_color_conv1(feature)))
        x2SH = self.actv_fn(self.bn2SH(self.head_color_conv2(x1SH)))
        x3SH = self.head_color_conv3(x2SH)

        return x3, x3N, x3SH



class UnetNoCond7DS(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, return_lowres=False, return_2branches=False):
        super(UnetNoCond7DS, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv6 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv7 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        self.upconv3 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512
        self.upconv4 = UpConv2DBlock(8 * nf * 2, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512

        # Coord regressor
        self.upconvC5 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode='upsample') # 16
        self.upconvC6 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode='upsample') # 32
        self.upconvC7 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upsample') # 64x64, 128

        if return_2branches:
            # Normal regressor
            self.upconvN5 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode='upconv') # 32x32, 256
            self.upconvN6 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode='upconv') # 64x64, 128
            self.upconvN7 = UpConv2DBlock(1 * nf * 2, 3, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 64x64, 128

 

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)
        d7 = self.conv7(d6)

        # shared decoder layers
        u1 = self.upconv1(d7, d6)
        u2 = self.upconv2(u1, d5)
        u3 = self.upconv3(u2, d4)
        u4 = self.upconv4(u3, d3)

        # coord regressor
        uc5 = self.upconvC5(u4, d2)
        uc6 = self.upconvC6(uc5, d1)
        uc7 = self.upconvC7(uc6)

        if self.return_2branches:
            # normal regressor
            un5 = self.upconvN5(u4, d2)
            un6 = self.upconvN6(un5, d1)
            un7 = self.upconvN7(un6)

            return uc7, un7

        return uc7
    
class activate_module(nn.Module):
    def __init__(self, hsize = 256, actv_fn='softplus'):
        self.hsize = hsize
        super(activate_module, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.actv_fn = nn.ReLU() if actv_fn=='relu' else nn.Softplus()

        self.conv6 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8 = torch.nn.Conv1d(self.hsize, 3, 1)
        self.conv6N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8N = torch.nn.Conv1d(self.hsize, 11, 1)


        self.bn1 = torch.nn.BatchNorm1d(self.hsize)
        self.bn2 = torch.nn.BatchNorm1d(self.hsize)
        self.bn3 = torch.nn.BatchNorm1d(self.hsize)
        self.bn4 = torch.nn.BatchNorm1d(self.hsize)

        self.bn5 = torch.nn.BatchNorm1d(self.hsize)
        self.bn6 = torch.nn.BatchNorm1d(self.hsize)
        self.bn7 = torch.nn.BatchNorm1d(self.hsize)

        self.bn6N = torch.nn.BatchNorm1d(self.hsize)
        self.bn7N = torch.nn.BatchNorm1d(self.hsize)  
    
    def forward(self, x5):
                # position pred
        x6 = self.actv_fn(self.bn6(self.conv6(x5)))
        x7 = self.actv_fn(self.bn7(self.conv7(x6)))
        x8 = self.conv8(x7)

        # rotations pred
        xN6 = self.actv_fn(self.bn6N(self.conv6N(x5)))
        xN7 = self.actv_fn(self.bn7N(self.conv7N(xN6)))
        xN8 = self.conv8N(xN7)

        rotations = xN8[:,:4,:]
        scales = self.sigmoid(xN8[:,4:7,:])
        # scales = -1 * self.actv_fn(xN8[:,4:7,:])
        opacitys = self.sigmoid(xN8[:,7:8,:])
        shs = self.sigmoid(xN8[:,8:,:])

        return x8, rotations, scales, opacitys, shs


class ShapeDecoder(nn.Module):
    '''
    The "Shape Decoder" in the POP paper Fig. 2. The same as the "shared MLP" in the SCALE paper.
    - with skip connection from the input features to the 4th layer's output features (like DeepSDF)
    - branches out at the second-to-last layer, one branch for position pred, one for normal pred
    '''
    def __init__(self, in_size, hsize = 256, actv_fn='softplus'):
        self.hsize = hsize
        super(ShapeDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_size, self.hsize, 1)
        self.conv2 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv3 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv4 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv5 = torch.nn.Conv1d(self.hsize+in_size, self.hsize, 1)
        self.conv6 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8 = torch.nn.Conv1d(self.hsize, 3, 1)

        self.conv6SH = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7SH = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8SH = torch.nn.Conv1d(self.hsize, 3, 1)

        self.conv6N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8N = torch.nn.Conv1d(self.hsize, 1, 1)


        self.bn1 = torch.nn.BatchNorm1d(self.hsize)
        self.bn2 = torch.nn.BatchNorm1d(self.hsize)
        self.bn3 = torch.nn.BatchNorm1d(self.hsize)
        self.bn4 = torch.nn.BatchNorm1d(self.hsize)

        self.bn5 = torch.nn.BatchNorm1d(self.hsize)
        self.bn6 = torch.nn.BatchNorm1d(self.hsize)
        self.bn7 = torch.nn.BatchNorm1d(self.hsize)

        self.bn6N = torch.nn.BatchNorm1d(self.hsize)
        self.bn7N = torch.nn.BatchNorm1d(self.hsize)  

        self.bn6SH = torch.nn.BatchNorm1d(self.hsize)
        self.bn7SH = torch.nn.BatchNorm1d(self.hsize)

        self.actv_fn = nn.ReLU() if actv_fn=='relu' else nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.tan = nn.Tanh()

    def forward(self, x):
        x1 = self.actv_fn(self.bn1(self.conv1(x)))
        x2 = self.actv_fn(self.bn2(self.conv2(x1)))
        x3 = self.actv_fn(self.bn3(self.conv3(x2)))
        x4 = self.actv_fn(self.bn4(self.conv4(x3)))
        x5 = self.actv_fn(self.bn5(self.conv5(torch.cat([x,x4],dim=1))))

        # position pred
        x6 = self.actv_fn(self.bn6(self.conv6(x5)))
        x7 = self.actv_fn(self.bn7(self.conv7(x6)))
        x8 = self.conv8(x7)

        # scales pred
        xN6 = self.actv_fn(self.bn6N(self.conv6N(x5)))
        xN7 = self.actv_fn(self.bn7N(self.conv7N(xN6)))
        xN8 = self.conv8N(xN7)

        # shs pred
        xSH6 = self.actv_fn(self.bn6SH(self.conv6SH(x5)))
        xSH7 = self.actv_fn(self.bn7SH(self.conv7SH(xSH6)))
        xSH8 = self.conv8SH(xSH7)

        # rotations = xN8[:,:4,:]
        scales = self.sigmoid(xN8)
        # scales = -1 * self.actv_fn(xN8[:,4:7,:])
        # opacitys = self.sigmoid(xN8[:,7:8,:])
        shs = self.sigmoid(xSH8)

        return x8, scales, shs


def gen_transf_mtx_full_uv(verts, faces):
    '''
    given a positional uv map, for each of its pixel, get the matrix that transforms the prediction from local to global coordinates
    The local coordinates are defined by the posed body mesh (consists of vertcs and faces)

    :param verts: [batch, Nverts, 3]
    :param faces: [uv_size, uv_size, 3], uv_size =e.g. 32
    
    :return: [batch, uv_size, uv_size, 3,3], per example a map of 3x3 rot matrices for local->global transform

    NOTE: local coords are NOT cartesian! uu an vv axis are edges of the triangle,
          not perpendicular (more like barycentric coords)
    '''
    tris = verts[:, faces] # [batch, uv_size, uv_size, 3, 3]
    v1, v2, v3 = tris[:, :, :, 0, :], tris[:, :, :, 1, :], tris[:, :, :, 2, :]
    uu = v2 - v1 # u axis of local coords is the first edge, [batch, uv_size, uv_size, 3]
    vv = v3 - v1 # v axis, second edge
    ww_raw = torch.cross(uu, vv, dim=-1)
    ww = F.normalize(ww_raw, p=2, dim=-1) # unit triangle normal as w axis
    ww_norm = (torch.norm(uu, dim=-1).mean(-1).mean(-1) + torch.norm(vv, dim=-1).mean(-1).mean(-1)) / 2.
    ww = ww*ww_norm.view(len(ww_norm),1,1,1)
    
    # shape of transf_mtx will be [batch, uv_size, uv_size, 3, 3], where the last two dim is like:
    #  |   |   |
    #[ uu  vv  ww]
    #  |   |   |
    # for local to global, say coord in the local coord system is (r,s,t)
    # then the coord in world system should be r*uu + s*vv+ t*ww
    # so the uu, vv, ww should be colum vectors of the local->global transf mtx
    # so when stack, always stack along dim -1 (i.e. column)
    transf_mtx = torch.stack([uu, vv, ww], dim=-1)

    return transf_mtx


def gen_transf_mtx_from_vtransf(vtransf, bary_coords, faces, scaling=1.0):
    '''
    interpolate the local -> global coord transormation given such transformations defined on 
    the body verts (pre-computed) and barycentric coordinates of the query points from the uv map.

    Note: The output of this function, i.e. the transformation matrix of each point, is not a pure rotation matrix (SO3).
    
    args:
        vtransf: [batch, #verts, 3, 3] # per-vertex rotation matrix
        bary_coords: [uv_size, uv_size, 3] # barycentric coordinates of each query point (pixel) on the query uv map 
        faces: [uv_size, uv_size, 3] # the vert id of the 3 vertices of the triangle where each uv pixel locates

    returns: 
        [batch, uv_size, uv_size, 3, 3], transformation matrix for points on the uv surface
    '''
    #  
    vtransf_by_tris = vtransf[:, faces] # shape will be [batch, uvsize, uvsize, 3, 3, 3], where the the last 2 dims being the transf (pure rotation) matrices, the other "3" are 3 points of each triangle
    transf_mtx_uv_pts = torch.einsum('bpqijk,pqi->bpqjk', vtransf_by_tris, bary_coords) # [batch, uvsize, uvsize, 3, 3], last 2 dims are the rotation matix
    transf_mtx_uv_pts *= scaling
    return transf_mtx_uv_pts




class Embedder():
    '''
    Simple positional encoding, adapted from NeRF: https://github.com/bmild/nerf
    '''
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    '''
    Helper function for positional encoding, adapted from NeRF: https://github.com/bmild/nerf
    '''
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class PositionalEncoding():
    def __init__(self, input_dims=2, num_freqs=10, include_input=False):
        super(PositionalEncoding,self).__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.input_dims = input_dims

    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        freq_bands = 2. ** torch.linspace(0, self.num_freqs-1, self.num_freqs)
        periodic_fns = [torch.sin, torch.cos]

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(math.pi * x * freq))
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self,coords):
        '''
        use periodic positional encoding to transform cartesian positions to higher dimension
        :param coords: [N, 3]
        :return: [N, 3*2*num_freqs], where 2 comes from that for each frequency there's a sin() and cos()
        '''
        return torch.cat([fn(coords) for fn in self.embed_fns], dim=-1)


def normalize_uv(uv):
    '''
    normalize uv coords from range [0,1] to range [-1,1]
    '''
    return uv * 2. - 1.


def uv_to_grid(uv_idx_map, resolution):
    '''
    uv_idx_map: shape=[batch, N_uvcoords, 2], ranging between 0-1
    this function basically reshapes the uv_idx_map and shift its value range to (-1, 1) (required by F.gridsample)
    the sqaure of resolution = N_uvcoords
    '''
    bs = uv_idx_map.shape[0]
    grid = uv_idx_map.reshape(bs, resolution, resolution, 2) * 2 - 1.
    grid = grid.transpose(1,2)
    return grid

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim
