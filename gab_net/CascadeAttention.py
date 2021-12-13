import torch
import torch.nn as nn
from torch.nn import functional as F


class get_up_conv(nn.Module):
    """
    Up Convolution Block for Attention gate
    Args:
        in_ch(int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, activation):
        super(get_up_conv, self).__init__()

        self.activation = activation
        self.conv = convolution_2layers(in_ch + out_ch, out_ch, self.activation)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        # self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        x = torch.cat([inputs1, outputs2], 1)
        return self.conv(x)


def convolution_1layers(in_dim, out_dim, activation):
    """
        Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. """

    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True)


def convolution_2layers(in_dim, out_dim, activation):
    """ A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d)."""
    return nn.Sequential(
        convolution_1layers(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )


class attention_block(nn.Module):
    """
    Attention Block:
    - take g which is the spatially smaller signal (coarser scale), do a conv to get the same number of feature channels
    as x (bigger spatially)
    - do a conv on x to also get same feature channels (theta_x)
    - then, upsample g to be same size as x
    - add x and g (concat_xg)
    - relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients

    """

    """ 
    In the constructor we instantiate two nn.Linear modules and assign them as member variables. 
    """

    """ 
    In the forward function we accept a Tensor of input data and we must return a Tensor of output data. 
    We can use Modules defined in the constructor as well as arbitrary operators on Tensors.
    
    """

    def __init__(self, F_in, F_g, F_inter):
        super(attention_block, self).__init__()

        # linear transformations

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_inter)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_in, F_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_inter)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_inter, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        input_size = x.size()

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        W_x_size = x1.size()

        phi_g = F.interpolate(g1, size=W_x_size[2:], mode='trilinear', align_corners=False)
        psi = self.relu(phi_g + x1)
        psi = self.psi(psi)
        up_psi = F.interpolate(psi, size=input_size[2:], mode='trilinear', align_corners=False)
        out = x * up_psi

        return out


class cascade_UA_3D(nn.Module):
    def __init__(self, in_dim, num_classes=3, feature_scale=8):
        super(cascade_UA_3D, self).__init__()

        self.in_dim = in_dim
        self.feature_scale = feature_scale
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # It is common for a convolutional layer to learn from 32 to 512 filters in parallel for a given input. This
        # gives the model 32, or even 512, different ways of extracting features from an input, or many different
        # ways of both “learning to see” and after training, many different ways of “seeing” the input data.

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # 1) Stage
        # Down-sampling

        self.down_1 = convolution_2layers(self.in_dim, filters[0], self.activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = convolution_2layers(filters[0], filters[1], self.activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = convolution_2layers(filters[1], filters[2], self.activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = convolution_2layers(filters[2], filters[3], self.activation)
        self.pool_4 = max_pooling_3d()
        self.bridge = convolution_2layers(filters[3], filters[4], self.activation)

        # Up sampling
        self.up_4 = get_up_conv(filters[4], filters[3], self.activation)
        self.up_3 = get_up_conv(filters[3], filters[2], self.activation)
        self.up_2 = get_up_conv(filters[2], filters[1], self.activation)
        self.up_1 = get_up_conv(filters[1], filters[0], self.activation)

        # Output
        # self.out1 = convolution_1layers(self.num_filters, out_dim, self.activation)
        # self.out1 = convolution_1layers(filters[0], num_classes, self.activation)
        self.out1 = nn.Conv3d(filters[0], num_classes, 1)

        # 2) Stage

        # Down-sampling

        self.down_6 = convolution_2layers(num_classes, filters[0], self.activation)
        self.pool_6 = max_pooling_3d()
        self.down_7 = convolution_2layers(filters[0], filters[1], self.activation)
        self.pool_7 = max_pooling_3d()
        self.down_8 = convolution_2layers(filters[1], filters[2], self.activation)
        self.pool_8 = max_pooling_3d()
        self.down_9 = convolution_2layers(filters[2], filters[3], self.activation)
        self.pool_9 = max_pooling_3d()
        self.bridge2 = convolution_2layers(filters[3], filters[4], self.activation)

        # gating
        self.gate = convolution_1layers(filters[4], filters[4], self.activation)

        # Attention Block

        self.attention_7 = attention_block(F_in=filters[1], F_g=filters[2], F_inter=filters[1])
        self.attention_8 = attention_block(F_in=filters[2], F_g=filters[3], F_inter=filters[2])
        self.attention_9 = attention_block(F_in=filters[3], F_g=filters[4], F_inter=filters[3])

        # Upsampling

        self.up_9 = get_up_conv(filters[4], filters[3], self.activation)
        self.up_8 = get_up_conv(filters[3], filters[2], self.activation)
        self.up_7 = get_up_conv(filters[2], filters[1], self.activation)
        self.up_6 = get_up_conv(filters[1], filters[0], self.activation)

        # output
        self.out2 = nn.Conv3d(filters[0], num_classes, 1)

    def forward(self, x):
        # 1 Stage

        # Down-sampling
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        # center
        bridge1 = self.bridge(pool_4)

        # Up sampling

        up_4 = self.up_4(down_4, bridge1)
        # print('up_4 {}'.format(up_4.shape))
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)

        # Output
        out1 = self.out1(up_1)
        print('out1 {}'.format(out1.shape))

        # Second stage

        # Down-sampling
        down_6 = self.down_6(out1)
        pool_6 = self.pool_6(down_6)

        down_7 = self.down_7(pool_6)
        # print('down7: {}'.format(down_7.shape))
        pool_7 = self.pool_7(down_7)

        down_8 = self.down_8(pool_7)
        pool_8 = self.pool_8(down_8)

        down_9 = self.down_9(pool_8)
        # print('down9: {}'.format(down_9.shape))
        pool_9 = self.pool_9(down_9)
        # print('pool9: {}'.format(pool_9.shape))

        bridge2 = self.bridge2(pool_9)
        # print('bridge2 {}'.format(bridge2.shape))
        gate = self.gate(bridge2)
        # print('gate {}'.format(gate.shape))

        # Up-sampling + Attention Block

        att9 = self.attention_9(x=down_9, g=bridge2)
        # print('att9 {}'.format(att9.shape))
        up_9 = self.up_9(att9, bridge2)
        # print('up_9 {}'.format(up_9.shape))

        att8 = self.attention_8(x=down_8, g=up_9)
        # print('att8 {}'.format(att8.shape))
        up_8 = self.up_8(att8, up_9)
        # print('up_8 {}'.format(up_8.shape))

        att7 = self.attention_7(x=down_7, g=up_8)
        # print('att7 {}'.format(att7.shape))
        up_7 = self.up_7(att7, up_8)
        # print('up_7 {}'.format(up_7.shape))
        # up_7 = self.up_7(down_7, up_8)

        up_6 = self.up_6(down_6, up_7)

        # Output
        out2 = self.out2(up_6)

        return out2

    # softmax returns a tensor of the same dimension and shape as the input with values in the range [0, 1]
    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


