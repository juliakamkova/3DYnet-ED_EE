import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import torch
import torch.nn as nn
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset


class RepeatConv(nn.Module) :
    """
    Repeat Conv + PReLU n times
    """

    def __init__(self, n_channels, n_conv) :
        super(RepeatConv, self).__init__()

        conv_list = []
        for _ in range(n_conv) :
            conv_list.append(nn.Conv3d(n_channels, n_channels, kernel_size=5, padding=2))
            conv_list.append(nn.PReLU())

        self.conv = nn.Sequential(
            *conv_list
        )

    def forward(self, x) :
        return self.conv(x)


class Down(nn.Module) :
    def __init__(self, in_channels, out_channels, n_conv) :
        super(Down, self).__init__()

        self.downconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = RepeatConv(out_channels, n_conv)

    def forward(self, x) :
        out = self.downconv(x)
        return out + self.conv(out)


class Up(nn.Module) :
    def __init__(self, in_channels, out_channels, n_conv) :
        super(Up, self).__init__()

        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, int(out_channels / 2), kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = RepeatConv(out_channels, n_conv)

    def forward(self, x, down) :
        x = self.upconv(x)
        cat = torch.cat([x, down], dim=1)
        return cat + self.conv(cat)


class VNet(nn.Module) :
    """
    Main model
    """

    def __init__(self, in_channels, num_class) :
        super(VNet, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=5, padding=2),
            nn.PReLU()
        )

        self.down2 = Down(16, 32, 2)
        self.down3 = Down(32, 64, 3)
        self.down4 = Down(64, 128, 3)
        self.down5 = Down(128, 256, 3)

        self.up1 = Up(256, 256, 3)
        self.up2 = Up(256, 128, 3)
        self.up3 = Up(128, 64, 2)
        self.up4 = Up(64, 32, 1)

        self.up5 = nn.Sequential(
            nn.Conv3d(32, num_class, kernel_size=1),
            nn.PReLU()
        )

    def forward(self, x) :
        down1 = self.down1(x) + torch.cat(16 * [x], dim=1)
        #print('down', down1.shape)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.down5(down4)
        up1 = self.up1(center, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)
        return self.up5(up4)


# def init(module) :
#     if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d) :
#         nn.init.kaiming_normal_(module.weight.data, 0.25)
#         nn.init.constant_(module.bias.data, 0)
#
#
# net = VNet(in_channels=1, num_class=3)
# net.apply(init)
#
# # Output data dimension check
# net = net.cuda()
# data = torch.randn((1, 1, 96, 96, 96)).cuda()
# label = torch.randint(0, 2, (1, 1, 96, 96, 96)).cuda()
# # print('label', label.shape)
# res = net(data)
# for item in res :
#     print(item.size())
#
# # Calculate network parameters
# num_parameter = .0
# for item in net.modules() :
#
#     if isinstance(item, nn.Conv3d) or isinstance(item, nn.ConvTranspose3d) :
#         num_parameter += (item.weight.size(0) * item.weight.size(1) *
#                           item.weight.size(2) * item.weight.size(3) * item.weight.size(4))
#
#         if item.bias is not None :
#             num_parameter += item.bias.size(0)
#
#     elif isinstance(item, nn.PReLU) :
#         num_parameter += item.num_parameters
#
# print(num_parameter)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# iters = 0
# # training simulation
# for epoch in range(20) :  # loop over the dataset multiple times
#     inputs = data
#     masks = label
#     for i in range(len(inputs)) :
#         running_loss = 0.0
#         # get the inputs; data is a list of [inputs, labels]
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         #print(masks)
#         loss = criterion(outputs, masks[i])
#         #print('mask i', masks[i])
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         iters += 1
#
#         if iters % 2 == 0 :
#             print('Prev Loss: {:.4f} Prev Acc: {:.4f}'.format(
#                 loss.item(), torch.sum(outputs == masks) / inputs.size(0)))
#             epoch_loss = running_loss / (len(inputs))
#         # if i % 2000 == 1999:  # print every 2000 mini-batches
#         #     print('[%d, %5d] loss: %.3f' %
#         #           (epoch + 1, i + 1, running_loss / 2000))
#         #     running_loss = 0.0
#

