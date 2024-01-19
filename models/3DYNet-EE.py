import torch.nn as nn
import torch
import pytorch_lightning as pl

CUDA_LAUNCH_BLOCKING = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.batchNorm = nn.BatchNorm3d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.selu(x)
        return x


class VGGNet(nn.Module):
    def __init__(self, in_channels=1, VGG_CHANNELS=[64, 128, 256, 512, 512]):
        super().__init__()
        self.in_channels = in_channels

        # Block 0:
        self.conv_0_0 = nn.Conv3d(self.in_channels, VGG_CHANNELS[0], kernel_size=(3, 3, 3), padding=1)
        self.relu_0_0 = nn.ReLU()
        self.down_0aT = nn.Conv3d(VGG_CHANNELS[0], VGG_CHANNELS[0], kernel_size=(3, 3, 3), padding=1)
        self.relu_0aT = nn.ReLU()
        self.maxp_0 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        # Block 1:
        self.conv_1_0 = nn.Conv3d(VGG_CHANNELS[0], VGG_CHANNELS[1], kernel_size=(3, 3, 3), padding=1)
        self.relu_1_0 = nn.ReLU()
        self.down_1aT = nn.Conv3d(VGG_CHANNELS[1], VGG_CHANNELS[1], kernel_size=(3, 3, 3), padding=1)
        self.relu_1aT = nn.ReLU()
        self.maxp_1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        # Block 2:
        self.conv_2_0 = nn.Conv3d(VGG_CHANNELS[1], VGG_CHANNELS[2], kernel_size=(3, 3, 3), padding=1)
        self.relu_2_0 = nn.ReLU()
        self.conv_2_1 = nn.Conv3d(VGG_CHANNELS[2], VGG_CHANNELS[2], kernel_size=(3, 3, 3), padding=1)
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv3d(VGG_CHANNELS[2], VGG_CHANNELS[2], kernel_size=(3, 3, 3), padding=1)
        self.relu_2_2 = nn.ReLU()
        self.down_2aT = nn.Conv3d(VGG_CHANNELS[2], VGG_CHANNELS[2], kernel_size=(3, 3, 3), padding=1)
        self.relu_2aT = nn.ReLU()
        self.maxp_2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        # Block 3:
        self.conv_3_0 = nn.Conv3d(VGG_CHANNELS[2], VGG_CHANNELS[3], kernel_size=(3, 3, 3), padding=1)
        self.relu_3_0 = nn.ReLU()
        self.conv_3_1 = nn.Conv3d(VGG_CHANNELS[3], VGG_CHANNELS[3], kernel_size=(3, 3, 3), padding=1)
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv3d(VGG_CHANNELS[3], VGG_CHANNELS[3], kernel_size=(3, 3, 3), padding=1)
        self.relu_3_2 = nn.ReLU()
        self.down_3aT = nn.Conv3d(VGG_CHANNELS[3], VGG_CHANNELS[3], kernel_size=(3, 3, 3), padding=1)
        self.relu_3aT = nn.ReLU()
        self.maxp_3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        # Block 4:
        self.conv_4_0 = nn.Conv3d(VGG_CHANNELS[3], VGG_CHANNELS[4], kernel_size=(3, 3, 3), padding=1)
        self.relu_4_0 = nn.ReLU()
        self.conv_4_1 = nn.Conv3d(VGG_CHANNELS[4], VGG_CHANNELS[4], kernel_size=(3, 3, 3), padding=1)
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = nn.Conv3d(VGG_CHANNELS[4], VGG_CHANNELS[4], kernel_size=(3, 3, 3), padding=1)
        self.relu_4_2 = nn.ReLU()
        self.down_4aT = nn.Conv3d(VGG_CHANNELS[4], VGG_CHANNELS[4], kernel_size=(3, 3, 3), padding=1)
        self.relu_4aT = nn.ReLU()
        self.maxp_4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x = self.relu_0_0(self.conv_0_0(x))
        down0 = self.relu_0aT(self.down_0aT(x))
        x = self.maxp_0(down0)

        x = self.relu_1_0(self.conv_1_0(x))
        down1 = self.relu_1aT(self.down_1aT(x))  # change conv_1_at with down
        x = self.maxp_1(down1)

        x = self.relu_2_0(self.conv_2_0(x))
        x = self.relu_2_1(self.conv_2_1(x))
        x = self.relu_2_2(self.conv_2_2(x))
        down2 = self.relu_2aT(self.down_2aT(x))
        x = self.maxp_2(down2)

        x = self.relu_3_0(self.conv_3_0(x))
        x = self.relu_3_1(self.conv_3_1(x))
        x = self.relu_3_2(self.conv_3_2(x))
        down3 = self.relu_3aT(self.down_3aT(x))
        x = self.maxp_3(down3)

        x = self.relu_4_0(self.conv_4_0(x))
        x = self.relu_4_1(self.conv_4_1(x))
        x = self.relu_4_2(self.conv_4_2(x))
        down4 = self.relu_4aT(self.down_4aT(x))
        x = self.maxp_4(down4)
        print('x', x.shape)

        return x, down0, down1, down2, down3, down4


class YNet(nn.Module):
    """ Warning: Check your learning rate. The bigger your network, more parameters to learn.
    That means you also need to decrease the learning rate."""

    def __init__(self, n_class=3):
        super().__init__()

        CHANNELS = [64, 128, 256, 512, 1024]

        self.vggnet_0 = VGGNet(in_channels=1)
        self.vggnet_1 = VGGNet(in_channels=1)

        self.convBlock_c0 = ConvBlock(CHANNELS[4], CHANNELS[4])
        self.convBlock_c1 = ConvBlock(CHANNELS[4], CHANNELS[4])

        self.upsampler_4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock_4_0 = ConvBlock(2 * CHANNELS[4], CHANNELS[3])
        self.convBlock_4_1 = ConvBlock(CHANNELS[3], CHANNELS[3])
        self.convBlock_4_2 = ConvBlock(CHANNELS[3], CHANNELS[3])

        self.upsampler_3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock_3_0 = ConvBlock(CHANNELS[4] + CHANNELS[3], CHANNELS[3])
        self.convBlock_3_1 = ConvBlock(CHANNELS[3], CHANNELS[3])
        self.convBlock_3_2 = ConvBlock(CHANNELS[3], CHANNELS[2])

        self.upsampler_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock_2_0 = ConvBlock(CHANNELS[3] + CHANNELS[2], CHANNELS[2])
        self.convBlock_2_1 = ConvBlock(CHANNELS[2], CHANNELS[2])
        self.convBlock_2_2 = ConvBlock(CHANNELS[2], CHANNELS[1])

        self.upsampler_1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock_1_0 = ConvBlock(CHANNELS[2] + CHANNELS[1], CHANNELS[1])
        self.convBlock_1_1 = ConvBlock(CHANNELS[1], CHANNELS[1])
        self.convBlock_1_2 = ConvBlock(CHANNELS[1], CHANNELS[0])

        self.upsampler_0 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock_0_0 = ConvBlock(CHANNELS[1] + CHANNELS[0], CHANNELS[0])
        self.convBlock_0_1 = ConvBlock(CHANNELS[0], CHANNELS[0])
        self.convBlock_0_2 = ConvBlock(CHANNELS[0], CHANNELS[0])

        # final conv (without any concat)
        self.final = nn.Conv3d(CHANNELS[0], n_class, 1)

    def forward(self, x):
        x0, down0_0, down1_0, down2_0, down3_0, down4_0 = self.vggnet_0(x)
        x1, down0_1, down1_1, down2_1, down3_1, down4_1 = self.vggnet_1(x)

        print('x0', x0.shape)
        print('x1', x1.shape)
        center = torch.cat([x0, x1], dim=1)
        # print('center', center.shape)
        center = self.convBlock_c0(center)
        center = self.convBlock_c1(center)
        print('center final', center.shape)

        up4 = self.upsampler_4(center)
        # print('up4', up4.shape)
        down4 = torch.cat([down4_0, down4_1], dim=1)
        # print('down4_0', down4_0.shape)
        # print('down4-1', down4_1.shape)
        print('down4', down4.shape)
        up4 = torch.cat([down4, up4], dim=1)
        up4 = self.convBlock_4_0(up4)
        up4 = self.convBlock_4_1(up4)
        up4 = self.convBlock_4_2(up4)
        print('up4 FINAL', up4.shape)

        up3 = self.upsampler_3(up4)
        down3 = torch.cat([down3_0, down3_1], dim=1)
        print('down3', down3.shape)
        up3 = torch.cat([down3, up3], dim=1)
        up3 = self.convBlock_3_0(up3)
        up3 = self.convBlock_3_1(up3)
        up3 = self.convBlock_3_2(up3)
        print('up3 FINAL', up3.shape)

        up2 = self.upsampler_2(up3)

        down2 = torch.cat([down2_0, down2_1], dim=1)
        print('down2', down2.shape)
        up2 = torch.cat([down2, up2], dim=1)
        up2 = self.convBlock_2_0(up2)
        up2 = self.convBlock_2_1(up2)
        up2 = self.convBlock_2_2(up2)
        print('up2', up2.shape)

        up1 = self.upsampler_1(up2)
        down1 = torch.cat([down1_0, down1_1], dim=1)
        print('down1', down1.shape)
        up1 = torch.cat([down1, up1], dim=1)
        up1 = self.convBlock_1_0(up1)
        up1 = self.convBlock_1_1(up1)
        up1 = self.convBlock_1_2(up1)
        print('up1', up1.shape)

        up0 = self.upsampler_0(up1)
        down0 = torch.cat([down0_0, down0_1], dim=1)
        print('down0', down0.shape)
        up0 = torch.cat([down0, up0], dim=1)
        up0 = self.convBlock_0_0(up0)
        up0 = self.convBlock_0_1(up0)
        up0 = self.convBlock_0_2(up0)
        print('up0', up0.shape)

        final = self.final(up0)
        print('final', final.shape)

        return final

# # Output data dimension check
#
# data = torch.randn((1, 1, 96, 96, 96)).to(device)  # the input has to be 96
# label = torch.randint(0, 2, (1, 1, 96, 96, 96)).to(device)
# net = YNet_noskip()
# #net.eval()
# net = net.to(device)
# net.apply(init)
# #net.eval()
#
# res = net(data)
# for item in res:
#     print(item.size())
#
# # Calculate network parameters
# num_parameter = .0
# for item in net.modules():
#
#     if isinstance(item, nn.Conv3d) or isinstance(item, nn.ConvTranspose3d):
#         num_parameter += (item.weight.size(0) * item.weight.size(1) *
#                           item.weight.size(2) * item.weight.size(3) * item.weight.size(4))
#
#         if item.bias is not None:
#             num_parameter += item.bias.size(0)
#
#     elif isinstance(item, nn.PReLU):
#         num_parameter += item.num_parameters
#
# print(num_parameter)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
# iters = 0
# # training simulation
#
# for epoch in range(8):  # loop over the dataset multiple times
#     inputs = data
#     masks = label
#     for i in range(len(inputs)):
#         running_loss = 0.0
#         # get the inputs; data is a list of [inputs, labels]
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         # print(masks)
#         loss = criterion(outputs, masks[i])
#         # print('mask i', masks[i])
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         iters += 1
#
#         if iters % 2 == 0:
#             print('Prev Loss: {:.4f} '.format(
#                 loss.item()))
#             epoch_loss = running_loss / (len(inputs))
