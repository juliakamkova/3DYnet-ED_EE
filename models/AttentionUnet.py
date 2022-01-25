import torch
import torch.nn as nn
from torch.nn import functional as F


def conv1(in_dim, out_dim, activation) :
    """
        Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. """

    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


class ConvBlock(nn.Module) :
    def __init__(self, in_channels, out_channel, kernel_size=3, stride=1, padding=1) :
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        )

    def forward(self, x) :

        return self.block(x)


class Upsample(nn.Module) :
    """
    upsample, concat and conv
    """

    def __init__(self, in_channels, inter_channel, out_channel) :
        super(Upsample, self).__init__()
        self.up = nn.Sequential(
            ConvBlock(in_channels, inter_channel),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        self.conv = ConvBlock(2 * inter_channel, out_channel)

    def forward(self, x1, x2) :
        x1 = self.up(x1)
        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        return out


class AttentionGate(nn.Module) :
    """
    filter the features propagated through the skip connections
    """

    def __init__(self, in_channels, gating_channel, inter_channel) :
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv3d(gating_channel, inter_channel, kernel_size=1)
        self.W_x = nn.Conv3d(in_channels, inter_channel, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.psi = nn.Conv3d(inter_channel, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, g) :
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        out = self.relu(g_conv + x_conv)
        out = self.sig(self.psi(out))
        out = F.upsample(out, size=x.size()[2 :], mode='trilinear')
        out = x * out
        return out


class AttentionUNet(nn.Module) :
    """
    Main model
    """

    def __init__(self, in_channels, num_class, filters=[64, 128, 256, 512, 1024]) :
        super(AttentionUNet, self).__init__()

        f1, f2, f3, f4, f5 = filters

        self.down1 = ConvBlock(in_channels, f1)
        self.down2 = ConvBlock(f1, f2)
        self.down3 = ConvBlock(f2, f3)
        self.down4 = ConvBlock(f3, f4)
        self.center = ConvBlock(f4, f5)

        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.ag1 = AttentionGate(f4, f5, f4)
        self.ag2 = AttentionGate(f3, f4, f3)
        self.ag3 = AttentionGate(f2, f3, f2)
        self.ag4 = AttentionGate(f1, f2, f1)

        self.up1 = Upsample(f5, f4, f4)
        self.up2 = Upsample(f4, f3, f3)
        self.up3 = Upsample(f3, f2, f2)
        self.up4 = Upsample(f2, f1, f1)

        self.Conv = nn.Conv3d(64, num_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x) :
        down1 = self.down1(x)
        print('down1', down1.shape)

        down2 = self.MaxPool(down1)
        down2 = self.down2(down2)
        print('down2', down2.shape)

        down3 = self.MaxPool(down2)
        down3 = self.down3(down3)
        print('down3', down3.shape)

        down4 = self.MaxPool(down3)
        down4 = self.down4(down4)
        print('down4', down4.shape)

        center = self.MaxPool(down4)
        center = self.center(center)
        print('center', center.shape)

        ag1 = self.ag1(down4, center)
        print('ag1', ag1.shape)
        up1 = self.up1(center, ag1)
        print('up1', ag1.shape)
        ag2 = self.ag2(down3, up1)
        print('ag2', ag2.shape)
        up2 = self.up2(up1, ag2)
        print('up2', up2.shape)
        ag3 = self.ag3(down2, up2)
        print('ag3', ag3.shape)
        up3 = self.up3(up2, ag3)
        print('up3', up3.shape)
        ag4 = self.ag4(down1, up3)
        print('ag4', ag4.shape)
        up4 = self.up4(up3, down1)
        print('up4', up4.shape)

        out = self.Conv(up4)
        print(out.shape)

        return out
#
#
# def init(module) :
#     if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d) :
#         nn.init.kaiming_normal_(module.weight.data, 0.25)
#         nn.init.constant_(module.bias.data, 0)
#
#
# net = AttentionUNet(in_channels=1, num_class=3)
# net.apply(init)
#
# # Output data dimension check
# net = net.cuda()
# data = torch.randn((1, 1, 48, 48, 48)).cuda()  # the input has to be 96
# label = torch.randint(0, 2, (1, 1, 48, 48, 48)).cuda()
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
#         if iters % 2 == 0 :
#             print('Prev Loss: {:.4f} Prev Acc: {:.4f}'.format(
#                 loss.item(), torch.sum(outputs == masks) / inputs.size(0)))
#             epoch_loss = running_loss / (len(inputs))
#         # if i % 2000 == 1999:  # print every 2000 mini-batches
#         #     print('[%d, %5d] loss: %.3f' %
#         #           (epoch + 1, i + 1, running_loss / 2000))
#         #     running_loss = 0.0
