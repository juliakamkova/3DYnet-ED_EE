import torch
import torch.nn as nn
from torch.nn import functional as F


def convolution_1layers(in_dim , out_dim , activation) :
    """
        Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. """

    return nn.Sequential(
        nn.Conv3d(in_dim , out_dim , kernel_size=3 , stride=1 , padding=1) ,
        nn.BatchNorm3d(out_dim) ,
        activation , )


class AttentionGate(nn.Module) :
    """
    filter the features propagated through the skip connections
    """

    def __init__(self , in_channel , gating_channel , inter_channel) :
        super(AttentionGate , self).__init__()
        self.W_g = nn.Conv3d(gating_channel , inter_channel , kernel_size=1)
        self.W_x = nn.Conv3d(in_channel , inter_channel , kernel_size=2 , stride=2)
        self.relu = nn.ReLU()
        self.psi = nn.Conv3d(inter_channel , 1 , kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self , x , g) :
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        out = self.relu(g_conv + x_conv)
        out = self.sig(self.psi(out))
        out = F.upsample(out , size=x.size()[2 :] , mode='trilinear')
        out = x * out
        return out


class RepeatConv(nn.Module) :
    """
    Repeat Conv + PReLU n times
    """

    def __init__(self , n_channels , n_conv) :
        super(RepeatConv , self).__init__()

        conv_list = []
        for _ in range(n_conv) :
            conv_list.append(nn.Conv3d(n_channels , n_channels , kernel_size=5 , padding=2))
            conv_list.append(nn.PReLU())

        self.conv = nn.Sequential(
            *conv_list
        )

    def forward(self , x) :
        return self.conv(x)


class Down(nn.Module) :
    def __init__(self , in_channels , out_channels , n_conv) :
        super(Down , self).__init__()

        self.downconv = nn.Sequential(
            nn.Conv3d(in_channels , out_channels , kernel_size=2 , stride=2) ,
            nn.PReLU()
        )
        self.conv = RepeatConv(out_channels , n_conv)

    def forward(self , x) :
        out = self.downconv(x)
        return out + self.conv(out)


class Up(nn.Module) :
    def __init__(self , in_channels , out_channels , n_conv) :
        super(Up , self).__init__()

        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels , int(out_channels / 2) , kernel_size=2 , stride=2) ,
            nn.PReLU()
        )
        self.conv = RepeatConv(out_channels , n_conv)

    def forward(self , x , down) :
        x = self.upconv(x)
        cat = torch.cat([x , down] , dim=1)
        return cat + self.conv(cat)


class VNet(nn.Module) :
    """
    Main model
    """

    def __init__(self , in_channels , num_class) :
        super(VNet , self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv3d(1 , 16 , kernel_size=5 , padding=2) ,
            nn.PReLU()
        )

        self.down2 = Down(16 , 32 , 2)
        self.down3 = Down(32 , 64 , 3)
        self.down4 = Down(64 , 128 , 3)
        self.down5 = Down(128 , 256 , 3)

        self.up1 = Up(256 , 256 , 3)
        self.up2 = Up(256 , 128 , 3)
        self.up3 = Up(128 , 64 , 2)
        self.up4 = Up(64 , 32 , 1)

        self.up5 = nn.Sequential(
            nn.Conv3d(32 , num_class , kernel_size=1) ,
            nn.PReLU()
        )

        self.ag1 = AttentionGate(128 , 256 , 128)
        self.ag2 = AttentionGate(64 , 256 , 64)
        self.ag3 = AttentionGate(32, 128 , 32)

    def forward(self , x) :
        down1 = self.down1(x) + torch.cat(16 * [x] , dim=1)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.down5(down4)
        ag1 = self.ag1(down4 , center)
        up1 = self.up1(center , ag1)
        ag2 = self.ag2(down3 , up1)
        up2 = self.up2(up1 , ag2)
        ag3 = self.ag3(down2 , up2)
        up3 = self.up3(up2 , down2)
        up4 = self.up4(up3 , down1)
        return self.up5(up4)


def init(module) :
    if isinstance(module , nn.Conv3d) or isinstance(module , nn.ConvTranspose3d) :
        nn.init.kaiming_normal_(module.weight.data , 0.25)
        nn.init.constant_(module.bias.data , 0)


net = VNet(in_channels=1 , num_class=3)
net.apply(init)

# Output data dimension check
net = net.cuda()
data = torch.randn((1 , 1 , 16 , 160 , 160)).cuda()
res = net(data)
for item in res :
    print(item.size())

# Calculate network parameters
num_parameter = .0
for item in net.modules() :

    if isinstance(item , nn.Conv3d) or isinstance(item , nn.ConvTranspose3d) :
        num_parameter += (item.weight.size(0) * item.weight.size(1) *
                          item.weight.size(2) * item.weight.size(3) * item.weight.size(4))

        if item.bias is not None :
            num_parameter += item.bias.size(0)
