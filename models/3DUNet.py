import torch
import torch.nn as nn
import pytorch_lightning as pl


class Encoder3D(nn.Module):
    def __init__(self, in_channels):
        super(Encoder3D, self).__init__()

        self.conv = DoubleConvSame3D(c_in=in_channels, c_out=in_channels * 2)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)

        return c,p


class DoubleConvSame3D(nn.Module):
    def __init__(self, c_in, c_out):
        super(DoubleConvSame3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=c_out,
                out_channels=c_out,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(pl.LightningModule):
    def __init__(self, in_channels, num_class):
        super(UNet3D, self).__init__()

        self.conv1 = DoubleConvSame3D(c_in=in_channels, c_out=64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc1 = Encoder3D(64)
        self.enc2 = Encoder3D(128)
        self.enc3 = Encoder3D(256)
        self.enc4 = Encoder3D(512)

        self.conv5 = DoubleConvSame3D(c_in=512, c_out=1024)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose3d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up2 = nn.ConvTranspose3d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up3 = nn.ConvTranspose3d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up4 = nn.ConvTranspose3d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )

        self.up_conv1 = DoubleConvSame3D(c_in=1024, c_out=512)
        self.up_conv2 = DoubleConvSame3D(c_in=512, c_out=256)
        self.up_conv3 = DoubleConvSame3D(c_in=256, c_out=128)
        self.up_conv4 = DoubleConvSame3D(c_in=128, c_out=64)

        self.conv_1x1 = nn.Conv3d(in_channels=64, out_channels=num_class, kernel_size=1)

    def forward(self, x):
        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2, p2 = self.enc1(p1)
        c3, p3 = self.enc2(p2)
        c4, p4 = self.enc3(p3)

        """BOTTLE-NECK"""

        c5 = self.conv5(p4)

        """DECODER"""

        u1 = self.up1(c5)
        cat1 = torch.cat([u1, c4], dim=1)
        uc1 = self.up_conv1(cat1)

        u2 = self.up2(uc1)
        cat2 = torch.cat([u2, c3], dim=1)
        uc2 = self.up_conv2(cat2)

        u3 = self.up3(uc2)
        cat3 = torch.cat([u3, c2], dim=1)
        uc3 = self.up_conv3(cat3)

        u4 = self.up4(uc3)
        cat4 = torch.cat([u4, c1], dim=1)
        uc4 = self.up_conv4(cat4)

        outputs = self.conv_1x1(uc4)

        return outputs
