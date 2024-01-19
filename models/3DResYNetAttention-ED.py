import torch.nn as nn
import torch
from torch.nn import functional as F
from functools import partial
import pytorch_lightning as pl

torch.cuda.empty_cache()
CUDA_LAUNCH_BLOCKING = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def get_inplanes():
    return [8, 16, 32, 64, 128]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=True,
                 shortcut_type='B',
                 widen_factor=1.0,
                 num_class=3):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 1, 1),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)  # Change the kernel size to from 3 to 2
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.layer5 = self._make_layer(block,
                                       block_inplanes[4],
                                       layers[4],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((8, 8, 2))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_class)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        avg0 = self.avgpool(layer4)

        avg = avg0.view(avg0.size(0), -1)

        return avg0, layer1, layer2, layer3, layer4, layer5


def generate_model(model_depth, **kwargs):
    assert model_depth in [18, 34], "Supported model depths are 18 or 34."

    if model_depth == 18:
        model = ResNet(Bottleneck, [2, 2, 2, 2, 1], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)

    return model


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


class AttentionGate(nn.Module):
    """
    filter the features propagated through the skip connections
    """

    def __init__(self, in_channels, gating_channel, inter_channel):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv3d(gating_channel, inter_channel, kernel_size=1)
        self.W_x = nn.Conv3d(in_channels, inter_channel, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.psi = nn.Conv3d(inter_channel, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, g):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        out = self.relu(g_conv + x_conv)
        out = self.sig(self.psi(out))
        out = F.upsample(out, size=x.size()[2:], mode='trilinear')
        out = x * out
        return out


class VGGNet(nn.Module):
    def __init__(self, in_channels=1, VGG_CHANNELS=[8, 16, 32, 64, 64]):
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
        # self.down_2aT = nn.Conv3d(VGG_CHANNELS[2], VGG_CHANNELS[2], kernel_size=(3, 3, 3), padding=1)
        # self.relu_2aT = nn.ReLU()
        self.maxp_2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        # Block 3:
        self.conv_3_0 = nn.Conv3d(VGG_CHANNELS[2], VGG_CHANNELS[3], kernel_size=(3, 3, 3), padding=1)
        self.relu_3_0 = nn.ReLU()
        self.conv_3_1 = nn.Conv3d(VGG_CHANNELS[3], VGG_CHANNELS[3], kernel_size=(3, 3, 3), padding=1)
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv3d(VGG_CHANNELS[3], VGG_CHANNELS[3], kernel_size=(3, 3, 3), padding=1)
        self.relu_3_2 = nn.ReLU()
        # self.down_3aT = nn.Conv3d(VGG_CHANNELS[3], VGG_CHANNELS[3], kernel_size=(3, 3, 3), padding=1)
        # self.relu_3aT = nn.ReLU()
        self.maxp_3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        # Block 4:
        self.conv_4_0 = nn.Conv3d(VGG_CHANNELS[3], VGG_CHANNELS[4], kernel_size=(3, 3, 3), padding=1)
        self.relu_4_0 = nn.ReLU()
        self.conv_4_1 = nn.Conv3d(VGG_CHANNELS[4], VGG_CHANNELS[4], kernel_size=(3, 3, 3), padding=1)
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = nn.Conv3d(VGG_CHANNELS[4], VGG_CHANNELS[4], kernel_size=(3, 3, 3), padding=1)
        self.relu_4_2 = nn.ReLU()
        # self.down_4aT = nn.Conv3d(VGG_CHANNELS[4], VGG_CHANNELS[4], kernel_size=(3, 3, 3), padding=1)
        # self.relu_4aT = nn.ReLU()
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
        down2 = self.relu_2_2(self.conv_2_2(x))
        # down2 = self.relu_2aT(self.down_2aT(x))
        x = self.maxp_2(down2)

        x = self.relu_3_0(self.conv_3_0(x))
        x = self.relu_3_1(self.conv_3_1(x))
        down3 = self.relu_3_2(self.conv_3_2(x))
        # down3 = self.relu_3aT(self.down_3aT(x))
        x = self.maxp_3(down3)

        x = self.relu_4_0(self.conv_4_0(x))
        x = self.relu_4_1(self.conv_4_1(x))
        down4 = self.relu_4_2(self.conv_4_2(x))
        # down4 = self.relu_4aT(self.down_4aT(x))
        x = self.maxp_4(down4)
        # print('x', x.shape)

        return x, down0, down1, down2, down3, down4


class ResYNetED_Attention_3D(pl.LightningModule):

    def __init__(self, n_class=3):
        super().__init__()

        CHANNELS = [8, 16, 32, 64, 128]

        self.vggnet_0 = ResNet(BasicBlock, [2, 2, 2, 2, 1], get_inplanes())
        self.vggnet_1 = ResNet(BasicBlock, [2, 2, 2, 2, 1], get_inplanes())

        self.convBlock_c0 = ConvBlock(CHANNELS[4], CHANNELS[4])
        self.convBlock_c1 = ConvBlock(CHANNELS[4], CHANNELS[4])

        self.upsampler_4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock_4_0 = ConvBlock(4 * CHANNELS[4], CHANNELS[4])
        self.convBlock_4_1 = ConvBlock(CHANNELS[4], CHANNELS[4])
        self.convBlock_4_2 = ConvBlock(CHANNELS[4], CHANNELS[3])

        self.upsampler_3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock_3_0 = ConvBlock(2 * CHANNELS[4], CHANNELS[3])
        self.convBlock_3_1 = ConvBlock(CHANNELS[3], CHANNELS[3])
        self.convBlock_3_2 = ConvBlock(CHANNELS[3], CHANNELS[2])

        self.upsampler_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock_2_0 = ConvBlock(CHANNELS[4], CHANNELS[2])
        self.convBlock_2_1 = ConvBlock(CHANNELS[2], CHANNELS[2])
        self.convBlock_2_2 = ConvBlock(CHANNELS[2], CHANNELS[1])

        self.upsampler_1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock_1_0 = ConvBlock(CHANNELS[3], CHANNELS[1])
        self.convBlock_1_1 = ConvBlock(CHANNELS[1], CHANNELS[1])
        self.convBlock_1_2 = ConvBlock(CHANNELS[1], CHANNELS[0])

        self.upsampler_0 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convBlock_0_0 = ConvBlock(CHANNELS[2] + CHANNELS[0], CHANNELS[1])
        self.convBlock_0_1 = ConvBlock(CHANNELS[1], CHANNELS[1])
        self.convBlock_0_2 = ConvBlock(CHANNELS[1], CHANNELS[0])

        self.ag1 = AttentionGate(CHANNELS[3], CHANNELS[4], CHANNELS[3])
        self.ag2 = AttentionGate(CHANNELS[2], CHANNELS[3], CHANNELS[2])
        self.ag3 = AttentionGate(CHANNELS[1], CHANNELS[2], CHANNELS[1])
        self.ag4 = AttentionGate(CHANNELS[0], CHANNELS[1], CHANNELS[0])

        # final conv (without any concat)
        self.final = nn.Conv3d(CHANNELS[0], n_class, 1)

    def forward(self, x):
        x0, down0_0, down1_0, down2_0, down3_0, down4_0 = self.vggnet_0(x)
        x1, down0_1, down1_1, down2_1, down3_1, down4_1 = self.vggnet_1(x)

        print('x0', x0.shape)
        print('x1', x1.shape)
        center = torch.cat([x0, x1], dim=1)
        center = self.convBlock_c0(center)
        center = self.convBlock_c1(center)
        # print('center', center.shape)

        up4 = self.upsampler_4(center)
        down4_0 = torch.cat([down4_0, up4], dim=1)
        down4_1 = torch.cat([down4_1, up4], dim=1)
        down4 = torch.cat([down4_0, down4_1], dim=1)
        # print('down4', down4.shape)

        up4 = self.convBlock_4_0(down4)
        # up4 = self.convBlock_4_1(up4)
        up4 = self.convBlock_4_2(up4)
        ag1 = self.ag1(up4, center)
        # print('ag1', down4.shape)

        up3 = self.upsampler_3(ag1)
        down3_0 = torch.cat([down3_0, up3], dim=1)
        down3_1 = torch.cat([down3_1, up3], dim=1)
        down3 = torch.cat([down3_0, down3_1], dim=1)
        # print('down3', down3.shape)

        up3 = self.convBlock_3_0(down3)
        # up3 = self.convBlock_3_1(up3)
        up3 = self.convBlock_3_2(up3)
        # print('up3', up3.shape)
        ag2 = self.ag2(up3, up4)

        up2 = self.upsampler_2(ag2)
        down2_0 = torch.cat([down2_0, up2], dim=1)
        down2_1 = torch.cat([down2_1, up2], dim=1)
        down2 = torch.cat([down2_0, down2_1], dim=1)
        # print('down2', down2.shape)

        up2 = self.convBlock_2_0(down2)
        # up2 = self.convBlock_2_1(up2)
        up2 = self.convBlock_2_2(up2)
        # print('up2', up2.shape)
        ag3 = self.ag3(up2, up3)

        up1 = self.upsampler_1(ag3)
        down1_0 = torch.cat([down1_0, up1], dim=1)
        down1_1 = torch.cat([down1_1, up1], dim=1)
        down1 = torch.cat([down1_0, down1_1], dim=1)
        # print('down1', down1.shape)

        up1 = self.convBlock_1_0(down1)
        # up1 = self.convBlock_1_1(up1)
        up1 = self.convBlock_1_2(up1)
        ag4 = self.ag4(up1, up2)

        up0 = self.upsampler_0(ag4)
        down0_0 = torch.cat([down0_0, up0], dim=1)
        down0_1 = torch.cat([down0_0, up0], dim=1)
        down0 = torch.cat([down0_0, down0_1], dim=1)
        # print('down0', down0.shape)

        up0 = self.convBlock_0_0(down0)
        up0 = self.convBlock_0_1(up0)
        up0 = self.convBlock_0_2(up0)

        final = self.final(up0)

        return final

# Output data dimension check

data = torch.randn((1, 1, 256, 256, 64)).to(device)
label = torch.randint(0, 2, (1,1, 256, 256, 64)).to(device)
net = ResYNetED_Attention_3D()
# net.eval()
net = net.to(device)
# net.eval()

res = net(data)
for item in res:
    print(item.size())

# Calculate network parameters
num_parameter = .0
for item in net.modules():

    if isinstance(item, nn.Conv3d) or isinstance(item, nn.ConvTranspose3d):
        num_parameter += (item.weight.size(0) * item.weight.size(1) *
                          item.weight.size(2) * item.weight.size(3) * item.weight.size(4))

        if item.bias is not None:
            num_parameter += item.bias.size(0)

    elif isinstance(item, nn.PReLU):
        num_parameter += item.num_parameters

#print(num_parameter)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
iters = 0
# training simulation

for epoch in range(10):  # loop over the dataset multiple times
    inputs = data
    masks = label
    for i in range(len(inputs)):
        running_loss = 0.0
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # #print(masks)
        loss = criterion(outputs, masks[i])
        # #print('mask i', masks[i])
        loss.backward()
        optimizer.step()

        # #print statistics
        running_loss += loss.item()
        iters += 1

        if iters % 2 == 0:
            print('Prev Loss: {:.4f} '.format(
                loss.item()))
            epoch_loss = running_loss / (len(inputs))
