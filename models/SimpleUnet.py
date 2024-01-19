import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init(module) :
    if isinstance(module , nn.Conv3d) or isinstance(module , nn.ConvTranspose3d) :
        nn.init.kaiming_normal_(module.weight.data , 0.25)
        nn.init.constant_(module.bias.data , 0)


class UnetConv3(nn.Module) :
    def __init__(self , in_size , out_size , is_batchnorm , kernel_size=(3 , 3 , 1) , padding_size=(1 , 1 , 0) ,
                 init_stride=(1 , 1 , 1)) :
        super(UnetConv3 , self).__init__()

        if is_batchnorm :
            self.conv1 = nn.Sequential(nn.Conv3d(in_size , out_size , kernel_size , init_stride , padding_size) ,
                                       nn.BatchNorm3d(out_size) ,
                                       nn.ReLU(inplace=True) , )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size , out_size , kernel_size , 1 , padding_size) ,
                                       nn.BatchNorm3d(out_size) ,
                                       nn.ReLU(inplace=True) , )
        else :
            self.conv1 = nn.Sequential(nn.Conv3d(in_size , out_size , kernel_size , init_stride , padding_size) ,
                                       nn.ReLU(inplace=True) , )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size , out_size , kernel_size , 1 , padding_size) ,
                                       nn.ReLU(inplace=True) , )

    def forward(self , inputs) :
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3(nn.Module) :
    def __init__(self , in_size , out_size , is_deconv , is_batchnorm=True) :
        super(UnetUp3 , self).__init__()
        if is_deconv :
            self.conv = UnetConv3(in_size , out_size , is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size , out_size , kernel_size=(4 , 4 , 1) , stride=(2 , 2 , 1) ,
                                         padding=(1 , 1 , 0))
        else :
            self.conv = UnetConv3(in_size + out_size , out_size , is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2 , 2 , 1) , mode='trilinear')

    def forward(self , inputs1 , inputs2) :
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2 , offset // 2 , 0]
        outputs1 = F.pad(inputs1 , padding)
        return self.conv(torch.cat([outputs1 , outputs2] , 1))


class unet_3D(nn.Module) :

    def __init__(self , feature_scale=4 , n_classes=3 , is_deconv=True , in_channels=1 , is_batchnorm=True) :
        super(unet_3D , self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [16, 32, 64, 128, 256]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels , filters[0] , self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2 , 2 , 1))

        self.conv2 = UnetConv3(filters[0] , filters[1] , self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2 , 2 , 1))

        self.conv3 = UnetConv3(filters[1] , filters[2] , self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2 , 2 , 1))

        self.conv4 = UnetConv3(filters[2] , filters[3] , self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2 , 2 , 1))

        self.center = UnetConv3(filters[3] , filters[4] , self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp3(filters[4] , filters[3] , self.is_deconv , is_batchnorm)
        self.up_concat3 = UnetUp3(filters[3] , filters[2] , self.is_deconv , is_batchnorm)
        self.up_concat2 = UnetUp3(filters[2] , filters[1] , self.is_deconv , is_batchnorm)
        self.up_concat1 = UnetUp3(filters[1] , filters[0] , self.is_deconv , is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0] , n_classes , 1)

    def forward(self , inputs) :
        conv1 = self.conv1(inputs)
        # print('conv1', conv1.shape)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        print('conv2', conv2.shape)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        print('conv3', conv3.shape)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4 , center)
        up3 = self.up_concat3(conv3 , up4)
        up2 = self.up_concat2(conv2 , up3)
        up1 = self.up_concat1(conv1 , up2)

        final = self.final(up1)
        print('final', final.shape)

        return final

    @staticmethod
    def apply_argmax_softmax(pred) :
        log_p = F.softmax(pred , dim=1)

        return log_p


data = torch.randn((1, 1, 96, 96, 96)).cuda()
label = torch.randint(0, 2, (1, 1, 96, 96, 96)).cuda()
# print('label', label.shape)
net = unet_3D()
net = net.cuda()
net.apply(init)
res = net(data)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
