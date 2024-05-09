import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet
from typing import Optional, Callable

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x
class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out

class ResUNet(nn.Module):
    def __init__(self,
                 encoder='resnet50',
                 pretrained=True,
                 coarse_out_ch=128,
                 fine_out_ch=128,
                 c1: int = 16, c2: int = 32, c3: int = 64, c4: int = 128, c5: int = 256, dim: int = 128,
                 agg_mode: str = 'cat',  # sum, cat, fpn
                 single_head: bool = True,
                 pe: bool = False,
                 ):

        super(ResUNet, self).__init__()

        # Config
        self.gate = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.pe = pe

        # ================================== feature encoder
        if self.pe:
            self.position_encoding = PositionEncodingSine()
            self.block1 = ConvBlock(3 + 8, c1, self.gate, nn.BatchNorm2d)
        else:
            self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)

        self.block2 = ResBlock(inplanes=c1, planes=c2, stride=1,
                               downsample=nn.Conv2d(c1, c2, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block3 = ResBlock(inplanes=c2, planes=c3, stride=1,
                               downsample=nn.Conv2d(c2, c3, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block4 = ResBlock(inplanes=c3, planes=c4, stride=1,
                               downsample=nn.Conv2d(c3, c4, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block5 = ResBlock(inplanes=c4, planes=c5, stride=1,
                               downsample=nn.Conv2d(c4, c5, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        block_dims = [32, 64, 128, 256]
        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[2]),
        )
        self.conv_coarse = conv1x1(256, coarse_out_ch)
        #self.conv_fine = conv1x1(196, fine_out_ch, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ResNet Backbone
        if self.pe:
            x1 = self.position_encoding(x)
            x1 = self.block1(x1)  # B x c1 x H x W
        else:
            x = self.block1(x)  # B x c1 x H x W
        x1 = self.pool2(x)
        x1 = self.block2(x1)  # B x c2 x H/2 x W/2
        x2 = self.pool2(x1)
        x2 = self.block3(x2)  # B x c3 x H/4 x W/4
        x3 = self.pool2(x2)
        x3 = self.block4(x3)  # B x dim x H/8 x W/8
        x4 = self.pool2(x3)
        x4 = self.block5(x4)  # B x dim x H/16 x W/16

        x_coarse = self.conv_coarse(x4)
        # FPN
        x4_out = self.layer4_outconv(x4) #1*1

        x4_out_2x = F.interpolate(x4_out, x3.shape[2:], mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)

        x3_out = self.layer3_outconv2(x3_out + x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, x2.shape[2:], mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x_fine = self.layer2_outconv2(x2_out + x3_out_2x)

        return {'global_map':x_coarse, 'local_map':x_fine, 'local_map_small':x2}

class ResUNetHR(nn.Module):
    def __init__(self,
                 encoder='resnet50',
                 pretrained=True,
                 coarse_out_ch=128,
                 fine_out_ch=128
                 ):

        super(ResUNetHR, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        self.firstconv = resnet.conv1  # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # H/4

        # encoder
        self.layer1 = resnet.layer1  # H/4
        self.layer2 = resnet.layer2  # H/8
        self.layer3 = resnet.layer3  # H/16

        # coarse-level conv
        self.conv_coarse = conv(filters[2], coarse_out_ch, 1, 1)

        # decoder
        self.upconv3 = upconv(filters[2], 512, 3, 2)
        self.iconv3 = conv(filters[1] + 512, 512, 3, 1)
        self.upconv2 = upconv(512, 256, 3, 2)
        self.iconv2 = conv(filters[0] + 256, 256, 3, 1)
        self.upconv1 = upconv(256,192,3,2)
        self.iconv1 = conv(64 + 192, 256, 3, 1)

        # fine-level conv
        self.conv_fine = conv(256, fine_out_ch, 1, 1)
        self.out_channels = [fine_out_ch, coarse_out_ch]

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        x_first1 = self.firstrelu(self.firstbn(self.firstconv(x)))
        x_first = self.firstmaxpool(x_first1)

        x1 = self.layer1(x_first)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x_coarse = self.conv_coarse(x3) #H/16

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x = self.upconv1(x)
        x = self.skipconnect(x_first1, x)
        x = self.iconv1(x)

        x_fine = self.conv_fine(x) #H/2

        return {'global_map':x_coarse, 'local_map':x_fine, 'local_map_small':x_first1}

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)
#import torch
#img = torch.rand(1,3,480,640)
#model = ResUNet()
#print('params is {}'.format(sum(p.numel() for p in model.parameters())))
#model(img)
#print(model.block1.conv1.weight.grad.max().item())