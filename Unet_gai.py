
from PIL import Image
from OpticalFlow_Visualization import flow_vis
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

args = utils.parse_command()
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, True),
            #nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                  nn.Tanh())

    def forward(self, x):
        return self.conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),

        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Unet_En0(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(Unet_En0, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.down6 = Down(1024, 1024)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 2))
        self.fc = nn.Linear(1024, 1024)
        self.sigmoid = nn.Sigmoid()
        self.tan = nn.Tanh()
    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.avgpool(x5)
        x5 = x5.view(x5.size(0), -1)
        x5 = self.fc(x5)
        x5 = self.tan(x5)
        flow = x5.reshape(x5.size(0), 2, 16, 32)

        return  flow

class Unet_De20(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(Unet_De20, self).__init__()
        self.n_classes = n_classes
        #测试网络层数256首次放大层数区别
        self.upGrid = nn.Sequential( DoubleConv2(2, 256),
                                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
                                     DoubleConv2(128, 128),
                                    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
                                     DoubleConv2(64, 64),
                                    nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
                                     DoubleConv2(32, 32),
                                    nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(16), nn.LeakyReLU(0.2, True),
                                     DoubleConv2(16, 16),
                                    OutConv(16, 2)
                                    )
        self.outc = DoubleConv2(3, 3, 32)
    def forward(self, input, flow):
        # gridUp = flow.permute(0, 3, 1, 2)
        gridUp = self.upGrid(flow)
        # flow_color = flow_vis.flow_to_color(gridUp[0].data.squeeze().cpu().permute(1, 2, 0).numpy(), convert_to_bgr=False)
        # filename = r'E:\liujingguo\rotate_img\Fig1_flow.png'
        # flow_color = Image.fromarray(flow_color.astype('uint8'))
        # flow_color.save(filename)
        gridUp = gridUp.permute(0, 2, 3, 1)
        input = self.deform_input(input, gridUp)
        return input

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return torch.nn.functional.grid_sample(inp, deformation, align_corners=True)