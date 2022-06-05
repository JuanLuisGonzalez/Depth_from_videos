import torch
import torch.nn as nn
import torch.nn.functional as F
import random

__all__ = [
    'Dispnet_bw_base'
]


def Dispnet_bw_base(data=None, random_blur=False):
    model = Dispnet(batchNorm=False, random_blur=random_blur)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model


def conv_elu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, pad=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ELU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad,
                      bias=True),
            nn.ELU(inplace=True)
        )


class conv_elu_g(nn.Module):
    def __init__(self, batchNorm, in_planes, out_planes, kernel_size=3, stride=1, pad=1):
        super(conv_elu_g, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=True)
        self.conv_mask = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad,
                                   bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.sigmoid(self.conv_mask(x))
        x = self.elu(self.conv1(x))
        return mask * x


class deconv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(deconv, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ref):
        x = F.interpolate(x, size=(ref.size(2), ref.size(3)), mode='nearest')
        x = self.elu(self.conv1(x))
        return x


class residual_block(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(residual_block, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                               bias=False)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                               bias=False)

    def forward(self, x):
        x = self.elu(self.conv2(self.elu(self.conv1(x))) + x)
        return x


class BackBone(nn.Module):
    def __init__(self, batchNorm=True, no_in=3, no_flow=1, no_out_disp=1, no_out_dlog=30, no_ep=8, random_blur=False):
        super(BackBone, self).__init__()
        self.batchNorm = batchNorm
        self.random_blur = random_blur

        # Encoder
        # Encode pixel position from 2 to no_ep
        self.conv_ep1 = conv_elu(self.batchNorm, 2, 16, kernel_size=1, pad=0)
        self.conv_ep2 = conv_elu(self.batchNorm, 16, no_ep, kernel_size=1, pad=0)

        # Two input layers at full and half resolution
        self.conv0 = conv_elu(self.batchNorm, no_in, 64, kernel_size=3)
        self.conv0_1 = residual_block(64)

        # Strided convs of encoder
        self.conv1 = conv_elu(self.batchNorm, 64 + no_ep + no_flow, 128, pad=1, stride=2)
        self.conv1_1 = residual_block(128)
        self.conv2 = conv_elu(self.batchNorm, 128 + no_ep + no_flow, 256, pad=1, stride=2)
        self.conv2_1 = residual_block(256)
        self.conv3 = conv_elu(self.batchNorm, 256 + no_ep + no_flow, 256, pad=1, stride=2)
        self.conv3_1 = residual_block(256)
        self.conv4 = conv_elu(self.batchNorm, 256 + no_ep + no_flow, 256, pad=1, stride=2)
        self.conv4_1 = residual_block(256)
        self.conv5 = conv_elu(self.batchNorm, 256 + no_ep + no_flow, 256, pad=1, stride=2)
        self.conv5_1 = residual_block(256)
        self.conv6 = conv_elu(self.batchNorm, 256 + no_ep + no_flow, 256, pad=1, stride=2)
        self.conv6_1 = residual_block(256)
        self.elu = nn.ELU(inplace=True)

        # Decoder
        # i and up convolutions
        self.deconv6 = deconv(256, 128)
        self.iconv6 = conv_elu(self.batchNorm, 256 + 128, 256)
        self.deconv5 = deconv(256, 128)
        self.iconv5 = conv_elu(self.batchNorm, 256 + 128, 256)
        self.deconv4 = deconv(256, 128)
        self.iconv4 = conv_elu(self.batchNorm, 128 + 256, 256)
        self.deconv3 = deconv(256, 128)
        self.iconv3 = conv_elu(self.batchNorm, 128 + 256, 256)
        self.deconv2 = deconv(256, 128)
        self.iconv2 = conv_elu(self.batchNorm, 128 + 128, 128)
        self.deconv1 = deconv(128, 64)
        self.iconv1 = nn.Conv2d(64 + 64, no_out_disp, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv1dlog = nn.Conv2d(64 + 64, no_out_dlog, kernel_size=3, stride=1, padding=1, bias=False)

        # Initialize conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, grid, flow):
        _, _, H, W = x.shape

        # Encoder section
        # Early feature extraction at target resolution
        out_conv0 = self.conv0_1(self.conv0(x))

        # One strided conv encoder stage
        grid = self.conv_ep2(self.conv_ep1(grid))
        out_conv1 = self.conv1_1(self.conv1(torch.cat((out_conv0, grid, flow), 1)))

        # Deep feature extraction
        dflow = F.interpolate(flow, size=(out_conv1.shape[2], out_conv1.shape[3]), mode='bilinear', align_corners=True)
        dgrid = F.interpolate(grid, size=(out_conv1.shape[2], out_conv1.shape[3]), align_corners=True, mode='bilinear')
        out_conv2 = self.conv2_1(self.conv2(torch.cat((out_conv1, dgrid, dflow), 1)))

        dflow = F.interpolate(flow, size=(out_conv2.shape[2], out_conv2.shape[3]), mode='bilinear', align_corners=True)
        dgrid = F.interpolate(grid, size=(out_conv2.shape[2], out_conv2.shape[3]), align_corners=True, mode='bilinear')
        out_conv3 = self.conv3_1(self.conv3(torch.cat((out_conv2, dgrid, dflow), 1)))

        dflow = F.interpolate(flow, size=(out_conv3.shape[2], out_conv3.shape[3]), mode='bilinear', align_corners=True)
        dgrid = F.interpolate(grid, size=(out_conv3.shape[2], out_conv3.shape[3]), align_corners=True, mode='bilinear')
        out_conv4 = self.conv4_1(self.conv4(torch.cat((out_conv3, dgrid, dflow), 1)))

        dflow = F.interpolate(flow, size=(out_conv4.shape[2], out_conv4.shape[3]), mode='bilinear', align_corners=True)
        dgrid = F.interpolate(grid, size=(out_conv4.shape[2], out_conv4.shape[3]), align_corners=True, mode='bilinear')
        out_conv5 = self.conv5_1(self.conv5(torch.cat((out_conv4, dgrid, dflow), 1)))

        dflow = F.interpolate(flow, size=(out_conv5.shape[2], out_conv5.shape[3]), mode='bilinear', align_corners=True)
        dgrid = F.interpolate(grid, size=(out_conv5.shape[2], out_conv5.shape[3]), align_corners=True, mode='bilinear')
        out_conv6 = self.conv6_1(self.conv6(torch.cat((out_conv5, dgrid, dflow), 1)))

        # Decoder section
        out_deconv6 = self.deconv6(out_conv6, out_conv5)
        concat6 = torch.cat((out_deconv6, out_conv5), 1)
        iconv6 = self.iconv6(concat6)

        out_deconv5 = self.deconv5(iconv6, out_conv4)
        concat5 = torch.cat((out_deconv5, out_conv4), 1)
        iconv5 = self.iconv5(concat5)

        out_deconv4 = self.deconv4(iconv5, out_conv3)
        if self.random_blur and self.training:
            if random.random() > 0.5:
                out_deconv4 = F.avg_pool2d(out_deconv4, kernel_size=3, padding=1, count_include_pad=False, stride=1)
        concat4 = torch.cat((out_deconv4, out_conv3), 1)
        iconv4 = self.iconv4(concat4)

        out_deconv3 = self.deconv3(iconv4, out_conv2)
        if self.random_blur and self.training:
            if random.random() > 0.5:
                out_deconv3 = F.avg_pool2d(out_deconv3, kernel_size=3, padding=1, count_include_pad=False, stride=1)
        concat3 = torch.cat((out_deconv3, out_conv2), 1)
        iconv3 = self.iconv3(concat3)

        out_deconv2 = self.deconv2(iconv3, out_conv1)
        if self.random_blur and self.training:
            if random.random() > 0.5:
                out_deconv2 = F.avg_pool2d(out_deconv2, kernel_size=5, padding=2, count_include_pad=False, stride=1)
        concat2 = torch.cat((out_deconv2, out_conv1), 1)
        iconv2 = self.iconv2(concat2)

        out_deconv1 = self.deconv1(iconv2, out_conv0)
        if self.random_blur and self.training:
            if random.random() > 0.5:
                out_deconv1 = F.avg_pool2d(out_deconv1, kernel_size=5, padding=2, count_include_pad=False, stride=1)
        concat1 = torch.cat((out_deconv1, out_conv0), 1)
        disp = self.iconv1(concat1)
        dlog = self.iconv1dlog(concat1)

        return disp, dlog


class Dispnet(nn.Module):
    def __init__(self, batchNorm, random_blur):
        super(Dispnet, self).__init__()
        self.no_h = 15
        self.no_v = 15
        self.backbone = BackBone(batchNorm, no_in=3, no_flow=2,
                                 no_out_disp=1, no_out_dlog=self.no_h+self.no_v, random_blur=random_blur)
        self.sigmoid = nn.Sigmoid()

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, input_view, in_grid, min_disp, max_disp,
                intrinsics=None, fx=718 / 2000, fy=604 / 2000, focal_factor=1):
        B, C, H, W = input_view.shape

        if intrinsics is not None:
            fx = intrinsics[:, 0, 0].unsqueeze(1).unsqueeze(2) / 2000  # normalized?
            fy = intrinsics[:, 1, 1].unsqueeze(1).unsqueeze(2) / 2000  # normalized?

        fx = focal_factor * fx
        fy = focal_factor * fy

        # convert into 2 channel feature
        flow = torch.ones(B, 2, H, W).type(input_view.type())
        flow[:, 0, :, :] = fx * flow[:, 0, :, :]
        flow[:, 1, :, :] = fy * flow[:, 1, :, :]

        # Get disp
        disp, dlog = self.backbone(input_view, in_grid, torch.abs(flow))
        disp = self.sigmoid(disp) * (max_disp.unsqueeze(1) - min_disp.unsqueeze(1)) + min_disp.unsqueeze(1)

        return disp