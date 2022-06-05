import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from inverse_warp import m_inverse_warp_grid

__all__ = [
    'Dispnet_fw_base'
]


def Dispnet_fw_base(data=None, no_levels=33, random_blur=False):
    model = Dispnet(batchNorm=False, no_levels=no_levels, random_blur=random_blur)
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
    def __init__(self, batchNorm=True, no_in=3, no_flow=1, no_out_disp=1, no_ep=8, random_blur=False):
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

        return disp


class Dispnet(nn.Module):
    def __init__(self, batchNorm, no_levels, random_blur):
        super(Dispnet, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.no_levels = no_levels
        self.no_h = 15
        self.no_v = 15
        self.backbone = BackBone(batchNorm, no_in=3, no_flow=2,
                                 no_out_disp=self.no_levels, no_out_dlog=self.no_h+self.no_v, random_blur=random_blur)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, input_view, in_grid, min_disp, max_disp,
                pose=None, the_focal=None, baseline=None, intrinsics=None, intrinsics_inv=None,
                ret_disp=True, ret_subocc=False, ret_pan=False, exp_mask=None,
                ret_smdlog=False, ret_rdisp=False, ref=None, gamma=1, double_sample=False,
                fx=718 / 2000, fy=604 / 2000, focal_factor=1, ret_dp=False, mask_out=False):
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

        # Get disp and kernels
        if isinstance(in_grid, list):
            m_in_grid = in_grid[1]
        else:
            m_in_grid = in_grid
        dlog = self.backbone(input_view, m_in_grid, torch.abs(flow))
        sm_dlog = self.softmax(dlog * gamma)

        lev = self.no_levels
        with torch.no_grad():
            c_dst = torch.linspace(0, 1, steps=lev).view(1, -1, 1, 1).cuda()

        # Get disp and conf mask from kernel
        w = max_disp.view(B, 1, 1, 1) * torch.exp(torch.log(max_disp / min_disp).view(B, 1, 1, 1) * (c_dst - 1))
        disp = torch.sum(w * sm_dlog, 1).unsqueeze(1)

        if ret_disp and not ret_subocc and not ret_pan:
            output = []
            if ret_disp:
                output.append(disp)
            if ret_smdlog:
                output.append(sm_dlog)
            return output

        if double_sample:
            input_view = torch.cat((input_view, input_view), 0)
            dlog = torch.cat((dlog, dlog), 0)
            sm_dlog = torch.cat((sm_dlog, sm_dlog), 0)
            ref = torch.cat((ref, ref), 0)
            max_disp = torch.cat((max_disp, max_disp), 0)
            min_disp = torch.cat((min_disp, min_disp), 0)
            intrinsics = torch.cat((intrinsics, intrinsics), 0)
            intrinsics_inv = torch.cat((intrinsics_inv, intrinsics_inv), 0)
            B = B * 2

        if mask_out and exp_mask is not None:
            ignore_region = (in_grid[:, 1, :, :].unsqueeze(1) < -0.25).type_as(in_grid)
            ignore_region = torch.cat((ignore_region, ignore_region), 0)
            exp_mask_ = exp_mask + ignore_region
            exp_mask_[exp_mask_ > 1] = 1
            dlog = exp_mask_.detach() * dlog

        with torch.no_grad():
            c_dst = torch.linspace(0, 1, steps=self.no_levels).view(1, -1, 1, 1).cuda()

        # Get disp and conf mask from kernel
        w = max_disp.view(B, 1, 1, 1) * torch.exp(torch.log(max_disp / min_disp).view(B, 1, 1, 1) * (c_dst - 1))
        w = w * torch.ones(B, 1, H, W).type_as(w)
        w_b = w.view(-1, H, W)
        # Get rotation and translation grids
        i_tetha = torch.zeros(B, 2, 3).cuda()
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        i_grid = F.affine_grid(i_tetha, [B, C, H, W], align_corners=True)

        ones = torch.ones(B, H, W).type(input_view.type())
        rot_pose = pose.clone()
        rot_pose[:, 0:3] = 0
        trans_pose = pose.clone()
        trans_pose[:, 3:6] = 0

        one_pix_flow_r = m_inverse_warp_grid(baseline *
                                             the_focal / ones, rot_pose, intrinsics,
                                             intrinsics_inv, rotation_mode='euler', padding_mode='border') - i_grid
        one_pix_flow_r = torch.repeat_interleave(one_pix_flow_r, repeats=lev, dim=0)

        one_pix_flow_t = m_inverse_warp_grid(torch.repeat_interleave(baseline * the_focal, repeats=lev, dim=0) / w_b,
                                             torch.repeat_interleave(trans_pose, repeats=lev, dim=0),
                                             torch.repeat_interleave(intrinsics, repeats=lev, dim=0),
                                             torch.repeat_interleave(intrinsics_inv, repeats=lev, dim=0),
                                             rotation_mode='euler', padding_mode='border') - torch.repeat_interleave(i_grid, repeats=lev, dim=0)

        out_grid = torch.repeat_interleave(i_grid, repeats=lev, dim=0).clone()
        out_grid = out_grid + one_pix_flow_t + one_pix_flow_r

        dlog = dlog.view(-1, 1, H, W)
        Dprob = F.grid_sample(dlog, out_grid, align_corners=True).view(B, -1, H, W)
        Dprob = self.softmax(Dprob)

        # Blend shifted features
        p_im0 = 0
        maskR = 0
        maskL = 0
        dispr = 0
        syn_mask = 0

        # # Get synth right view
        if ret_pan:
            if ref is not None:
                o_im0 = torch.repeat_interleave(ref, repeats=lev, dim=0)
            else:
                o_im0 = torch.repeat_interleave(input_view, repeats=lev, dim=0)
            p_im0 = F.grid_sample(o_im0, out_grid, align_corners=True).view(B, lev, C, H, W) * Dprob.unsqueeze(2)
            p_im0 = p_im0.sum(1)

        if ret_subocc:
            with torch.no_grad():
                # Get content visible in right that is also visible in left
                maskR = F.grid_sample(sm_dlog.view(-1, 1, H, W).detach(), out_grid.detach(), align_corners=True)
                maskR = maskR.view(B, lev, H, W).sum(1).unsqueeze(1)
                # Get content visible in left that is also visible in right
                out_grid1 = torch.repeat_interleave(i_grid, repeats=lev, dim=0).clone()
                out_grid1 = out_grid1 - one_pix_flow_t - one_pix_flow_r
                maskL = F.grid_sample(Dprob.view(-1, 1, H, W).detach(), out_grid1.detach(), align_corners=True)
                maskL = maskL.view(B, lev, H, W).sum(1).unsqueeze(1)

        if ret_rdisp:
            dispr = torch.sum(w * Dprob, 1).unsqueeze(1)

        if exp_mask is not None:
            exp_mask = torch.repeat_interleave(exp_mask, repeats=lev, dim=0)
            syn_mask = F.grid_sample(exp_mask, out_grid.detach(), align_corners=True).view(B, lev, 1, H, W) * Dprob.unsqueeze(2)
            syn_mask = syn_mask.sum(1)

        # Return selected outputs (according to input arguments)
        output = []
        if ret_pan:
            output.append(p_im0)
        if ret_disp:
            output.append(disp)
        if ret_rdisp:
            output.append(dispr)
        if ret_smdlog:
            output.append(Dprob)
        if ret_subocc:
            maskL[maskL > 1] = 1
            output.append(maskL)
            maskR[maskR > 1] = 1
            output.append(maskR)
        if exp_mask is not None:
            output.append(syn_mask)
        if ret_dp:
            output.append(sm_dlog[0:B // 2])

        return output
