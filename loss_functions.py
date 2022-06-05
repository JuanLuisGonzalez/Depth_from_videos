import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from myUtils import coords_to_normals, disp_to_coords_n
from inverse_warp import inverse_warp


class Vgg19_pc(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_pc, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        vgg_pretrained_features = nn.DataParallel(vgg_pretrained_features.cuda())

        # this has Vgg config E:
        # partial convolution paper uses up to pool3
        # [64,'r', 64,r, 'M', 128,'r', 128,r, 'M', 256,'r', 256,r, 256,r, 256,r, 'M', 512,'r', 512,r, 512,r, 512,r]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        n_new = 0
        for x in range(5):  # pool1,
            self.slice1.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(5, 10):  # pool2
            self.slice2.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(10, 19):  # pool3
            self.slice3.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(19, 28):  # pool4
            self.slice4.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, full=False):
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_4 = self.slice3(h_relu2_2)
        if full:
            h_relu4_4 = self.slice4(h_relu3_4)
            return h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4
        else:
            return h_relu1_2, h_relu2_2, h_relu3_4


vgg = Vgg19_pc()


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=32):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 256 x 512
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 256
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 128
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 64
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 32
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 16
            nn.Conv2d(ndf * 16, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 8
            nn.Conv2d(ndf * 16, 1, (4, 8), 1, 0, bias=False),
            nn.Sigmoid()
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        return self.main(input)


def pose_loss(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, ign_mask, pose,
                rotation_mode='euler', padding_mode='zeros', epoch=0):
    loss = 0
    warped_imgs = []
    for i in range(len(ref_imgs)):
        current_pose = pose[:, i, :]
        ref_img_warped = inverse_warp(ref_imgs[i], depth[:, 0, :, :], current_pose, intrinsics,
                                      intrinsics_inv, rotation_mode, padding_mode)
        warped_imgs.append(ref_img_warped)
        out_of_bound = 1 - (ref_img_warped.detach() == 0).prod(dim=1, keepdim=True).type_as(ref_img_warped)
        warp_error = (ign_mask * out_of_bound * torch.abs((tgt_img - ref_img_warped))).mean(3).mean(2).mean(1)

        # Handle static frames
        if epoch > 0:
            stat_error = (ign_mask * torch.abs(tgt_img - ref_imgs[i])).mean(3).mean(2).mean(1)
            stat_mask = (warp_error < stat_error).type_as(warp_error).detach()
            loss = loss + (stat_mask * warp_error).sum() / (stat_mask.sum() + 0.0001)
        else:
            loss = loss + warp_error.mean()

    return loss, warped_imgs


def pose_loss_masked(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, masks, pose,
                rotation_mode='euler', padding_mode='zeros', alpha=0):
    eps = 1e-5
    loss = 0
    warped_imgs = []
    warped_masks = []
    for i in range(len(ref_imgs)):
        current_pose = pose[:, i, :]
        ref_img_warped = inverse_warp(ref_imgs[i], depth, current_pose, intrinsics,
                                      intrinsics_inv, rotation_mode, padding_mode)
        out_of_bound = 1 - (ref_img_warped.detach() == 0).prod(dim=1, keepdim=True).type_as(ref_img_warped)
        warp_error = torch.sum(masks[i] * out_of_bound * torch.abs(tgt_img - ref_img_warped), dim=(3, 2, 1)) / (
                tgt_img.shape[1] * torch.sum(masks[i] * out_of_bound, dim=(3, 2, 1)) + eps)

        if alpha == 0:
            dn_mask = F.interpolate(masks[i] * out_of_bound, scale_factor=0.5)
            ddn_mask = F.interpolate(masks[i] * out_of_bound, scale_factor=0.25)
            dn_warped = F.interpolate(ref_img_warped, scale_factor=0.5, mode='bilinear', align_corners=True)
            ddn_warped = F.interpolate(ref_img_warped, scale_factor=0.25, mode='bilinear', align_corners=True)
            dn_tgt = F.interpolate(tgt_img, scale_factor=0.5, mode='bilinear', align_corners=True)
            ddn_tgt = F.interpolate(tgt_img, scale_factor=0.25, mode='bilinear', align_corners=True)
            dn_warp_error = torch.sum(dn_mask * torch.abs(dn_warped - dn_tgt), dim=(3, 2, 1)) / (
                    tgt_img.shape[1] * torch.sum(dn_mask, dim=(3, 2, 1)) + eps)
            ddn_warp_error = torch.sum(ddn_mask * torch.abs(ddn_warped - ddn_tgt), dim=(3, 2, 1)) / (
                    tgt_img.shape[1] * torch.sum(ddn_mask, dim=(3, 2, 1)) + eps)

            loss = loss + (0.5 * warp_error + 0.3 * dn_warp_error + 0.2 * ddn_warp_error).mean()# / len(ref_imgs)
        else:
            loss = loss + warp_error.mean() / len(ref_imgs)

        warped_imgs.append(ref_img_warped)
        warped_masks.append(masks[i] * out_of_bound)

    if alpha > 0:
        all_warped = torch.cat((warped_imgs[0], warped_imgs[1]), 0)
        all_mask = torch.cat((warped_masks[0], warped_masks[1]), 0)
        loss = loss + alpha * perceptual_loss_masked(
            vgg((1 - all_mask) * torch.cat((tgt_img, tgt_img), 0) + all_mask * all_warped),
            vgg(torch.cat((tgt_img, tgt_img), 0)),
            all_mask).mean()

    return loss, warped_imgs


def pose_loss_automask(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, occ_masks, pose,
                rotation_mode='euler', padding_mode='zeros', epoch=0):
    warped_imgs = []
    stat_error = []
    proj_error = []
    for i in range(len(ref_imgs)):
        stat_error.append(torch.mean(torch.abs(tgt_img - ref_imgs[i]), dim=1, keepdim=True))

        current_pose = pose[:, i, :]
        ref_img_warped = inverse_warp(ref_imgs[i], depth[:, 0, :, :], current_pose, intrinsics,
                                      intrinsics_inv, rotation_mode, padding_mode)
        proj_error.append(torch.mean(torch.abs((tgt_img - ref_img_warped)), dim=1, keepdim=True))
        warped_imgs.append(ref_img_warped)

        if i == 0:
            min_error = proj_error[i]
        else:
            min_error = torch.cat((min_error, proj_error[i]), 1)

    # Mask errors that are not min
    with torch.no_grad():
        _, idx = torch.min(min_error, dim=1, keepdim=True)
        min_mask = torch.zeros(min_error.shape).type_as(min_error)
        for i in range(len(ref_imgs)):
            min_mask[:, i, :, :] = (idx[:, 0, :, :] == i).type_as(min_error)
    l_p = min_error * min_mask.detach()

    loss = 0
    for i in range(len(ref_imgs)):
        u = (proj_error[i] < stat_error[i]).type_as(proj_error[i]).detach()
        loss = loss + torch.mean(occ_masks[i] * u * l_p[:, i, :, :].unsqueeze(1))

    return loss, warped_imgs


class Normal_smoothness(nn.Module):
    def __init__(self):
        super(Normal_smoothness, self).__init__()

    def forward(self, img, disp, grid, h, w, gamma=1):
        B, C, H, W = img.shape

        # Convert img to grayscale for direct edge detection
        m_rgb = torch.ones((B, C, 1, 1)).cuda()
        m_rgb[:, 0, :, :] = 0.411 * m_rgb[:, 0, :, :]
        m_rgb[:, 1, :, :] = 0.432 * m_rgb[:, 1, :, :]
        m_rgb[:, 2, :, :] = 0.45 * m_rgb[:, 2, :, :]
        gray_img = getGrayscale(img + m_rgb)

        # Sobel filters for image edges
        sx_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])).unsqueeze(
            0).unsqueeze(0).cuda()
        sy_filter = torch.autograd.Variable(torch.Tensor([[0, -1, 0], [0, 2, 0], [0, -1, 0]])).unsqueeze(
            0).unsqueeze(0).cuda()
        dx_img = torch.abs(F.conv2d(gray_img, sx_filter, padding=1, stride=1))
        dy_img = torch.abs(F.conv2d(gray_img, sy_filter, padding=1, stride=1))

        # Get normals from disp
        normals = disp_to_coords_n(disp, grid, h, w)
        normals = coords_to_normals(normals)

        # Measure normals difference (with l2 norm squared)
        dx_n = torch.sum((normals[:, :, :, 1::] - normals[:, :, :, 0:-1]) ** 2, dim=1, keepdim=True)
        pad = (0, 1, 0, 0)
        dx_n = F.pad(dx_n, pad)
        dy_n = torch.sum((normals[:, :, 1::, :] - normals[:, :, 0:-1, :]) ** 2, dim=1, keepdim=True)
        pad = (0, 0, 0, 1)
        dy_n = F.pad(dy_n, pad)
        Cds = torch.mean(dx_n * torch.exp(-gamma * dx_img) + dy_n * torch.exp(-gamma * dy_img))
        return Cds, normals


def smoothness(img, disp):
    B, C, H, W = img.shape

    m_rgb = torch.ones((B, C, 1, 1)).cuda()
    m_rgb[:, 0, :, :] = 0.411 * m_rgb[:, 0, :, :]
    m_rgb[:, 1, :, :] = 0.432 * m_rgb[:, 1, :, :]
    m_rgb[:, 2, :, :] = 0.45 * m_rgb[:, 2, :, :]
    gray_img = getGrayscale(img + m_rgb)

    # Disparity smoothness
    dx_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dy_filter = torch.autograd.Variable(torch.Tensor([[0, -1, 0], [0, 2, 0], [0, -1, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dx_img = F.conv2d(gray_img, dx_filter, padding=1, stride=1)
    dy_img = F.conv2d(gray_img, dy_filter, padding=1, stride=1)
    dx_d = F.conv2d(disp, dx_filter, padding=1, stride=1)
    dy_d = F.conv2d(disp, dy_filter, padding=1, stride=1)
    Cds = torch.mean(
        torch.abs(dx_d) * torch.exp(-torch.abs(dx_img)) + torch.abs(dy_d) * torch.exp(-torch.abs(dy_img)))

    return Cds


def getGrayscale(input):
    # Input is mini-batch N x 3 x H x W of an RGB image (analog rgb from 0...1)
    output = torch.autograd.Variable(input.data.new(*input.size()))
    # Output is mini-batch N x 3 x H x W from y = 0 ... 1
    output[:, 0, :, :] = 0.299 * input[:, 0, :, :] + 0.587 * input[:, 1, :, :] + 0.114 * input[:, 2, :, :]
    return output[:, 0, :, :].unsqueeze(1)


class MultiscaleEPE(nn.Module):
    def __init__(self, multiscale_weights):
        super(MultiscaleEPE, self).__init__()
        self.w_m = multiscale_weights

    def forward(self, output_diparity, label_disparity):
        return multiscaleEPE(output_diparity, label_disparity, self.w_m, False)


def EPE(net_out, target, sparse=False, disp=True, mean=True):
    EPE_map = torch.norm(target - net_out, p=2, dim=1)  # l2 norm per image
    batch_size = EPE_map.size(0)
    if sparse:
        if disp:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = target[:, 0] == 0
        else:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = (target[:, 0] == 0) & (target[:, 1] == 0)
        EPE_map = EPE_map[~mask.data]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size


def sparse_max_pool(input, size):
    positive = (input > 0).float()
    negative = (input < 0).float()
    output = nn.functional.adaptive_max_pool2d(input * positive, size) - nn.functional.adaptive_max_pool2d(
        -input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):
        b, _, h, w = output.size()
        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = nn.functional.adaptive_avg_pool2d(target, (h, w))
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]  # if output is not a touple, make it a touple of one element
    if weights is None:
        weights = [0.001, 0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert (len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)  # linear combination of net outputs and weights
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
    return EPE(upsampled_output, target, sparse, mean=True)


def smoothness(img, disp, gamma=1, mask=1):
    B, C, H, W = img.shape

    m_rgb = torch.ones((B, C, 1, 1)).cuda()
    m_rgb[:, 0, :, :] = 0.411 * m_rgb[:, 0, :, :]
    m_rgb[:, 1, :, :] = 0.432 * m_rgb[:, 1, :, :]
    m_rgb[:, 2, :, :] = 0.45 * m_rgb[:, 2, :, :]
    gray_img = getGrayscale(img + m_rgb)

    # Disparity smoothness
    sx_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    sy_filter = torch.autograd.Variable(torch.Tensor([[0, -1, 0], [0, 2, 0], [0, -1, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dx_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dy_filter = torch.autograd.Variable(torch.Tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dx1_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dy1_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dx_img = F.conv2d(gray_img, sx_filter, padding=1, stride=1)
    dy_img = F.conv2d(gray_img, sy_filter, padding=1, stride=1)
    dx_d = F.conv2d(disp, dx_filter, padding=1, stride=1)
    dy_d = F.conv2d(disp, dy_filter, padding=1, stride=1)
    dx1_d = F.conv2d(disp, dx1_filter, padding=1, stride=1)
    dy1_d = F.conv2d(disp, dy1_filter, padding=1, stride=1)
    Cds = torch.mean(
        mask * (torch.abs(dx_d) + torch.abs(dx1_d)) * torch.exp(-gamma * torch.abs(dx_img)) +
        mask * (torch.abs(dy_d) + torch.abs(dy1_d)) * torch.exp(-gamma * torch.abs(dy_img)))
    return Cds


def perceptual_loss(out_vgg, label_vgg, layer=None):
    if layer is not None:
        l_p = torch.mean((out_vgg[layer] - label_vgg[layer]) ** 2)
    else:
        l_p = 0
        for i in range(3):
            l_p += torch.mean((out_vgg[i] - label_vgg[i]) ** 2)

    return l_p


def perceptual_loss_masked(out_vgg, label_vgg, mask, layer=None):
    eps = 1e-5
    if layer is not None:
        _, c, h, w = out_vgg[layer].shape
        rsz_mask = F.interpolate(mask, size=(h, w))
        l_p = (rsz_mask * (out_vgg[layer] - label_vgg[layer]) ** 2).sum((3, 2, 1)) \
              / (c * rsz_mask.sum((3, 2, 1)) + eps)
    else:
        l_p = 0
        for i in range(3):
            _, c, h, w = out_vgg[i].shape
            rsz_mask = F.interpolate(mask, size=(h, w))
            l_p += (rsz_mask * (out_vgg[i] - label_vgg[i]) ** 2).sum((3, 2, 1)) \
                   / (c*rsz_mask.sum((3, 2, 1)) + eps)
    return l_p
