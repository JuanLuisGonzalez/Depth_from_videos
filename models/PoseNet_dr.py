import torch
import torch.nn as nn

__all__ = [
    'PoseNet_dr'
]


def PoseNet_dr(data=None, nb_ref_imgs=2):
    model = pose_net(nb_ref_imgs)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model


def conv_elu(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=2),
        nn.ELU(inplace=True),
    )


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


class discrete_regressor(nn.Module):
    def __init__(self, in_planes, latent_planes, reg_planes):
        super(discrete_regressor, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, latent_planes, kernel_size=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()

        self.conv2 = nn.Conv2d(latent_planes, reg_planes, kernel_size=1, padding=0, bias=False)
        nn.init.zeros_(self.conv2.weight.data)

        self.softmax = nn.Softmax(dim=1)
        self.reg_planes = reg_planes

    def forward(self, feats, min, max):
        x = self.elu(self.conv1(feats)).mean(3).mean(2).unsqueeze(2).unsqueeze(3)
        x = self.softmax(self.conv2(x))

        i = torch.tensor(list(range(0, self.reg_planes))).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        c = i.type_as(feats) / (self.reg_planes - 1)
        w = c * (max - min) + min
        reg_value = torch.sum(w * x, dim=1)
        return reg_value


class pose_net(nn.Module):
    def __init__(self, nb_ref_imgs):
        super(pose_net, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        conv_planes = [32, 64, 128, 128, 256, 256, 256]

        # Common encoder (feature extractor)
        self.conv1 = conv_elu(3, conv_planes[0])
        self.conv1_1 = residual_block(conv_planes[0])
        self.conv2 = conv_elu(conv_planes[0], conv_planes[1])
        self.conv2_1 = residual_block(conv_planes[1])
        self.conv3 = conv_elu(conv_planes[1], conv_planes[2])
        self.conv3_1 = residual_block(conv_planes[2])
        self.conv4 = conv_elu(conv_planes[2], conv_planes[3])
        self.conv4_1 = residual_block(conv_planes[3])

        # Pose estimator
        self.conv5 = conv_elu(conv_planes[3] * 2, conv_planes[4])
        self.conv5_1 = residual_block(conv_planes[4])
        self.conv6 = conv_elu(conv_planes[4], conv_planes[5])
        self.conv6_1 = residual_block(conv_planes[5])

        # Initialize conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.x0_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)
        self.y0_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)
        self.z0_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)
        self.rx0_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)
        self.ry0_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)
        self.rz0_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)

        self.x1_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)
        self.y1_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)
        self.z1_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)
        self.rx1_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)
        self.ry1_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)
        self.rz1_reg = discrete_regressor(conv_planes[5] * 2, conv_planes[5] // 2, reg_planes=32)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def encoder(self, input):
        out_conv1 = self.conv1_1(self.conv1(input))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        return out_conv4

    def forward(self, target_image, ref_imgs):
        assert (len(ref_imgs) == self.nb_ref_imgs)

        # get target features
        tgt_feats = self.encoder(target_image)

        # Get ref imgs feats
        ref_feats = []
        for i in range(len(ref_imgs)):
            ref_feats.append(self.encoder(ref_imgs[i]))

        # Get pose feats for each pair
        out_conv5 = self.conv5_1(self.conv5(torch.cat((tgt_feats, ref_feats[0]), 1)))
        out_conv6_0 = self.conv6_1(self.conv6(out_conv5))

        out_conv5 = self.conv5_1(self.conv5(torch.cat((tgt_feats, ref_feats[1]), 1)))
        out_conv6_1 = self.conv6_1(self.conv6(out_conv5))

        out_conv5 = self.conv5_1(self.conv5(torch.cat((ref_feats[0], ref_feats[1]), 1)))
        out_conv6_01 = self.conv6_1(self.conv6(out_conv5))

        x_0 = self.x0_reg(torch.cat((out_conv6_0, out_conv6_01), 1), min=-2, max=2)
        y_0 = self.y0_reg(torch.cat((out_conv6_0, out_conv6_01), 1), min=-1, max=1)
        z_0 = self.z0_reg(torch.cat((out_conv6_0, out_conv6_01), 1), min=-10, max=10)
        rx_0 = self.rx0_reg(torch.cat((out_conv6_0, out_conv6_01), 1), min=-3.1416 / 6, max=3.1416 / 6)
        ry_0 = self.ry0_reg(torch.cat((out_conv6_0, out_conv6_01), 1), min=-3.1416 / 4, max=3.1416 / 4)
        rz_0 = self.rz0_reg(torch.cat((out_conv6_0, out_conv6_01), 1), min=-3.1416 / 6, max=3.1416 / 6)
        pose_0 = torch.cat((x_0, y_0, z_0, rx_0, ry_0, rz_0), 2)

        x_1 = self.x1_reg(torch.cat((out_conv6_1, out_conv6_01), 1), min=-2, max=2)
        y_1 = self.y1_reg(torch.cat((out_conv6_1, out_conv6_01), 1), min=-1, max=1)
        z_1 = self.z1_reg(torch.cat((out_conv6_1, out_conv6_01), 1), min=-10, max=10)
        rx_1 = self.rx1_reg(torch.cat((out_conv6_1, out_conv6_01), 1), min=-3.1416 / 6, max=3.1416 / 6)
        ry_1 = self.ry1_reg(torch.cat((out_conv6_1, out_conv6_01), 1), min=-3.1416 / 4, max=3.1416 / 4)
        rz_1 = self.rz1_reg(torch.cat((out_conv6_1, out_conv6_01), 1), min=-3.1416 / 6, max=3.1416 / 6)
        pose_1 = torch.cat((x_1, y_1, z_1, rx_1, ry_1, rz_1), 2)

        pose = torch.cat((pose_0, pose_1), 1)

        return pose
