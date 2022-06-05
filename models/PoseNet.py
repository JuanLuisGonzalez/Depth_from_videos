import torch
import torch.nn as nn

__all__ = [
    'PoseNet'
]


def PoseNet(data=None, nb_ref_imgs=2):
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


class pose_net(nn.Module):
    def __init__(self, nb_ref_imgs):
        super(pose_net, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        conv_planes = [32, 64, 128, 128, 256, 256, 256]
        self.conv1 = conv_elu(3 * (1 + self.nb_ref_imgs), conv_planes[0])
        self.conv1_1 = residual_block(conv_planes[0])
        self.conv2 = conv_elu(conv_planes[0], conv_planes[1])
        self.conv2_1 = residual_block(conv_planes[1])
        self.conv3 = conv_elu(conv_planes[1], conv_planes[2])
        self.conv3_1 = residual_block(conv_planes[2])
        self.conv4 = conv_elu(conv_planes[2], conv_planes[3])
        self.conv4_1 = residual_block(conv_planes[3])
        self.conv5 = conv_elu(conv_planes[3], conv_planes[4])
        self.conv5_1 = residual_block(conv_planes[4])
        self.conv6 = conv_elu(conv_planes[4], conv_planes[5])
        self.conv6_1 = residual_block(conv_planes[5])
        self.conv7 = conv_elu(conv_planes[5], conv_planes[6])
        self.conv7_1 = residual_block(conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6 * self.nb_ref_imgs, kernel_size=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, target_image, ref_imgs):
        assert (len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)

        out_conv1 = self.conv1_1(self.conv1(input))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        out_conv7 = self.conv7_1(self.conv7(out_conv6))

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = pose.view(pose.size(0), self.nb_ref_imgs, 6)

        return pose
