import torch.utils.data as data
import torch

import os
import os.path
from imageio import imread
import numpy as np
import random
import json

# Indexes for camera frames
cam1_index_00 = 0
cam1_index_0 = 1
cam1_index = 2
cam1_index_1 = 3
cam1_index_11 = 4


def img_loader(input_root, path_imgs, index):
    imgs = [os.path.join(input_root, path) for path in path_imgs]
    return imread(imgs[index])


# Assume avg focal length and centered camera
def calib_loader():
    intrinsics_c1 = torch.zeros(3, 3)
    intrinsics_c1[2, 2] = 1
    intrinsics_c1[0, 0] = 718
    intrinsics_c1[1, 1] = 604
    intrinsics_c1[0, 2] = 1200 / 2
    intrinsics_c1[1, 2] = 384 / 2

    return intrinsics_c1


class ListDataset(data.Dataset):
    def __init__(self, input_root, path_list, disp=False, of=False, transform=None,
                 target_transform=None, co_transform=None, max_pix=100, reference_transform=None, fix=False,
                 resize_crop_transform=None, resize_crop_transform_pose=None):
        self.input_root = input_root
        self.path_list = path_list
        self.transform = transform
        self.reference_transform = reference_transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.disp = disp
        self.of = of
        self.input_loader = img_loader
        self.max = max_pix
        self.fix_order = fix
        self.resize_crop_transform = resize_crop_transform
        self.resize_crop_transform_pose = resize_crop_transform_pose

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        inputs_l, inputs_r, targets = self.path_list[index]
        file_name = os.path.basename(inputs_l[cam1_index])[:-4]

        # Randomly pick frame from left or right
        if random.random() < 0.5:
            inputs = inputs_l
        else:
            inputs = inputs_r

        # Randomly pick continious or skip one frame
        if random.random() < 0.5:
            cam_id_0 = cam1_index_0
        else:
            cam_id_0 = cam1_index_00
        if random.random() < 0.5:
            cam_id_1 = cam1_index_1
        else:
            cam_id_1 = cam1_index_11

        cam_id = cam1_index
        intrinsics = calib_loader()
        ign_mask = 0

        # Read inputs at original (or) resolution, brightness, orientation, etc.
        if random.random() < 0.5 or self.fix_order:
            x_pix = self.max
            inputs_or = [self.input_loader(self.input_root, inputs, cam_id_0),
                      self.input_loader(self.input_root, inputs, cam_id),
                      self.input_loader(self.input_root, inputs, cam_id_1)]
        else:
            x_pix = -self.max
            inputs_or = [self.input_loader(self.input_root, inputs, cam_id_1),
                      self.input_loader(self.input_root, inputs, cam_id),
                      self.input_loader(self.input_root, inputs, cam_id_0)]

        grid = None
        # if self.reference_transform is not None:
        #     inputs[0] = self.reference_transform(inputs[0])

        # Apply resize followed by random crop for disp and pose networks
        if self.resize_crop_transform is not None:
            inputs, h, w, factor, _, intrinsics, grid = self.resize_crop_transform(inputs_or, targets, intrinsics)
        if self.resize_crop_transform_pose is not None:
            inputs_or = self.resize_crop_transform_pose(inputs_or)

        # Append images for pose network
        inputs = inputs + inputs_or
        if self.co_transform is not None:
            inputs, _, intrinsics, grid = self.co_transform(inputs, targets, intrinsics, grid)
        if self.transform is not None:
            for i in range(len(inputs)):
                inputs[i] = self.transform(inputs[i])

        # Transform grid to tensor (note grid is relative to disparity network inputs)
        if grid is not None:
            grid = np.transpose(grid, (2, 0, 1))
            grid = torch.from_numpy(grid.copy()).float()

        return inputs[len(inputs) - len(inputs_or)::], inputs[0:len(inputs) - len(inputs_or)], \
               intrinsics, grid, x_pix, h, w, factor, file_name, ign_mask
