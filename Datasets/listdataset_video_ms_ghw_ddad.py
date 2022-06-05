import torch.utils.data as data
import torch

import os
import os.path
from imageio import imread
import numpy as np
import random
import json

# Indexes for lr models
cam1_index_0 = 0
cam1_index = 1
cam1_index_1 = 2
cam5_index_0 = 3
cam5_index = 4
cam5_index_1 = 5
cam6_index = 6
cam7_index = 7
cam8_index = 8
cam9_index = 9


def img_loader(input_root, path_imgs, index):
    imgs = [os.path.join(input_root, path) for path in path_imgs]
    return imread(imgs[index])


def calib_loader(input_root, calib_path):
    calib_file = os.path.join(input_root, calib_path)
    with open(calib_file) as f:
        calib_json = json.load(f)

    intrinsics_c1 = torch.zeros(3, 3)
    intrinsics_c1[2, 2] = 1
    intrinsics_c1[0, 0] = calib_json['intrinsics'][1]['fx']
    intrinsics_c1[1, 1] = calib_json['intrinsics'][1]['fy']
    intrinsics_c1[0, 2] = calib_json['intrinsics'][1]['cx']
    intrinsics_c1[1, 2] = calib_json['intrinsics'][1]['cy']

    intrinsics_c5 = torch.zeros(3, 3)
    intrinsics_c5[2, 2] = 1
    intrinsics_c5[0, 0] = calib_json['intrinsics'][2]['fx']
    intrinsics_c5[1, 1] = calib_json['intrinsics'][2]['fy']
    intrinsics_c5[0, 2] = calib_json['intrinsics'][2]['cx']
    intrinsics_c5[1, 2] = calib_json['intrinsics'][2]['cy']

    intrinsics_c6 = torch.zeros(3, 3)
    intrinsics_c6[2, 2] = 1
    intrinsics_c6[0, 0] = calib_json['intrinsics'][3]['fx']
    intrinsics_c6[1, 1] = calib_json['intrinsics'][3]['fy']
    intrinsics_c6[0, 2] = calib_json['intrinsics'][3]['cx']
    intrinsics_c6[1, 2] = calib_json['intrinsics'][3]['cy']

    return [intrinsics_c1, intrinsics_c5, intrinsics_c6]


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
        inputs_1, inputs_5, inputs_6, calibration, targets = self.path_list[index]
        file_name = os.path.basename(inputs_1[cam1_index])[:-4]
        intrinsics_156 = calib_loader(self.input_root, calibration)

        rand_no = random.random()
        if rand_no < 1/3:
            inputs = inputs_1
            intrinsics = intrinsics_156[0]
            ign_mask = 0
        elif rand_no < 2/3:
            inputs = inputs_5
            intrinsics = intrinsics_156[1]
            ign_mask = 1
        else:
            inputs = inputs_6
            intrinsics = intrinsics_156[2]
            ign_mask = 1

        # Randomly skip one frame
        curr_idx = 2
        prev_idx = 0
        if random.random() < 0.5:
            prev_idx = 1
        next_idx = 3
        if random.random() < 0.5:
            next_idx = 4

        # Read inputs at original (or) resolution, brightness, orientation, etc.
        # Randomly reverse order if fix_order is false
        if random.random() < 0.5 or self.fix_order:
            x_pix = self.max
            inputs_or = [self.input_loader(self.input_root, inputs, prev_idx),
                      self.input_loader(self.input_root, inputs, curr_idx),
                      self.input_loader(self.input_root, inputs, next_idx)]
        else:
            x_pix = -self.max
            inputs_or = [self.input_loader(self.input_root, inputs, next_idx),
                      self.input_loader(self.input_root, inputs, curr_idx),
                      self.input_loader(self.input_root, inputs, prev_idx)]

        grid = None
        # if self.reference_transform is not None:
        #     inputs[0] = self.reference_transform(inputs[0])

        # Apply resize followed by random crop for disp and pose networks
        if self.resize_crop_transform is not None:
            inputs, h, w, _, intrinsics, grid = self.resize_crop_transform(inputs_or, targets, intrinsics)
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
               intrinsics, grid, x_pix, h, w, file_name, ign_mask
