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

    return [intrinsics_c1, intrinsics_c5]


class ListDataset(data.Dataset):
    def __init__(self, input_root, target_root, path_list, disp=False, of=False, transform=None,
                 target_transform=None, co_transform=None, max_pix=100, reference_transform=None, fix=False,
                 resize_crop_transform=None, stereo=True, video=False, stereo_video=False):
        self.input_root = input_root
        self.target_root = target_root
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
        self.stereo = stereo
        self.video = video
        self.stereo_video = stereo_video

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        inputs, calibration, targets = self.path_list[index]
        file_name = os.path.basename(inputs[cam1_index])[:-4]
        intrinsics_15 = calib_loader(self.input_root, calibration)
        intrinsics = None
        ign_mask = 0 # flag to ignore bottom part of images of lateral views

        if self.stereo:
            if random.random() < 0.5 or self.fix_order:
                x_pix = self.max
                inputs = [self.input_loader(self.input_root, inputs, cam5_index),
                          self.input_loader(self.input_root, inputs, cam1_index)]
            else:
                x_pix = -self.max
                inputs = [self.input_loader(self.input_root, inputs, cam1_index),
                          self.input_loader(self.input_root, inputs, cam5_index)]

        if self.video:
            if random.random() < 0.5:
                cam_id_0 = cam1_index_0
                cam_id = cam1_index
                cam_id_1 = cam1_index_1
                intrinsics = intrinsics_15[0]
                ign_mask = 0
            else:
                cam_id_0 = cam5_index_0
                cam_id = cam5_index
                cam_id_1 = cam5_index_1
                intrinsics = intrinsics_15[1]
                ign_mask = 1

            if random.random() < 0.5 or self.fix_order:
                x_pix = self.max
                inputs = [self.input_loader(self.input_root, inputs, cam_id_0),
                          self.input_loader(self.input_root, inputs, cam_id),
                          self.input_loader(self.input_root, inputs, cam_id_1)]
            else:
                x_pix = -self.max
                inputs = [self.input_loader(self.input_root, inputs, cam_id_1),
                          self.input_loader(self.input_root, inputs, cam_id),
                          self.input_loader(self.input_root, inputs, cam_id_0)]

        if self.stereo_video:
            if random.random() < 0.5 or self.fix_order:
                x_pix = self.max
                inputs = [self.input_loader(self.input_root, inputs, cam1_index_0),
                          self.input_loader(self.input_root, inputs, cam1_index),
                          self.input_loader(self.input_root, inputs, cam1_index_1),
                          self.input_loader(self.input_root, inputs, cam5_index),
                          self.input_loader(self.input_root, inputs, cam6_index)]
            else:
                x_pix = -self.max
                inputs = [self.input_loader(self.input_root, inputs, cam1_index_1),
                          self.input_loader(self.input_root, inputs, cam1_index),
                          self.input_loader(self.input_root, inputs, cam1_index_0),
                          self.input_loader(self.input_root, inputs, cam6_index),
                          self.input_loader(self.input_root, inputs, cam5_index)]

        grid = None
        # if self.reference_transform is not None:
        #     inputs[0] = self.reference_transform(inputs[0])
        if self.resize_crop_transform is not None:
            inputs, h, w, _, intrinsics, grid = self.resize_crop_transform(inputs, targets, intrinsics)
        if self.co_transform is not None:
            inputs, _, intrinsics, grid = self.co_transform(inputs, targets, intrinsics, grid)
        if self.transform is not None:
            for i in range(len(inputs)):
                inputs[i] = self.transform(inputs[i])

        # for grid
        if grid is not None:
            grid = np.transpose(grid, (2, 0, 1))
            grid = torch.from_numpy(grid.copy()).float()

        return inputs, intrinsics, grid, x_pix, h, w, file_name, ign_mask
