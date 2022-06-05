import torch.utils.data as data
import os
import os.path
from imageio import imread
from .pfm import readPFM
import numpy as np
import scipy.io as sio
from PIL import Image

LR_DATASETS = ['Monkaa', 'Driving', 'eigen_val']
L_DATASETS = ['Sintel', 'Kitti2015', 'Monkaa_mono', 'Driving_mono', 'Kitti2011']

# Indexes for lr models
Lt_indexlr = 0
Rt_indexlr = 1
Lt1_indexlr = 2
Rt1_indexlr = 3
Dl_indexlr = 0
Dr_indexlr = 1
Ofl_indexlr = 2
Ofr_indexlr = 3

# indexes for l models
Lt_index = 0
Rt_index = 1
Lt1_index = 2
Rt1_index = 3
Dl_index = 0
Ofl_index = 1

def Make3Ddisp_loader(input_root, path_imgs, index):
    disp = [os.path.join(input_root, path) for path in path_imgs]
    disp = sio.loadmat(disp[index], verify_compressed_data_integrity=False)
    disp = disp['Position3DGrid'][:,:,3]
    disp = Image.fromarray(disp).resize((1704, 2272), resample=Image.NEAREST)
    disp = np.array(disp)
    return disp[:, :, np.newaxis]

def img_loader(input_root, path_imgs, index):
    imgs = [os.path.join(input_root, path) for path in path_imgs]
    return imread(imgs[index])


def pfm_loader(target_root, path_pfm, index):
    pfms = [os.path.join(target_root, path) for path in path_pfm]
    return readPFM(pfms[index])


def rgbdisp_loader(target_root, path_rgbdisp, index):
    rgbdisp = [os.path.join(target_root, path) for path in path_rgbdisp]
    rgbdisp = disparity_read(rgbdisp[index])  # is returned in h, w
    return rgbdisp[:, :, np.newaxis]


def rgbflow_loader(target_root, path_rgbflow, index):
    rgbflow = [os.path.join(target_root, path) for path in path_rgbflow]
    return flow_read(rgbflow[index])


def kittidisp_loader(input_root, path_imgs, index):
    disp = [os.path.join(input_root, path) for path in path_imgs]
    disp = imread(disp[index])
    disp = disp / 256
    return disp[:, :, np.newaxis]


class ListDataset(data.Dataset):
    def __init__(self, input_root, target_root, path_list, disp=False, of=False, data_name='Monkaa', transform=None,
                 target_transform=None,
                 co_transform=None):
        self.input_root = input_root
        self.target_root = target_root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.disp = disp
        self.of = of
        self.data_name = data_name

        if data_name == 'Kitti2015' or data_name == 'eigen_val':  # for kitty if disp is 0 is nan (or -1)
            self.input_loader = img_loader
            if self.of:
                self.target_loader = rgbflow_loader
            elif self.disp:
                self.target_loader = kittidisp_loader
        elif data_name == 'Make3D':  # for kitty if disp is 0 is nan (or -1)
            self.input_loader = img_loader
            if self.disp:
                self.target_loader = Make3Ddisp_loader
        elif data_name == 'Sintel':
            self.input_loader = img_loader
            if self.of:
                self.target_loader = rgbflow_loader
            elif self.disp:
                self.target_loader = rgbdisp_loader
        else:
            self.input_loader = img_loader
            self.target_loader = pfm_loader

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]
        img_index = Rt_indexlr
        if self.data_name in LR_DATASETS:
            if self.of:
                img_index = Lt1_index
                targets = [self.target_loader(self.target_root, targets, Ofl_indexlr),
                           self.target_loader(self.target_root, targets, Ofr_indexlr)]
            elif self.disp:
                targets = [self.target_loader(self.target_root, targets, Dl_indexlr),
                           self.target_loader(self.target_root, targets, Dr_indexlr)]
        else:
            if self.of:
                img_index = Lt1_index
                targets = [self.target_loader(self.target_root, targets, Ofl_indexlr)]
            elif self.disp:
                targets = [self.target_loader(self.target_root, targets, Dl_indexlr)]

        file_name = os.path.basename(inputs[Lt_index])[:-4]
        inputs = [self.input_loader(self.input_root, inputs, Lt_index),
                  self.input_loader(self.input_root, inputs, img_index)]

        if self.co_transform is not None:
            inputs, targets = self.co_transform(inputs, targets)
        if self.transform is not None:
            for i in range(len(inputs)):
                inputs[i] = self.transform(inputs[i])
        if targets is None:
            return inputs, 0, file_name

        if self.target_transform is not None:
            for i in range(len(targets)):
                targets[i] = self.target_transform(targets[i])
                if targets[i].shape[0] > 2:
                    targets[i] = targets[i][0:2, :, :]  # remove 1 channel filled with 0s in optical flows
        return inputs, targets, file_name
