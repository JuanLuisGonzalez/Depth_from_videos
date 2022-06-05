from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage
from PIL import Image
import torch.nn.functional as F
import scipy.stats as stats

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target, intrinsics, grid):
        for t in self.co_transforms:
            input, target, intrinsics, grid = t(input, target, intrinsics, grid)
        return input, target, intrinsics, grid


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array.copy())
        # put it from HWC to CHW format
        return tensor.float()


class ArrayToIntTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array.copy())
        # put it from HWC to CHW format
        return tensor.int()


class RandomResizeCrop_hw(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, down, up):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.s_factor = (down, up)

    def __call__(self, inputs, targets=None, intrinsics=None, get_grid=True):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        o_inputs = []
        o_targets = None

        min_factor = max(max((th + 1) / h, (tw + 1) / w), self.s_factor[0])  # plus one to ensure
        max_factor = self.s_factor[1]
        factor = np.random.uniform(low=min_factor, high=max_factor)

        for i in range(len(inputs)):
            input = Image.fromarray(inputs[i]).resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
            o_inputs.append(np.array(input))

        if targets is not None:
            o_targets = []
            for i in range(len(targets)):
                target = Image.fromarray(targets[i]).resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
                o_targets.append(np.array(target))

        # get grid
        a_grid = None
        if get_grid:
            i_tetha = torch.zeros(1, 2, 3)
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            a_grid = F.affine_grid(i_tetha, torch.Size([1, 3, int(h * factor), int(w * factor)]), align_corners=True)
            a_grid = a_grid[0, :, :, :].numpy()

        h, w, _ = o_inputs[0].shape
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for i in range(len(o_inputs)):
            o_inputs[i] = o_inputs[i][y1: y1 + th, x1: x1 + tw]
            o_inputs[i][o_inputs[i] > 255] = 255

        if o_targets is not None:
            for i in range(len(o_targets)):
                o_targets[i] = o_targets[i][y1: y1 + th, x1: x1 + tw]

        if intrinsics is not None:
            intrinsics = intrinsics * factor
            intrinsics[2, 2] = 1
            intrinsics[0, 2] = intrinsics[0, 2] - x1
            intrinsics[1, 2] = intrinsics[1, 2] - y1

        if a_grid is not None:
            a_grid = a_grid[y1: y1 + th, x1: x1 + tw]
        return o_inputs, h, w, factor, o_targets, intrinsics, a_grid


class RandomResizeCrop_hwn(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, down, up, mean, std):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.s_factor = (down, up)
        self.mean = mean
        self.std = std

    def __call__(self, inputs, targets=None, intrinsics=None, get_grid=True):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        o_inputs = []
        o_targets = None

        min_factor = max(max((th + 1) / h, (tw + 1) / w), self.s_factor[0])  # plus one to ensure
        max_factor = self.s_factor[1]
        factor = stats.truncnorm.rvs((min_factor - self.mean) / self.std, (max_factor - self.mean) / self.std,
                                 loc=self.mean, scale=self.std)
        if type(factor) is list:
            factor = factor[0]
        factor = max(min(factor, max_factor), min_factor)

        for i in range(len(inputs)):
            input = Image.fromarray(inputs[i]).resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
            o_inputs.append(np.array(input))

        if targets is not None:
            o_targets = []
            for i in range(len(targets)):
                target = Image.fromarray(targets[i]).resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
                o_targets.append(np.array(target))

        # get grid
        a_grid = None
        if get_grid:
            i_tetha = torch.zeros(1, 2, 3)
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            a_grid = F.affine_grid(i_tetha, torch.Size([1, 3, int(h * factor), int(w * factor)]), align_corners=True)
            a_grid = a_grid[0, :, :, :].numpy()

        h, w, _ = o_inputs[0].shape
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for i in range(len(o_inputs)):
            o_inputs[i] = o_inputs[i][y1: y1 + th, x1: x1 + tw]
            o_inputs[i][o_inputs[i] > 255] = 255

        if o_targets is not None:
            for i in range(len(o_targets)):
                o_targets[i] = o_targets[i][y1: y1 + th, x1: x1 + tw]

        if intrinsics is not None:
            intrinsics = intrinsics * factor
            intrinsics[2, 2] = 1
            intrinsics[0, 2] = intrinsics[0, 2] - x1
            intrinsics[1, 2] = intrinsics[1, 2] - y1

        if a_grid is not None:
            a_grid = a_grid[y1: y1 + th, x1: x1 + tw]
        return o_inputs, h, w, o_targets, intrinsics, a_grid


class Resize(object):
    """Resizes given input images"""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs):
        th, tw = self.size
        o_inputs = []

        for i in range(len(inputs)):
            input = Image.fromarray(inputs[i]).resize((int(tw), int(th)), resample=Image.BICUBIC)
            o_inputs.append(np.array(input))

        return o_inputs


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
        if doing this on disparity estimation you need both disparities left and right need stereo targets
    """

    def __init__(self, is_stereo=False, is_video=False, is_stereo_video=False):
        self.stereo = is_stereo
        self.video = is_video
        self.stereo_video = is_stereo_video

    def __call__(self, inputs, targets=None, intrinsics=None, grid=None):
        o_inputs = []
        o_target = []

        if random.random() < 0.5:
            if intrinsics is not None:
                h, w, _ = inputs[0].shape
                intrinsics[0, 2] = w - intrinsics[0, 2]

            if self.stereo:
                o_inputs.append(np.copy(np.fliplr(inputs[1])))
                o_inputs.append(np.copy(np.fliplr(inputs[0])))
                if grid is not None:
                    grid[:, :, 0] = -grid[:, :, 0]
                    grid = np.copy(np.fliplr(grid))
                return o_inputs, targets, intrinsics, grid
            if self.video:
                for i in range(len(inputs)):
                    o_inputs.append(np.copy(np.fliplr(inputs[i])))
                if grid is not None:
                    grid[:, :, 0] = -grid[:, :, 0]
                    grid = np.copy(np.fliplr(grid))
                return o_inputs, targets, intrinsics, grid
            if self.stereo_video:
                for i in range(3):
                    o_inputs.append(np.copy(np.fliplr(inputs[i])))
                o_inputs.append(np.copy(np.fliplr(inputs[5])))
                o_inputs.append(np.copy(np.fliplr(inputs[4])))
                if grid is not None:
                    grid[:, :, 0] = -grid[:, :, 0]
                    grid = np.copy(np.fliplr(grid))
                return o_inputs, targets, intrinsics, grid
        else:
            return inputs, targets, intrinsics, grid


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
        if doing this on disparity estimation you need both disparities left and right need stereo targets
    """

    def __init__(self, is_stereo=False, is_video=False, is_stereo_video=False):
        self.stereo = is_stereo
        self.video = is_video
        self.stereo_video = is_stereo_video

    def __call__(self, inputs, targets=None, intrinsics=None, grid=None):
        o_inputs = []
        o_target = []

        if random.random() < 0.5:
            if intrinsics is not None:
                h, w, _ = inputs[0].shape
                intrinsics[1, 2] = h - intrinsics[1, 2]

            if self.stereo:
                o_inputs.append(np.copy(np.flipud(inputs[0])))
                o_inputs.append(np.copy(np.flipud(inputs[1])))
                if grid is not None:
                    grid = np.copy(np.flipud(grid))
                return o_inputs, targets, intrinsics, grid
            if self.video:
                for i in range(len(inputs)):
                    o_inputs.append(np.copy(np.flipud(inputs[i])))
                if grid is not None:
                    grid = np.copy(np.flipud(grid))
                return o_inputs, targets, intrinsics, grid
            if self.stereo_video:
                for i in range(3):
                    o_inputs.append(np.copy(np.flipud(inputs[i])))
                o_inputs.append(np.copy(np.flipud(inputs[5])))
                o_inputs.append(np.copy(np.flipud(inputs[4])))
                if grid is not None:
                    grid = np.copy(np.flipud(grid))
                return o_inputs, targets, intrinsics, grid
        else:
            return inputs, targets, intrinsics, grid


class RandomDownUp(object):
    def __init__(self, max_down):
        self.down_factor = max_down

    def __call__(self, input):
        factor = np.random.uniform(low=1 / self.down_factor, high=1)
        h, w, _ = input.shape
        input = Image.fromarray(input)
        input = input.resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
        input = input.resize((int(w), int(h)), resample=Image.BICUBIC)
        input = np.array(input)
        return input


class RandomGamma(object):
    def __init__(self, min=1, max=1):
        self.min = min
        self.max = max
        self.A = 255

    def __call__(self, inputs, targets=None, intrinsics=None, grid=None):
        if random.random() < 0.5:
            factor = random.uniform(self.min, self.max)
            for i in range(len(inputs)):
                inputs[i] = self.A * ((inputs[i] / 255) ** factor)
            return inputs, targets, intrinsics, grid
        else:
            return inputs, targets, intrinsics, grid


class RandomBrightness(object):
    def __init__(self, min=0, max=0):
        self.min = min
        self.max = max

    def __call__(self, inputs, targets=None, intrinsics=None, grid=None):
        if random.random() < 0.5:
            factor = random.uniform(self.min, self.max)
            for i in range(len(inputs)):
                inputs[i] = inputs[i] * factor
                inputs[i][inputs[i] > 255] = 255
            return inputs, targets, intrinsics, grid
        else:
            return inputs, targets, intrinsics, grid


class RandomCBrightness(object):
    def __init__(self, min=0, max=0):
        self.min = min
        self.max = max

    def __call__(self, inputs, targets=None, intrinsics=None, grid=None):
        if random.random() < 0.5:
            for c in range(3):
                factor = random.uniform(self.min, self.max)
                for i in range(len(inputs)):
                    inputs[i][:, :, c] = inputs[i][:, :, c] * factor
                inputs[i][inputs[i] > 255] = 255
            return inputs, targets, intrinsics, grid
        else:
            return inputs, targets, intrinsics, grid



class RandomMSResizeCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, down, up, crop_down, crop_up):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.s_factor = (down, up)
        self.crop_factor = (crop_down, crop_up)

    def __call__(self, inputs, targets=None):
        h, w, _ = inputs[0].shape
        th, tw = self.size

        min_factor = max(max((th + 1) / h, (tw + 1) / w), self.s_factor[0])  # plus one to ensure
        max_factor = self.s_factor[1]
        factor = np.random.uniform(low=min_factor, high=max_factor)

        down_inputs = []
        up_inputs = []

        for i in range(len(inputs)):
            input = Image.fromarray(inputs[i]).resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
            dinput = Image.fromarray(inputs[i]).resize((int(w * factor * self.crop_factor[0]),
                                                        int(h * factor * self.crop_factor[0])), resample=Image.BICUBIC)
            uinput = Image.fromarray(inputs[i]).resize((int(w * factor * self.crop_factor[1]),
                                                        int(h * factor * self.crop_factor[1])), resample=Image.BICUBIC)
            inputs[i] = np.array(input)
            down_inputs.append(np.array(dinput))
            up_inputs.append(np.array(uinput))
        if targets is not None:
            for i in range(len(targets)):
                target = Image.fromarray(targets[i]).resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
                targets[i] = np.array(target)

        # get grid
        i_tetha = torch.zeros(1, 2, 3)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        a_grid = F.affine_grid(i_tetha, torch.Size([1, 3, int(h * factor), int(w * factor)]), align_corners=True)
        inputs.append(a_grid[0, :, :, :].numpy())

        h, w, _ = inputs[0].shape
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for i in range(len(inputs)):
            inputs[i] = inputs[i][y1: y1 + th, x1: x1 + tw]
            inputs[i][inputs[i] > 255] = 255
        for i in range(len(down_inputs)):
            down_inputs[i] = down_inputs[i][int(y1 * self.crop_factor[0]+0.5): int((y1 + th) * self.crop_factor[0]+0.5),
                             int(x1 * self.crop_factor[0]+0.5): int((x1 + tw) * self.crop_factor[0]+0.5)]
            down_inputs[i] = Image.fromarray(down_inputs[i]).resize((int(tw * self.crop_factor[0]),
                                                           int(th * self.crop_factor[0])), resample=Image.BICUBIC)
            down_inputs[i] = np.array(down_inputs[i])
            down_inputs[i][down_inputs[i] > 255] = 255
        for i in range(len(up_inputs)):
            up_inputs[i] = up_inputs[i][int(y1 * self.crop_factor[1]+0.5): int((y1 + th) * self.crop_factor[1]+0.5),
                             int(x1 * self.crop_factor[1]+0.5): int((x1 + tw) * self.crop_factor[1]+0.5)]
            up_inputs[i] = Image.fromarray(up_inputs[i]).resize((int(tw * self.crop_factor[1]),
                                                           int(th * self.crop_factor[1])), resample=Image.BICUBIC)
            up_inputs[i] = np.array(up_inputs[i])
            up_inputs[i][up_inputs[i] > 255] = 255
        if targets is not None:
            for i in range(len(targets)):
                targets[i] = targets[i][y1: y1 + th, x1: x1 + tw]
        return [inputs, down_inputs, up_inputs], targets


class RandomMSHorizontalFlipG(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
        if doing this on disparity estimation you need both disparities left and right need stereo targets
    """

    def __init__(self, disp=False, of=False):
        self.of = of
        self.disp = disp

    def __call__(self, inputs, targets=None):
        o_inputs = []
        od_inputs = []
        ou_inputs = []
        o_target = []

        if random.random() < 0.5:
            o_inputs.append(np.copy(np.fliplr(inputs[0][1])))
            o_inputs.append(np.copy(np.fliplr(inputs[0][0])))
            od_inputs.append(np.copy(np.fliplr(inputs[1][1])))
            od_inputs.append(np.copy(np.fliplr(inputs[1][0])))
            ou_inputs.append(np.copy(np.fliplr(inputs[2][1])))
            ou_inputs.append(np.copy(np.fliplr(inputs[2][0])))
            inputs[0][2][:,:,0] = -inputs[0][2][:,:,0]
            o_inputs.append(np.copy(np.fliplr(inputs[0][2])))
            return [o_inputs, od_inputs, ou_inputs], targets
        else:
            return inputs, targets


class RandomMSGamma(object):
    def __init__(self, min=1, max=1):
        self.min = min
        self.max = max
        self.A = 255

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            factor = random.uniform(self.min, self.max)
            for i in range(2):
                inputs[0][i] = self.A * ((inputs[0][i] / 255) ** factor)
                inputs[1][i] = self.A * ((inputs[1][i] / 255) ** factor)
                inputs[2][i] = self.A * ((inputs[2][i] / 255) ** factor)
            return inputs, targets
        else:
            return inputs, targets


class RandomMSBrightness(object):
    def __init__(self, min=0, max=0):
        self.min = min
        self.max = max

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            factor = random.uniform(self.min, self.max)
            for i in range(2):
                inputs[0][i] = inputs[0][i] * factor
                inputs[0][i][inputs[0][i] > 255] = 255
                inputs[1][i] = inputs[1][i] * factor
                inputs[1][i][inputs[1][i] > 255] = 255
                inputs[2][i] = inputs[2][i] * factor
                inputs[2][i][inputs[2][i] > 255] = 255
            return inputs, targets
        else:
            return inputs, targets


class RandomMSCBrightness(object):
    def __init__(self, min=0, max=0):
        self.min = min
        self.max = max

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            for i in range(2):
                for c in range(3):
                    factor = random.uniform(self.min, self.max)
                    inputs[0][i][:, :, c] = inputs[0][i][:, :, c] * factor
                    inputs[1][i][:, :, c] = inputs[1][i][:, :, c] * factor
                    inputs[2][i][:, :, c] = inputs[2][i][:, :, c] * factor
                inputs[0][i][inputs[0][i] > 255] = 255
                inputs[1][i][inputs[1][i] > 255] = 255
                inputs[2][i][inputs[2][i] > 255] = 255
            return inputs, targets
        else:
            return inputs, targets