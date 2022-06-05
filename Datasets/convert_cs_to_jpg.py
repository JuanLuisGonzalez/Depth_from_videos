import os.path
import os
import glob
from PIL import Image
import numpy as np


def img_loader(input_root, path_img):
    img = os.path.join(input_root, path_img)
    img = Image.open(img)
    return img


def make_dataset(main_dir):
    directories = []
    left_img_folder = os.path.join(main_dir, 'leftImg8bit_sequence')
    for ttv_dirs in os.listdir(left_img_folder):
        if os.path.isfile(os.path.join(left_img_folder, ttv_dirs)):
            continue
        for city_dirs in os.listdir(os.path.join(left_img_folder, ttv_dirs)):
            if os.path.isfile(os.path.join(left_img_folder, ttv_dirs, city_dirs)):
                continue

            left_dir = os.path.join(left_img_folder, ttv_dirs, city_dirs)

            for target in glob.iglob(os.path.join(left_dir, '*.png')):
                target = os.path.basename(target)
                root_filename = target[:-15]  # remove leftImg8bit.png
                imgl_t = os.path.join('leftImg8bit_sequence', ttv_dirs, city_dirs,
                                      root_filename + 'leftImg8bit.png')  # rgb input left
                imgr_t = os.path.join('rightImg8bit_sequence', ttv_dirs, city_dirs,
                                      root_filename + 'rightImg8bit.png')  # rgb input right

                # Check valid files
                if not (os.path.isfile(os.path.join(main_dir, imgl_t))
                        and os.path.isfile(os.path.join(main_dir, imgr_t))):
                    continue
                directories.append(imgl_t)
                directories.append(imgr_t)

    return directories


# if __name__ == '__main__':
read_path = 'D:\Datasets/Cityscapes_video'
save_path = 'D:\Datasets/Cityscapes_video_jpg'

if not os.path.exists(save_path):
    os.makedirs(save_path)

print('Making dirs')
dirs = make_dataset(read_path)

print('Converting')
for i in range(len(dirs)):
    basename = os.path.basename(dirs[i])
    this_save_path = os.path.join(save_path, dirs[i][:-len(basename)])
    if not os.path.exists(this_save_path):
        os.makedirs(this_save_path)

    if os.path.isfile(os.path.join(this_save_path, basename[:-3] + 'jpg')):
        continue

    img = img_loader(read_path, dirs[i])
    img = np.array(img)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
    h, w, c = img.shape
    img = img[25:h - 200, 100:, :]
    img = Image.fromarray(img)
    img.save(os.path.join(this_save_path, basename[:-3] + 'jpg'))

print('Finish')
