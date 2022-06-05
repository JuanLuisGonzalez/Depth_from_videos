import os.path
import glob
from random import shuffle
from .listdataset_video_ms_ghw_cs import ListDataset as video_ms_ghwListDataset


def make_dataset(main_dir, skip):
    train_directories = []
    val_directories = []
    directories = (train_directories, val_directories)
    left_folder = 'leftImg8bit_sequence'
    right_folder = 'rightImg8bit_sequence'
    left_img_folder = os.path.join(main_dir, left_folder)
    for ttv_dirs in os.listdir(left_img_folder):
        if os.path.isfile(os.path.join(left_img_folder, ttv_dirs)):
            continue
        if ttv_dirs == 'val':
            selector = 1  # put that in the validation folder
        else:
            selector = 0
        for city_dirs in os.listdir(os.path.join(left_img_folder, ttv_dirs)):
            if os.path.isfile(os.path.join(left_img_folder, ttv_dirs, city_dirs)):
                continue
            left_dir = os.path.join(left_img_folder, ttv_dirs, city_dirs)
            for calibration_filename in glob.iglob(os.path.join(left_dir, '*.json')):
                # file format  cityname_000000_000000_camera.JSON
                calibration_filename = os.path.basename(calibration_filename)
                left_img_filename = calibration_filename[:-len('camera.json')] + 'leftImg8bit.jpg'
                cityname_ = left_img_filename[
                            :-len('000000_000000_leftImg8bit.png')]  # remove 000000_000000_leftImg8bit.png
                scene_number = left_img_filename[len(cityname_):len(cityname_) + 6]
                calib_image_number = int(left_img_filename[len(cityname_) + 7:len(cityname_) + 7 + 6])

                for image_number in range(calib_image_number - (19 - 2), calib_image_number + (10 - 2)):
                    # rgb input left
                    left_t00 = os.path.join(left_folder, ttv_dirs, city_dirs, cityname_ + scene_number + '_' +
                                            '{:06d}'.format(image_number - 2 * skip) + '_leftImg8bit.jpg')
                    left_t0 = os.path.join(left_folder, ttv_dirs, city_dirs, cityname_ + scene_number + '_' +
                                           '{:06d}'.format(image_number - 1 * skip) + '_leftImg8bit.jpg')
                    left_t = os.path.join(left_folder, ttv_dirs, city_dirs, cityname_ + scene_number + '_' +
                                          '{:06d}'.format(image_number) + '_leftImg8bit.jpg')
                    left_t1 = os.path.join(left_folder, ttv_dirs, city_dirs, cityname_ + scene_number + '_' +
                                           '{:06d}'.format(image_number + 1 * skip) + '_leftImg8bit.jpg')
                    left_t11 = os.path.join(left_folder, ttv_dirs, city_dirs, cityname_ + scene_number + '_' +
                                            '{:06d}'.format(image_number + 2 * skip) + '_leftImg8bit.jpg')

                    # rgb input right
                    right_t00 = os.path.join(right_folder, ttv_dirs, city_dirs, cityname_ + scene_number + '_' +
                                             '{:06d}'.format(image_number - 2 * skip) + '_rightImg8bit.jpg')
                    right_t0 = os.path.join(right_folder, ttv_dirs, city_dirs, cityname_ + scene_number + '_' +
                                            '{:06d}'.format(image_number - 1 * skip) + '_rightImg8bit.jpg')
                    right_t = os.path.join(right_folder, ttv_dirs, city_dirs, cityname_ + scene_number + '_' +
                                           '{:06d}'.format(image_number) + '_rightImg8bit.jpg')
                    right_t1 = os.path.join(right_folder, ttv_dirs, city_dirs, cityname_ + scene_number + '_' +
                                            '{:06d}'.format(image_number + 1 * skip) + '_rightImg8bit.jpg')
                    right_t11 = os.path.join(right_folder, ttv_dirs, city_dirs, cityname_ + scene_number + '_' +
                                             '{:06d}'.format(image_number + 2 * skip) + '_rightImg8bit.jpg')

                    # Check images are available
                    lt00_isfile = os.path.isfile(os.path.join(main_dir, left_t00))
                    lt0_isfile = os.path.isfile(os.path.join(main_dir, left_t0))
                    lt_isfile = os.path.isfile(os.path.join(main_dir, left_t))
                    lt1_isfile = os.path.isfile(os.path.join(main_dir, left_t1))
                    lt11_isfile = os.path.isfile(os.path.join(main_dir, left_t11))
                    rt00_isfile = os.path.isfile(os.path.join(main_dir, right_t00))
                    rt0_isfile = os.path.isfile(os.path.join(main_dir, right_t0))
                    rt_isfile = os.path.isfile(os.path.join(main_dir, right_t))
                    rt1_isfile = os.path.isfile(os.path.join(main_dir, right_t1))
                    rt11_isfile = os.path.isfile(os.path.join(main_dir, right_t11))

                    calibration = os.path.join(left_folder, ttv_dirs, city_dirs, calibration_filename)
                    calib_isfile = os.path.isfile(os.path.join(main_dir, calibration))

                    # If all files are valid append sample to data list
                    if lt0_isfile and lt_isfile and lt1_isfile and rt0_isfile and rt_isfile and rt1_isfile \
                            and lt00_isfile and lt11_isfile and rt00_isfile and rt11_isfile and calib_isfile:
                        directories[selector].append([[left_t00, left_t0, left_t, left_t1, left_t11],
                                                      [right_t00, right_t0, right_t, right_t1, right_t11],
                                                      calibration, None])

    return directories[0], directories[1]


def Cityscapes_video_jpg(split, **kwargs):
    input_root = kwargs.pop('root')
    resize_crop_transform_pose = kwargs.pop('resize_crop_transform_pose', None)
    resize_crop_transform = kwargs.pop('resize_crop_transform', None)
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    reference_transform = kwargs.pop('reference_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    max_pix = kwargs.pop('max_pix', 100)
    fix = kwargs.pop('fix', False)
    skip_frames = kwargs.pop('skip_frames', 1)

    train_list, test_list = make_dataset(input_root, skip_frames)

    train_dataset = video_ms_ghwListDataset(input_root, train_list, disp=False, of=False,
                                            transform=transform, target_transform=target_transform,
                                            co_transform=co_transform, resize_crop_transform=resize_crop_transform,
                                            resize_crop_transform_pose=resize_crop_transform_pose,
                                            max_pix=max_pix, reference_transform=reference_transform, fix=fix)
    test_dataset = video_ms_ghwListDataset(input_root, test_list, disp=False, of=False,
                                           transform=transform, target_transform=target_transform, fix=fix)

    return train_dataset, test_dataset


def Cityscapes_video_jpg_list(split, **kwargs):
    input_root = kwargs.pop('root')
    train_list, test_list = make_dataset(input_root, split)
    return train_list, test_list
