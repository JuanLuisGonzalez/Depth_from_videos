import os.path
import glob
from .listdataset import ListDataset
from random import shuffle
from .listdataset_pan import ListDataset as panListDataset
from .listdataset_pan_g import ListDataset as pangListDataset
from .listdataset_mspan_g import ListDataset as mspangListDataset
from .listdataset_pan_ghw import ListDataset as panghwListDataset


def make_dataset(main_dir, split):
    train_directories = []
    val_directories = []
    directories = (train_directories, val_directories)
    selector = 0
    left_img_folder = os.path.join(main_dir, 'leftImg8bit')
    for ttv_dirs in os.listdir(left_img_folder):
        if os.path.isfile(os.path.join(left_img_folder, ttv_dirs)):
            continue
        if ttv_dirs == 'val':
            selector = 1 # put that in the validation folder
        else:
            selector = 0
        for city_dirs in os.listdir(os.path.join(left_img_folder, ttv_dirs)):
            if os.path.isfile(os.path.join(left_img_folder, ttv_dirs, city_dirs)):
                continue
            left_dir = os.path.join(left_img_folder, ttv_dirs, city_dirs)
            for target in glob.iglob(os.path.join(left_dir, '*.jpg')):
                target = os.path.basename(target)
                root_filename = target[:-15]  # remove leftImg8bit.png
                imgl_t = os.path.join('leftImg8bit', ttv_dirs, city_dirs, root_filename + 'leftImg8bit.jpg')  # rgb input left
                imgr_t = os.path.join('rightImg8bit', ttv_dirs, city_dirs, root_filename + 'rightImg8bit.jpg')  # rgb input right

                # Check valid files
                if not (os.path.isfile(os.path.join(main_dir, imgl_t))
                        and os.path.isfile(os.path.join(main_dir, imgr_t))):
                    continue
                directories[selector].append([[imgl_t, imgr_t], None])

    return directories[0], directories[1]


def Cityscapes_jpg(split, **kwargs):
    input_root = kwargs.pop('root')
    resize_crop_transform = kwargs.pop('resize_crop_transform', None)
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    reference_transform = kwargs.pop('reference_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    use_pan_dataset = kwargs.pop('use_pan', False)
    use_panghw_dataset = kwargs.pop('use_panghw', False)
    use_pang_dataset = kwargs.pop('use_pang', False)
    use_mspang_dataset = kwargs.pop('use_mspang', False)
    max_pix = kwargs.pop('max_pix', 100)
    fix = kwargs.pop('fix', False)

    train_list, test_list = make_dataset(input_root, split)

    if use_pan_dataset:
        train_dataset = panListDataset(input_root, input_root, train_list, data_name='Kitti2015', disp=False, of=False,
                                       transform=transform, target_transform=target_transform, co_transform=co_transform,
                                       max_pix=max_pix, reference_transform=reference_transform, fix=fix)
        shuffle(test_list)
        test_dataset = panListDataset(input_root, input_root, test_list, data_name='Kitti2015', disp=False, of=False,
                                   transform=transform, target_transform=target_transform, fix=fix)
    elif use_pang_dataset:
        train_dataset = pangListDataset(input_root, input_root, train_list, data_name='Kitti2015', disp=False, of=False,
                                       transform=transform, target_transform=target_transform, co_transform=co_transform,
                                       max_pix=max_pix, reference_transform=reference_transform, fix=fix)
        shuffle(test_list)
        test_dataset = pangListDataset(input_root, input_root, test_list, data_name='Kitti2015', disp=False, of=False,
                                   transform=transform, target_transform=target_transform, fix=fix)
    elif use_panghw_dataset:
        train_dataset = panghwListDataset(input_root, input_root, train_list, data_name='Kitti2015', disp=False, of=False,
                                       transform=transform, target_transform=target_transform,
                                       co_transform=co_transform, resize_crop_transform=resize_crop_transform,
                                       max_pix=max_pix, reference_transform=reference_transform, fix=fix)
        shuffle(test_list)
        test_dataset = panghwListDataset(input_root, input_root, test_list, data_name='Kitti2015', disp=False, of=False,
                                      transform=transform, target_transform=target_transform, fix=fix)
    elif use_mspang_dataset:
        train_dataset = mspangListDataset(input_root, input_root, train_list, data_name='Kitti2015', disp=False, of=False,
                                       transform=transform, target_transform=target_transform,
                                       co_transform=co_transform,
                                       max_pix=max_pix, reference_transform=reference_transform, fix=fix)
        shuffle(test_list)
        test_dataset = mspangListDataset(input_root, input_root, test_list, data_name='Kitti2015', disp=False, of=False,
                                      transform=transform, target_transform=target_transform, fix=fix)
    else:
        train_dataset = ListDataset(input_root, input_root, train_list, data_name='Kitti2015', disp=False, of=False,
                                    transform=transform, target_transform=target_transform, co_transform=co_transform)
        shuffle(test_list)
        test_dataset = ListDataset(input_root, input_root, test_list, data_name='Kitti2015', disp=False, of=False,
                                   transform=transform, target_transform=target_transform)
    return train_dataset, test_dataset


def Cityscapes_list_jpg(split, **kwargs):
    input_root = kwargs.pop('root')
    train_list, test_list = make_dataset(input_root, split)
    return train_list, test_list


