import os.path
from .util import split2list
from .listdataset import ListDataset
from .listdataset_pan_ghw import ListDataset as panghwListDataset
from .listdataset_mspan_g import ListDataset as mspangListDataset
from .listdataset_video_ms_ghw_kitti import ListDataset as video_ms_ghwListDataset
from random import shuffle


def Kitti(split, **kwargs):
    # Data
    input_root = kwargs.pop('root')
    is_eigen = kwargs.pop('is_eigen', False)
    max_pix = kwargs.pop('max_pix', 100)
    fix = kwargs.pop('fix', False)

    # Data agumentations
    resize_crop_transform = kwargs.pop('resize_crop_transform', None)
    resize_crop_transform_pose = kwargs.pop('resize_crop_transform_pose', None)
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    reference_transform = kwargs.pop('reference_transform', None)
    co_transform = kwargs.pop('co_transform', None)

    # Data loader specifics
    use_panghw_dataset = kwargs.pop('use_panghw', False)
    use_mspang_dataset = kwargs.pop('use_mspang', False)
    is_stereo = kwargs.pop('is_stereo', False)
    is_video = kwargs.pop('is_video', False)
    is_video_ms = kwargs.pop('is_video_ms', False)
    is_stereo_video = kwargs.pop('is_stereo_video', False)

    train_list = []
    if is_eigen:
        if is_video or is_video_ms:
            with open("Datasets/kitti_eigen_train.txt", 'r') as f:
                train_lines = list(f.read().splitlines())
                train_lines.sort()
                for i in range(2, len(train_lines) - 2):
                    left_t00, right_t00 = train_lines[i - 2].split(" ")
                    left_t0, right_t0 = train_lines[i - 1].split(" ")
                    left_t, right_t = train_lines[i].split(" ")
                    left_t1, right_t1 = train_lines[i + 1].split(" ")
                    left_t11, right_t11 = train_lines[i + 2].split(" ")
                    # Ensure frames are from same scene (could have some skippers)
                    curr_dir = left_t[0:-len(os.path.basename(left_t)) ]
                    prev_dir = left_t0[0:-len(os.path.basename(left_t0)) ]
                    prev_dir0 = left_t00[0:-len(os.path.basename(left_t00)) ]
                    next_dir = left_t1[0:-len(os.path.basename(left_t1))]
                    next_dir1 = left_t11[0:-len(os.path.basename(left_t11))]
                    if curr_dir != prev_dir or curr_dir != next_dir or \
                            curr_dir != prev_dir0 or curr_dir != next_dir1:
                        continue

                    lt00_isfile = os.path.isfile(os.path.join(input_root, left_t00))
                    lt0_isfile  = os.path.isfile(os.path.join(input_root, left_t0))
                    lt_isfile   = os.path.isfile(os.path.join(input_root, left_t))
                    lt1_isfile  = os.path.isfile(os.path.join(input_root, left_t1))
                    lt11_isfile = os.path.isfile(os.path.join(input_root, left_t11))
                    rt00_isfile = os.path.isfile(os.path.join(input_root, right_t00))
                    rt0_isfile  = os.path.isfile(os.path.join(input_root, right_t0))
                    rt_isfile   = os.path.isfile(os.path.join(input_root, right_t))
                    rt1_isfile  = os.path.isfile(os.path.join(input_root, right_t1))
                    rt11_isfile = os.path.isfile(os.path.join(input_root, right_t11))

                    if lt0_isfile and lt_isfile and lt1_isfile and rt0_isfile and rt_isfile and rt1_isfile\
                            and lt00_isfile and lt11_isfile and rt00_isfile and rt11_isfile:
                        train_list.append([[left_t00, left_t0, left_t, left_t1, left_t11],
                                           [right_t00, right_t0, right_t, right_t1, right_t11], None])
        elif is_stereo:
            with open("Datasets/kitti_eigen_train.txt", 'r') as f:
                train_list = list(f.read().splitlines())
                train_list = [[line.split(" "), None] for line in train_list if
                              os.path.isfile(os.path.join(input_root, line.split(" ")[0]))]
    else:
        if is_video or is_video_ms:
            with open("Datasets/kitti_train_files.txt", 'r') as f:
                train_lines = list(f.read().splitlines())
                train_lines.sort()
                for i in range(1, len(train_lines) - 1):
                    left_t0, right_t0 = train_lines[i - 1].split(" ")
                    left_t, right_t = train_lines[i].split(" ")
                    left_t1, right_t1 = train_lines[i + 1].split(" ")
                    # Ensure frames are from same scene (might have some skippers)
                    curr_dir = left_t[0:-len(os.path.basename(left_t)) ]
                    prev_dir = left_t0[0:-len(os.path.basename(left_t0)) ]
                    next_dir = left_t1[0:-len(os.path.basename(left_t1))]
                    if curr_dir != prev_dir or curr_dir != next_dir:
                        continue

                    lt0_isfile = os.path.isfile(os.path.join(input_root, left_t0))
                    lt_isfile  = os.path.isfile(os.path.join(input_root, left_t))
                    lt1_isfile = os.path.isfile(os.path.join(input_root, left_t1))
                    rt0_isfile = os.path.isfile(os.path.join(input_root, right_t0))
                    rt_isfile  = os.path.isfile(os.path.join(input_root, right_t))
                    rt1_isfile = os.path.isfile(os.path.join(input_root, right_t1))

                    if lt0_isfile and lt_isfile and lt1_isfile and rt0_isfile and rt_isfile and rt1_isfile:
                        train_list.append([[left_t0, left_t, left_t1], None])
                        train_list.append([[right_t0, right_t, right_t1], None])
        elif is_stereo:
            with open("Datasets/kitti_train_files.txt", 'r') as f:
                train_list = list(f.read().splitlines())
                train_list = [[line.split(" "), None] for line in train_list if
                              os.path.isfile(os.path.join(input_root, line.split(" ")[0]))]

    train_list, test_list = split2list(train_list, split)
    if is_video_ms and use_panghw_dataset:
        train_dataset = video_ms_ghwListDataset(input_root, train_list, disp=False, of=False,
                                          transform=transform, target_transform=target_transform,
                                          co_transform=co_transform, resize_crop_transform=resize_crop_transform,
                                          resize_crop_transform_pose=resize_crop_transform_pose,
                                          max_pix=max_pix, reference_transform=reference_transform, fix=fix)
        test_dataset = video_ms_ghwListDataset(input_root, test_list, disp=False, of=False,
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


def Kitti_list(split, **kwargs):
    input_root = kwargs.pop('root')
    with open('kitti_train_files.txt', 'r') as f:
        train_list = list(f.read().splitlines())
    train_list, test_list = split2list(train_list, split)
    return train_list, test_list
