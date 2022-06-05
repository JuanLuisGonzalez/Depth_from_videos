import os.path
import glob
from .listdataset import ListDataset
from .listdataset_mspan_g import ListDataset as mspangListDataset
from .listdataset_pan_ghw_ddad import ListDataset as panghwListDataset
from .listdataset_video_ms_ghw_ddad import ListDataset as video_ms_ghwListDataset


def make_dataset(main_dir, split):
    # DDAD dataset:
    # Directories 0 - 149 are train
    # Directories 150 - 199 are eval
    start_eval_dir = 150

    train_directories = []
    val_directories = []
    directories = (train_directories, val_directories)
    ddad_train_val = os.path.join(main_dir, 'ddad_train_val')
    for scene_dir in os.listdir(ddad_train_val):
        if os.path.isfile(os.path.join(ddad_train_val, scene_dir)):
            continue
        if int(scene_dir) >= start_eval_dir:
            selector = 1  # put sample in the validation folder
        else:
            selector = 0

        # Load calibration file
        calibration = list(glob.iglob(os.path.join(ddad_train_val, scene_dir, 'calibration', '*.json')))
        calibration = calibration[0]
        is_file_cal = os.path.isfile(os.path.join(main_dir, calibration))
        if is_file_cal:
            calibration = os.path.join('ddad_train_val', scene_dir, 'calibration', os.path.basename(calibration))
        else:
            continue

        ddad_rgb = os.path.join(ddad_train_val, scene_dir, 'rgb')
        cam1_dir = os.path.join(ddad_rgb, 'CAMERA_01')
        file_names = list(glob.iglob(os.path.join(cam1_dir, '*.png')))
        file_names.sort()

        for n_img in range(2, len(file_names) - 2):
            target = os.path.basename(file_names[n_img])
            target0 = os.path.basename(file_names[n_img - 1])
            target00 = os.path.basename(file_names[n_img - 2])
            target1 = os.path.basename(file_names[n_img + 1])
            target11 = os.path.basename(file_names[n_img + 2])

            img1_t00 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_01', target00)
            img1_t0 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_01', target0)
            img1_t = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_01', target)
            img1_t1 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_01', target1)
            img1_t11 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_01', target11)

            img5_t00 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_05', target00)
            img5_t0 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_05', target0)
            img5_t = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_05', target)
            img5_t1 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_05', target1)
            img5_t11 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_05', target11)

            img6_t00 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_06', target00)
            img6_t0 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_06', target0)
            img6_t = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_06', target)
            img6_t1 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_06', target1)
            img6_t11 = os.path.join('ddad_train_val', scene_dir, 'rgb', 'CAMERA_06', target11)

            # Check valid files
            is_file1_00 = os.path.isfile(os.path.join(main_dir, img1_t00))
            is_file1_0 = os.path.isfile(os.path.join(main_dir, img1_t0))
            is_file1 = os.path.isfile(os.path.join(main_dir, img1_t))
            is_file1_1 = os.path.isfile(os.path.join(main_dir, img1_t1))
            is_file1_11 = os.path.isfile(os.path.join(main_dir, img1_t11))
            is_file1 = is_file1_00 and is_file1_0 and is_file1 and is_file1_1 and is_file1_11

            is_file5_00 = os.path.isfile(os.path.join(main_dir, img5_t00))
            is_file5_0 = os.path.isfile(os.path.join(main_dir, img5_t0))
            is_file5 = os.path.isfile(os.path.join(main_dir, img5_t))
            is_file5_1 = os.path.isfile(os.path.join(main_dir, img5_t1))
            is_file5_11 = os.path.isfile(os.path.join(main_dir, img5_t11))
            is_file5 = is_file5_00 and is_file5_0 and is_file5 and is_file5_1 and is_file5_11

            is_file6_00 = os.path.isfile(os.path.join(main_dir, img6_t00))
            is_file6_0 = os.path.isfile(os.path.join(main_dir, img6_t0))
            is_file6 = os.path.isfile(os.path.join(main_dir, img6_t))
            is_file6_1 = os.path.isfile(os.path.join(main_dir, img6_t1))
            is_file6_11 = os.path.isfile(os.path.join(main_dir, img6_t11))
            is_file6 = is_file6_00 and is_file6_0 and is_file6 and is_file6_1 and is_file6_11

            if not (is_file6 and is_file5 and is_file1):
                continue

            # directories[selector].append([[img1_t0, img1_t, img1_t1],
            #                               [img5_t0, img5_t, img5_t1],
            #                               [img6_t0, img6_t, img6_t1], calibration, None])
            # directories[selector].append([[img1_t0, img1_t, img1_t1],
            #                               [img5_t0, img5_t, img5_t1],
            #                               [img6_t0, img6_t, img6_t1], calibration, None])
            # directories[selector].append([[img1_t0, img1_t, img1_t1],
            #                               [img5_t0, img5_t, img5_t1],
            #                               [img6_t0, img6_t, img6_t1], calibration, None])

            # directories[selector].append([[img1_t00, img1_t, img1_t11],
            #                               [img5_t00, img5_t, img5_t11],
            #                               [img6_t00, img6_t, img6_t11], calibration, None])
            # directories[selector].append([[img1_t00, img1_t, img1_t11],
            #                               [img5_t00, img5_t, img5_t11],
            #                               [img6_t00, img6_t, img6_t11], calibration, None])
            # directories[selector].append([[img1_t00, img1_t, img1_t11],
            #                               [img5_t00, img5_t, img5_t11],
            #                               [img6_t00, img6_t, img6_t11], calibration, None])
            directories[selector].append([[img1_t00, img1_t0, img1_t, img1_t1, img1_t11],
                                          [img5_t00, img5_t0, img5_t, img5_t1, img5_t11],
                                          [img6_t00, img6_t0, img6_t, img6_t1, img6_t11], calibration, None])
            directories[selector].append([[img1_t00, img1_t0, img1_t, img1_t1, img1_t11],
                                          [img5_t00, img5_t0, img5_t, img5_t1, img5_t11],
                                          [img6_t00, img6_t0, img6_t, img6_t1, img6_t11], calibration, None])
            directories[selector].append([[img1_t00, img1_t0, img1_t, img1_t1, img1_t11],
                                          [img5_t00, img5_t0, img5_t, img5_t1, img5_t11],
                                          [img6_t00, img6_t0, img6_t, img6_t1, img6_t11], calibration, None])

    return directories[0], directories[1]


def DDAD(split, **kwargs):
    input_root = kwargs.pop('root')
    resize_crop_transform = kwargs.pop('resize_crop_transform', None)
    resize_crop_transform_pose = kwargs.pop('resize_crop_transform_pose', None)
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    reference_transform = kwargs.pop('reference_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    use_panghw_dataset = kwargs.pop('use_panghw', False)
    use_mspang_dataset = kwargs.pop('use_mspang', False)
    max_pix = kwargs.pop('max_pix', 100)
    fix = kwargs.pop('fix', False)
    is_stereo = kwargs.pop('is_stereo', False)
    is_video = kwargs.pop('is_video', False)
    is_video_ms = kwargs.pop('is_video_ms', False)
    is_stereo_video = kwargs.pop('is_stereo_video', False)

    train_list, test_list = make_dataset(input_root, split)

    # Reads multiple views, grid, image height and width data.
    if is_video_ms and use_panghw_dataset:
        train_dataset = video_ms_ghwListDataset(input_root, train_list, disp=False, of=False,
                                          transform=transform, target_transform=target_transform,
                                          co_transform=co_transform, resize_crop_transform=resize_crop_transform,
                                          resize_crop_transform_pose=resize_crop_transform_pose,
                                          max_pix=max_pix, reference_transform=reference_transform, fix=fix)
        test_dataset = video_ms_ghwListDataset(input_root, test_list, disp=False, of=False,
                                         transform=transform, target_transform=target_transform, fix=fix)
    elif use_panghw_dataset:
        train_dataset = panghwListDataset(input_root, input_root, train_list, disp=False, of=False,
                                          transform=transform, target_transform=target_transform,
                                          co_transform=co_transform, resize_crop_transform=resize_crop_transform,
                                          max_pix=max_pix, reference_transform=reference_transform, fix=fix,
                                          stereo=is_stereo, video=is_video, stereo_video=is_stereo_video)
        test_dataset = panghwListDataset(input_root, input_root, test_list, disp=False, of=False,
                                         transform=transform, target_transform=target_transform, fix=fix,
                                         stereo=is_stereo, video=is_video, stereo_video=is_stereo_video)
    # Reads multiple views at multiple resolutions, and grid data.
    elif use_mspang_dataset:
        train_dataset = mspangListDataset(input_root, input_root, train_list, data_name='Kitti2015', disp=False,
                                          of=False,
                                          transform=transform, target_transform=target_transform,
                                          co_transform=co_transform,
                                          max_pix=max_pix, reference_transform=reference_transform, fix=fix)
        test_dataset = mspangListDataset(input_root, input_root, test_list, data_name='Kitti2015', disp=False, of=False,
                                         transform=transform, target_transform=target_transform, fix=fix)
    else:
        train_dataset = ListDataset(input_root, input_root, train_list, data_name='Kitti2015', disp=False, of=False,
                                    transform=transform, target_transform=target_transform, co_transform=co_transform)
        test_dataset = ListDataset(input_root, input_root, test_list, data_name='Kitti2015', disp=False, of=False,
                                   transform=transform, target_transform=target_transform)
    return train_dataset, test_dataset


def DDAD_list(split, **kwargs):
    input_root = kwargs.pop('root')
    train_list, test_list = make_dataset(input_root, split)
    return train_list, test_list
