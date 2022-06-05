import os.path
from .util import split2list
from .listdataset import ListDataset
from random import shuffle


def Kitti_eigen_val(split, **kwargs):
    input_root = kwargs.pop('root')
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    annotated = kwargs.pop('annotated', True)
    shuffle_test = kwargs.pop('shuffle_test', True)

    label = 'velodyne_raw'
    if annotated:
        label = 'groundtruth'
    # depth_root = os.path.join('/home', 'juanluis','Documents','data', 'Kitti', 'KITTI', 'data_depth')
    # img_root = os.path.join('/home', 'juanluis','Documents','data', 'Kitti', 'KITTI')

    with open("Datasets/kitti_eigen_test.txt", 'r') as f:
        train_list = list(f.read().splitlines())
        train_list = [[line.split(" "),
                       [os.path.join(line.split(" ")[0][0:-29], 'proj_depth', 'velodyne_raw', 'image_02',
                                     line.split(" ")[0][-14:]),
                        os.path.join(line.split(" ")[0][0:-29], 'proj_depth', 'groundtruth', 'image_02',
                                     line.split(" ")[0][-14:])]]
                      for line in train_list if (os.path.isfile(os.path.join(
                input_root, line.split(" ")[0][0:-29], 'proj_depth', label, 'image_02', line.split(" ")[0][-14:]))
                                                 and os.path.isfile(os.path.join(input_root, line.split(" ")[0])))]
        # train_list = [[line.split(" "),
        #         [os.path.join(line.split(" ")[0][0:-29], 'proj_depth', 'velodyne_raw', 'image_02', line.split(" ")[0][-14:]),
        #          os.path.join(line.split(" ")[0][0:-29], 'proj_depth', 'groundtruth', 'image_02', line.split(" ")[0][-14:])]]
        #               for line in train_list if (os.path.isfile(os.path.join(img_root, line.split(" ")[0])))]

    train_list, test_list = split2list(train_list, split)

    train_dataset = ListDataset(input_root, input_root, train_list, data_name='eigen_val', disp=True, of=False,
                                transform=transform, target_transform=target_transform, co_transform=co_transform)
    if shuffle_test:
        shuffle(test_list)

    test_dataset = ListDataset(input_root, input_root, test_list, data_name='eigen_val', disp=True, of=False,
                               transform=transform, target_transform=target_transform)

    return train_dataset, test_dataset
