import os.path
from shutil import copy2

# Copy only files in kitti_train_files
input_root = '/home/juanluis/Documents/data/Kitti_old'
save_path = '/home/juanluis/Documents/data/Kitti_split'

if not os.path.exists(save_path):
    os.makedirs(save_path)

with open("/home/juanluis/PycharmProjects/Left_Right_estimation/Datasets/kitti_train_files.txt", 'r') as f:
    train_list = list(f.read().splitlines())
    train_list = [[line.split(" "), None] for line in train_list if os.path.isfile(os.path.join(input_root, line.split(" ")[0]))]

print('Copying kitti split')

for pair in train_list:
    pair = pair[0]
    source_l = os.path.join(input_root, pair[0])
    source_r = os.path.join(input_root, pair[1])
    dst_l = os.path.join(save_path, pair[0])
    dst_r = os.path.join(save_path, pair[1])

    if not os.path.exists(dst_l[:-14]):
        os.makedirs(dst_l[:-14])
    if not os.path.exists(dst_r[:-14]):
        os.makedirs(dst_r[:-14])

    copy2(source_l, dst_l)
    copy2(source_r, dst_r)

print('Finished')