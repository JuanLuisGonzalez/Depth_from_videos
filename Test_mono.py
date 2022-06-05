# Written by Juan Luis Gonzalez Bello
# Date June, 2022

import argparse
import time
import numpy as np
from imageio import imsave
import matplotlib.pyplot as plt
from PIL import Image
from math import exp

import Datasets
import models
from inverse_warp import m_inverse_warp_grid

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)

parser = argparse.ArgumentParser(description='DVF Pytorch',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Hardware settings
parser.add_argument('-gpu_no', '--gpu_no', default='0')
parser.add_argument('-w', '--workers', metavar='Workers', default=4)
# Dataset settings
# parser.add_argument('-d', '--data', metavar='DIR', default='/raid/20178196', help='path to dataset')
parser.add_argument('-d', '--data', metavar='DIR', default='E:\\Datasets\\', help='path to dataset')
parser.add_argument('-tn', '--tdataName', metavar='Test Data Set Name', default='Kitti_eigen_test_improved',
                    choices=dataset_names)
parser.add_argument('-relbase', '--rel_baselne', default=1, help='Relative baseline of testing dataset')
parser.add_argument('-mdisp', '--max_disp', default=0.3)  # of the training patch W
parser.add_argument('-mindisp', '--min_disp', default=0.005)  # of the training patch W
# Save results settings
parser.add_argument('-b', '--batch_size', metavar='Batch Size', default=1)
parser.add_argument('-eval', '--evaluate', default=True)
parser.add_argument('-save', '--save', default=True)
parser.add_argument('-save_pc', '--save_pc', default=False)
parser.add_argument('-x_c', '--x_c', default=0.0)
parser.add_argument('-y_c', '--y_c', default=0.0)
parser.add_argument('-z_c', '--z_c', default=0.0)
parser.add_argument('-save_input', '--save_input', default=False)
# Other settings
parser.add_argument('-save_gt', '--save_gt', default=False)
parser.add_argument('--sparse', default=False, action='store_true',
                    help='Depth GT is sparse, automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')

parser.add_argument('-scale_factor', '--scale_factor', default=1.0)
parser.add_argument('-focal_factor', '--focal_factor', default=1.0)
parser.add_argument('-pos_shift_x', '--pos_shift_x', default=0.5)
parser.add_argument('-pos_shift_y', '--pos_shift_y', default=0.0)
parser.add_argument('-dt', '--dataset', help='Dataset and training stage directory', default='Kitti')
parser.add_argument('-ts', '--time_stamp', help='Model timestamp', default='06-06-01_17')
parser.add_argument('-m', '--model', help='Model', default='Dispnet_bw_base')
parser.add_argument('-dtl', '--details', help='details',
                    default=',e110es10,b2,lr0.0001/checkpoint_disp.pth.tar')
parser.add_argument('-fpp', '--f_post_process', default=False, help='Post-processing with flipped input')
parser.add_argument('-mspp', '--ms_post_process', default=False, help='Post-processing with multi-scale input')
parser.add_argument('-median', '--median', default=True,
                    help='use median scaling (not needed when training from stereo')

parser.add_argument('-display_masks', '--display_masks', default=False)
parser.add_argument('-mask_scale', '--mask_scale', default=2/3)
parser.add_argument('-var_gamma', '--var_gamma', default=0.03)
parser.add_argument('-nvar_gamma', '--nvar_gamma', metavar='Set to 1 to disable', default=1.0)
parser.add_argument('-no_fix_pass', '--no_fix_pass', metavar='Set to 3 or 4', default=3)


def display_config(save_path):
    settings = ''
    settings = settings + '############################################################\n'
    settings = settings + '# DFV Test         -       Pytorch implementation          #\n'
    settings = settings + '# by Juan Luis Gonzalez   juanluisgb@kaist.ac.kr           #\n'
    settings = settings + '############################################################\n'
    settings = settings + '-------YOUR TRAINING SETTINGS---------\n'
    for arg in vars(args):
        settings = settings + "%15s: %s\n" % (str(arg), str(getattr(args, arg)))
    print(settings)
    # Save config in txt file
    with open(os.path.join(save_path, 'settings.txt'), 'w+') as f:
        f.write(settings)


def main():
    print('-------Testing on gpu ' + args.gpu_no + '-------')

    save_path = os.path.join('Test_Results', args.tdataName, args.model, args.time_stamp)
    if args.f_post_process:
        save_path = save_path + 'fpp'
    if args.ms_post_process:
        save_path = save_path + 'mspp'
    print('=> Saving to {}'.format(save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    display_config(save_path)

    input_transform = transforms.Compose([
        data_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),  # (input - mean) / std
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    target_transform = transforms.Compose([
        data_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0], std=[1]),
    ])

    # Torch Data Set List
    input_path = os.path.join(args.data, args.tdataName)
    [test_dataset, _] = Datasets.__dict__[args.tdataName](split=1,  # all to be tested
                                                          root=input_path,
                                                          disp=True,
                                                          shuffle_test=False,
                                                          transform=input_transform,
                                                          target_transform=target_transform)

    # Torch Data Loader
    args.batch_size = 1  # kitty mixes image sizes!
    args.sparse = True  # disparities are sparse (from lidar)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=False, shuffle=False)

    # create disp model
    model_dir = os.path.join(args.dataset, args.time_stamp, args.model + args.details)
    disp_network_data = torch.load(model_dir)
    disp_model = disp_network_data['m_model']
    print("=> using pre-trained model for disp '{}'".format(disp_model))
    disp_model = models.__dict__[disp_model](disp_network_data).cuda()
    disp_model = torch.nn.DataParallel(disp_model, device_ids=[0]).cuda()
    disp_model.eval()
    model_parameters = utils.get_n_params(disp_model)
    print("=> Number of parameters '{}'".format(model_parameters))
    cudnn.benchmark = True

    # evaluate on validation set
    validate(val_loader, disp_model, save_path, model_parameters)


def validate(val_loader, disp_model, save_path, model_param):
    global args
    batch_time = utils.AverageMeter()
    EPEs = utils.AverageMeter()
    kitti_erros = utils.multiAverageMeter(utils.kitti_error_names)
    D1_all = utils.AverageMeter()

    disp_path = os.path.join(save_path, 'disp')
    if not os.path.exists(disp_path):
        os.makedirs(disp_path)

    input_path = os.path.join(save_path, 'Input im')
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    gt_disp_path = os.path.join(save_path, 'GT')
    if not os.path.exists(gt_disp_path):
        os.makedirs(gt_disp_path)

    pc_path = os.path.join(save_path, 'Point_cloud')
    if not os.path.exists(pc_path):
        os.makedirs(pc_path)

    feats_path = os.path.join(save_path, 'feats')
    if not os.path.exists(feats_path):
        os.makedirs(feats_path)

    # Set the max disp
    right_shift = args.max_disp * args.rel_baselne

    with torch.no_grad():
        for i, (input, target, f_name) in enumerate(val_loader):
            target = target[0].cuda()
            input_left = input[0].cuda()
            input_right = input[1].cuda()
            if args.scale_factor != 1:
                input_left = F.interpolate(input_left, scale_factor=args.scale_factor, mode='bilinear',
                                           align_corners=True)
            B, C, H, W = input_left.shape
            _, _, tH, tW = target.shape

            # Prepare flip grid for post-processing
            i_tetha = torch.zeros(B, 2, 3).cuda()
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            flip_grid = F.affine_grid(i_tetha, [B, C, H, W], align_corners=True)
            flip_grid[:, :, :, 0] = -flip_grid[:, :, :, 0]

            # Prepare identity grid
            i_tetha = torch.zeros(B, 2, 3).cuda()
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            a_grid = F.affine_grid(i_tetha, [B, 3, H, W], align_corners=True)
            in_grid = torch.zeros(B, 2, H, W).type(a_grid.type())
            in_grid[:, 0, :, :] = a_grid[:, :, :, 0]
            in_grid[:, 1, :, :] = a_grid[:, :, :, 1]
            in_grid = in_grid

            # Convert min and max disp to bx1x1 tensors
            max_disp = torch.Tensor([right_shift]).unsqueeze(1).unsqueeze(1).type(input_left.type())
            min_disp = max_disp * args.min_disp / args.max_disp

            intrinsics = torch.zeros(1, 3, 3).type_as(input_left)
            intrinsics[:, 2, 2] = 1
            intrinsics[:, 0, 0] = 718
            intrinsics[:, 1, 1] = 604
            intrinsics[:, 0, 2] = 600
            intrinsics[:, 1, 2] = 192
            intrinsics_inv = torch.inverse(intrinsics)

            # Synthesis
            end = time.time()

            ### Run forward pass ###
            feats = None
            disp = disp_model(input_left, in_grid, min_disp, max_disp, focal_factor=args.focal_factor)
            if isinstance(disp, list):
                disp = disp[0]
            torch.cuda.synchronize()

            if args.display_masks:
                thres = args.var_gamma
                dfactor = args.mask_scale
                no_pass = args.no_fix_pass

                dcurr_frame = F.interpolate(input_left, size=(int(H * dfactor), int(W * dfactor)),
                                            mode='bilinear', align_corners=True)
                din_grid = F.interpolate(in_grid, size=(int(H * dfactor), int(W * dfactor)),
                                         mode='bilinear', align_corners=True)
                curr_frame_x4 = dcurr_frame.repeat(no_pass, 1, 1, 1)
                min_disp_x4 = min_disp.clone().repeat(no_pass, 1, 1)
                max_disp_x4 = max_disp.clone().repeat(no_pass, 1, 1)
                in_grid_shifted = din_grid.clone()
                in_grid_shifted[:, 0, :, :] = in_grid_shifted[:, 0, :, :] + 0.5
                in_grid_shifted_ = din_grid.clone()
                in_grid_shifted_[:, 0, :, :] = in_grid_shifted_[:, 0, :, :] - 0.5
                in_grid_shifted__ = din_grid.clone()
                in_grid_shifted__[:, 1, :, :] = in_grid_shifted_[:, 1, :, :] - 0.25
                if no_pass == 4:
                    in_grid_x4 = torch.cat((din_grid, in_grid_shifted_, in_grid_shifted, in_grid_shifted__), 0)
                else:
                    in_grid_x4 = torch.cat((din_grid, in_grid_shifted_, in_grid_shifted), 0)

                disp_x4x3 = disp_model(curr_frame_x4, in_grid_x4, min_disp_x4, max_disp_x4,
                                       focal_factor=dfactor)

                # For vertical shift, we must take care of change in depth distribution
                _, _, h, w = disp_x4x3.shape
                disp_x4 = disp_x4x3[0:B * no_pass]
                if no_pass == 4:
                    mean_disp = (disp_x4[0:B] + disp_x4[B:2 * B] + disp_x4[2 * B:3 * B]) / 3
                    # Our depth scaling
                    disp_mean, disp_std = F.avg_pool2d(mean_disp, kernel_size=(h, w)), \
                                          torch.var(mean_disp.view(B, 1 * h * w), dim=1).view(B, 1, 1, 1) ** (1 / 2)
                    disp_v = (disp_x4[3 * B::] - F.avg_pool2d(disp_x4[3 * B::], kernel_size=(h, w))) / \
                             torch.var(disp_x4[3 * B::].view(B, 1 * h * w), dim=1).view(B, 1, 1, 1) ** (1 / 2)
                    # Scale v disp
                    disp_v = disp_v * disp_std + disp_mean

                if no_pass == 4:
                    disp_volume_c = torch.cat((disp_x4[0:B], disp_x4[B:2 * B],
                                               disp_x4[2 * B:3 * B], disp_v), 1)
                else:
                    disp_volume_c = torch.cat((disp_x4[0:B], disp_x4[B:2 * B],
                                               disp_x4[2 * B:3 * B]), 1)

                depth_volume = 1 / disp_volume_c
                mean_depth = torch.mean(depth_volume, dim=1, keepdim=True)
                var_depth = torch.var(depth_volume, dim=1, keepdim=True)
                dispersion = var_depth / (mean_depth ** 2)
                stat_mask = (dispersion < thres).type_as(disp_x4)
                stat_mask = F.interpolate(stat_mask, size=(H, W))  # back to target resolution
                feats = [255 * stat_mask]

            # Some post-processing techniques
            if args.f_post_process:
                flip_disp = disp_model(F.grid_sample(input_left, flip_grid, align_corners=True),
                                       in_grid, min_disp, max_disp)
                flip_disp = F.grid_sample(flip_disp, flip_grid, align_corners=True)
                disp = (disp + flip_disp) / 2
            elif args.ms_post_process:
                disp = ms_ppg(input_left, in_grid, disp_model, flip_grid, disp, min_disp, max_disp)

            # measure elapsed time (do not measure first image)
            if i > 0:
                batch_time.update(time.time() - end, 1)

            if disp.shape[2] != tH:
                disp = F.interpolate(disp, size=(tH, tW), mode='bilinear', align_corners=True)

            # Save outputs to disk
            if args.save:
                disparity = disp.squeeze().cpu().numpy()

                # Other colormap for displaying disparity
                # disparity = Kitti_colormap.kitti_colormap(disparity, 100)
                # plt.imsave(os.path.join(disp_path, '{:010d}.png'.format(i)), disparity)

                disparity = 256 * np.clip(disparity / (np.percentile(disparity, 99) + 1e-6), 0, 1)
                plt.imsave(os.path.join(disp_path, '{:010d}.png'.format(i)), np.rint(disparity).astype(np.int32),
                           cmap='plasma', vmin=0, vmax=256)

                if args.save_gt:
                    disparity = target.squeeze().cpu().numpy()

                    # Other colormap for displaying disparity
                    # disparity = Kitti_colormap.kitti_colormap(disparity, 100)
                    # plt.imsave(os.path.join(gt_disp_path, '{:010d}.png'.format(i)), disparity)

                    disparity = 256 * np.clip(disparity / (np.percentile(disparity, 95) + 1e-6), 0, 1)
                    disparity = 256 * np.clip(disparity / (np.percentile(disparity, 95) + 1e-6), 0, 1)
                    plt.imsave(os.path.join(gt_disp_path, '{:010d}.png'.format(i)), np.rint(disparity).astype(np.int32),
                               cmap='plasma', vmin=0, vmax=256)

                if args.save_pc:
                    # equalize tone
                    m_rgb = torch.ones((B, C, 1, 1)).cuda()
                    m_rgb[:, 0, :, :] = 0.411 * m_rgb[:, 0, :, :]
                    m_rgb[:, 1, :, :] = 0.432 * m_rgb[:, 1, :, :]
                    m_rgb[:, 2, :, :] = 0.45 * m_rgb[:, 2, :, :]
                    point_cloud = utils.get_point_cloud((input_left + m_rgb) * 255, disp)
                    utils.save_point_cloud(point_cloud.squeeze(0).cpu().numpy(),
                                           os.path.join(pc_path, '{:010d}.ply'.format(i)))

                if args.save_input:
                    denormalize = np.array([0.411, 0.432, 0.45])
                    denormalize = denormalize[:, np.newaxis, np.newaxis]
                    p_im = input_left.squeeze().cpu().numpy() + denormalize
                    im = Image.fromarray(np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8))
                    im.save(os.path.join(input_path, '{:010d}.png'.format(i)))

                # save features per channel as grayscale images
                if feats is not None:
                    for layer in range(len(feats)):
                        _, nc, _, _ = feats[layer].shape
                        if nc == 3:
                            im = feats[layer].squeeze().cpu().numpy()
                            im[im < 0] = 0
                            im[im > 255] = 255
                            im = Image.fromarray(np.rint(im.transpose(1, 2, 0)).astype(np.uint8))
                            im.save(os.path.join(feats_path, '{:010d}_l{}.png'.format(i, layer)))
                        else:
                            for inc in range(nc):
                                feature = torch.abs(feats[layer][:, inc, :, :]).squeeze().cpu().numpy()
                                feature[feature < 0] = 0
                                feature[feature > 255] = 255
                                imsave(os.path.join(feats_path, '{:010d}_l{}_c{}.png'.format(i, layer, inc)),
                                       np.rint(feature).astype(np.uint8))

            if args.evaluate:
                # Measure and Record kitti metrics
                target_disp = target.squeeze(1).cpu().numpy()
                pred_disp = disp.squeeze(1).cpu().numpy()
                if args.tdataName == 'Kitti_eigen_test_improved' or \
                        args.tdataName == 'Kitti_eigen_test_original' or \
                        args.tdataName == 'Kitti_eigen_test_original_png':
                    target_depth, pred_depth = utils.disps_to_depths_kitti(target_disp, pred_disp)
                    kitti_erros.update(
                        utils.compute_kitti_errors(target_depth[0], pred_depth[0], use_median=args.median),
                        target.size(0))
                if args.tdataName == 'Kitti2015':
                    # D1all is usually measured in stereo disp estimation
                    # D1all = utils.compute_d1_all(target, disp)
                    # D1_all.update(D1all, target.size(0))

                    EPE = realEPE(disp, target, sparse=True)
                    EPEs.update(EPE.detach(), target.size(0))
                    target_depth, pred_depth = utils.disps_to_depths_kitti2015(target_disp, pred_disp)
                    kitti_erros.update(
                        utils.compute_kitti_errors(target_depth[0], pred_depth[0], use_median=args.median),
                        target.size(0))

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t Time {2}\t a1 {3:.4f}'.format(i, len(val_loader), batch_time,
                                                                       kitti_erros.avg[4]))

    # Save erros and number of parameters in txt file
    with open(os.path.join(save_path, 'errors.txt'), 'w+') as f:
        f.write('\nNumber of parameters {}\n'.format(model_param))
        f.write('\nEPE {}\n'.format(EPEs.avg))
        f.write('\nKitti metrics: \n{}\n'.format(kitti_erros))

    if args.evaluate:
        print('* EPE: {0}'.format(EPEs.avg))
        print('* D1: {0}'.format(D1_all.avg))
        print(kitti_erros)


def ms_ppg(input_view, in_grid, disp_model, flip_grid, disp, min_disp, max_pix):
    B, C, H, W = input_view.shape

    up_fac = 0.75
    upscaled = F.interpolate(F.grid_sample(input_view, flip_grid, align_corners=True), scale_factor=up_fac,
                             mode='bilinear', align_corners=True)
    shift_grid = in_grid.clone()
    shift_grid[:, 0, :, :] = shift_grid[:, 0, :, :] + 0
    shift_grid[:, 1, :, :] = shift_grid[:, 1, :, :] + 0
    ups_grid = F.interpolate(shift_grid, scale_factor=up_fac, mode='bilinear', align_corners=True)
    dwn_flip_disp = disp_model(upscaled, ups_grid, min_disp, max_pix, focal_factor=1.0)
    dwn_flip_disp = dwn_flip_disp[0]
    # med_scale_fac = torch.median(disp.detach().view([B, H * W]), dim=1)[0].unsqueeze(1).unsqueeze(2) \
    #                     .unsqueeze(3) / \
    #                 torch.median(dwn_flip_disp.view([B, dwn_flip_disp.shape[2] * dwn_flip_disp.shape[3]]), dim=1)[0].unsqueeze(1).unsqueeze(
    #                     2).unsqueeze(3)
    med_scale_fac = 1/up_fac
    dwn_flip_disp = med_scale_fac * F.interpolate(dwn_flip_disp, size=(H, W), mode='nearest')  # , align_corners=True)
    dwn_flip_disp = F.grid_sample(dwn_flip_disp, flip_grid, align_corners=True)

    up_fac = 1.25
    upscaled = F.interpolate(F.grid_sample(input_view, flip_grid, align_corners=True), scale_factor=up_fac,
                             mode='bilinear', align_corners=True)
    # upscaled = F.interpolate(input_view, scale_factor=up_fac, mode='bilinear', align_corners=True)
    shift_grid = in_grid.clone()
    shift_grid[:, 0, :, :] = shift_grid[:, 0, :, :] + 0
    shift_grid[:, 1, :, :] = shift_grid[:, 1, :, :] + 0
    ups_grid = F.interpolate(shift_grid, scale_factor=up_fac, mode='bilinear', align_corners=True)
    up_flip_disp = disp_model(upscaled, ups_grid, min_disp, max_pix, ret_disp=True, focal_factor=1.0)
    up_flip_disp = up_flip_disp[0]
    # med_scale_fac = torch.median(disp.detach().view([B, H * W]), dim=1)[0].unsqueeze(1).unsqueeze(2) \
    #                     .unsqueeze(3) / \
    #                 torch.median(up_flip_disp.view([B, up_flip_disp.shape[2] * up_flip_disp.shape[3]]), dim=1)[0].unsqueeze(1).unsqueeze(
    #                     2).unsqueeze(3)
    med_scale_fac = 1/up_fac
    up_flip_disp = med_scale_fac * F.interpolate(up_flip_disp, size=(H, W), mode='nearest')  # , align_corners=True)
    up_flip_disp = F.grid_sample(up_flip_disp, flip_grid, align_corners=True)

    norm = disp / F.max_pool2d(disp, kernel_size=(H, W))#(np.percentile(disp.detach().cpu().numpy(), 95) + 1e-6)
    norm[norm > 0.99] = 1

    return ((1 - norm**2) * up_flip_disp + norm * dwn_flip_disp + disp) / (1 + 1 - norm**2 + norm)


if __name__ == '__main__':
    import os

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no

    import torch
    import torch.utils.data
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torchvision.transforms as transforms
    import torch.nn.functional as F

    import myUtils as utils
    import data_transforms
    from loss_functions import realEPE
    # import Kitti_colormap

    main()
