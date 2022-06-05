# Written by Juan Luis and Jaeho @ VICLAB KAIST
# Date June, 2022

import argparse
import datetime
import time
import numpy as np
import random
from PIL import Image
from imageio import imsave
import copy

import Datasets
import models

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)

parser = argparse.ArgumentParser(description='Learning depth from videos in pytorch',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset settings
# parser.add_argument('-d', '--data', metavar='DIR', default='/raid/20178196', help='path to dataset')
parser.add_argument('-d', '--data', metavar='DIR', default='E:\\Datasets\\', help='path to dataset')
parser.add_argument('-n0', '--dataName0', metavar='Data Set Name 0', default='Kitti', choices=dataset_names)
parser.add_argument('-eigen', '--eigen_split', default=True)
parser.add_argument('-tn', '--tdataName', metavar='Test Data Set Name', default='Kitti2015', choices=dataset_names)
parser.add_argument('-maxd', '--max_disp', default=0.3)
parser.add_argument('-mind', '--min_disp', default=0.005)
parser.add_argument('-rtr', '--ref_trans', default=False)
parser.add_argument('-fix_order', '--fix_order', default=False)
# Hardware settings
parser.add_argument('-gpu_no', '--gpu_no', default='0')
parser.add_argument('-w', '--workers', metavar='Workers', default=4)
# Network settings
parser.add_argument('-mm', '--m_model', metavar='Mono Model', default='Dispnet_bw_base', choices=model_names)
parser.add_argument('-pm', '--pose_model', metavar='Pose Model', default='PoseNet_dr', choices=model_names)
parser.add_argument('-depth_scale', '--depth_scale', default='Direct')
parser.add_argument('-ignore_top', '--ignore_top', default=True)
parser.add_argument('-train_pose', '--train_pose', default=True)
parser.add_argument('-mask_scale', '--mask_scale', default=2/3)
parser.add_argument('-var_gamma', '--var_gamma', default=0.03)
parser.add_argument('-no_fix_pass', '--no_fix_pass', metavar='Set to 3 or 4', default=3)
# Loss weights
parser.add_argument('-perc', '--a_p', default=0.0)
parser.add_argument('-disp_smooth', '--a_disp_sm', type=float, default=0.01)
parser.add_argument('-disp_smooth_sch', '--disp_smooth_sch', default=6)
# Training details
parser.add_argument('-rand_down', '--rand_down', default=0.5)
parser.add_argument('-rand_up', '--rand_up', default=1.5)
parser.add_argument('-ch', '--crop_height', metavar='Batch crop H Size', default=384 // 2)
parser.add_argument('-cw', '--crop_width', metavar='Batch crop W Size', default=1280 // 2)
parser.add_argument('-chp', '--crop_height_pose', metavar='Batch crop H Size', default=384 // 2)
parser.add_argument('-cwp', '--crop_width_pose', metavar='Batch crop W Size', default=1280 // 2)
parser.add_argument('-s', '--split_value', metavar='Model', default=1.0)
# Optimizer settings
parser.add_argument('-b', '--batch_size', metavar='Batch Size', default=8)
parser.add_argument('-tbs', '--tbatch_size', metavar='val Batch Size', default=1)
parser.add_argument('--epochs', default=110, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-op', '--optimizer', metavar='Optimizer', default='adam')
parser.add_argument('--lr', metavar='learning Rate', default=0.0001)
parser.add_argument('--beta', metavar='BETA', type=float, help='Beta parameter for adam', default=0.999)
parser.add_argument('--momentum', default=0.5, type=float, metavar='Momentum', help='Momentum for Optimizer')
parser.add_argument('--milestones', default=[60, 80, 100], metavar='N', nargs='*',
                    help='epochs at which learning rate is divided by 2')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0.0, type=float, metavar='B', help='bias decay')
# Other settings
parser.add_argument('--sparse', default=True, action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                         'automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency')
# Load checkpoints
parser.add_argument('--start-epoch', default=5, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrained_disp', dest='pretrained_disp', default=None, help='path to pre-trained model')
# parser.add_argument('--pretrained_disp', dest='pretrained_disp',
#                     default='/checkpoint_disp.pth.tar',
#                     help='directory of run')
parser.add_argument('--pretrained_pose', dest='pretrained_pose', default=None, help='path to pre-trained model')
# parser.add_argument('--pretrained_pose', dest='pretrained_pose',
#                     default='pret//checkpoint_pose.pth.tar',
#                     help='directory of run')
parser.add_argument('--fixed_disp', dest='fixed_disp', default=None, help='path to model trained w/o spimo mask')
# parser.add_argument('--fixed_disp', dest='fixed_disp',
#                     default='pret//checkpoint_disp.pth.tar',
#                     help='directory of run')
parser.add_argument('-dev_note', '--dev_note', default='')


def display_config(save_path):
    settings = ''
    settings = settings + '############################################################\n'
    settings = settings + '# Learning depth from videos (using backward warping nets) #\n'
    settings = settings + '# by Juan Luis Gonzalez and Moon Jaeho                     #\n'
    settings = settings + '############################################################\n'
    settings = settings + '-------YOUR TRAINING SETTINGS---------\n'
    for arg in vars(args):
        settings = settings + "%15s: %s\n" % (str(arg), str(getattr(args, arg)))
    print(settings)
    # Save config in txt file
    with open(os.path.join(save_path, 'settings.txt'), 'w+') as f:
        f.write(settings)


def main():
    print('-------Training on gpu ' + args.gpu_no + '-------')
    best_a_1 = -1

    # Set up save path for model
    save_path = '{},e{}es{},b{},lr{}'.format(
        args.m_model,
        args.epochs,
        str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr,
    )
    timestamp = datetime.datetime.now().strftime("%m-%d-%H_%M")
    save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.dataName0, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    display_config(save_path)
    print('=> will save everything to {}'.format(save_path))

    # Set tensorboard logs
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    # Set data augmentations that impact left and right views for both datasets
    resize_crop_transform = data_transforms.RandomResizeCrop_hw((args.crop_height, args.crop_width),
                                                                down=args.rand_down, up=args.rand_up)
    resize_crop_transform_pose = data_transforms.Resize((args.crop_height_pose, args.crop_width_pose))

    co_transform = data_transforms.Compose([
        data_transforms.RandomHorizontalFlip(is_video=True),
        # flow_transforms.RandomVerticalFlip(stereo_targets=False), # vertical flip makes training more difficult
        data_transforms.RandomGamma(min=0.8, max=1.2),
        data_transforms.RandomBrightness(min=0.5, max=2.0),
        data_transforms.RandomCBrightness(min=0.8, max=1.0),
    ])

    # Randomly degrade network input view
    if args.ref_trans:
        reference_transform = transforms.Compose([
            data_transforms.RandomDownUp(max_down=2)
        ])
    else:
        reference_transform = None

    # Convert to torch tensor and normalize
    input_transform = transforms.Compose([
        data_transforms.ArrayToIntTensor(),
    ])

    # GT disparity is not normalized (given in pixels for the KITTI2015)
    target_transform = transforms.Compose([
        data_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0], std=[1]),
    ])

    # Get data set lists for each dataset
    input_path = os.path.join(args.data, args.dataName0)
    [train_dataset0, _] = Datasets.__dict__[args.dataName0] \
        (args.split_value,
         root=input_path,
         is_eigen=args.eigen_split,
         use_panghw=True,
         is_video_ms=True,
         transform=input_transform,
         target_transform=target_transform,
         resize_crop_transform=resize_crop_transform,
         resize_crop_transform_pose=resize_crop_transform_pose,
         co_transform=co_transform,
         reference_transform=reference_transform,
         max_pix=args.max_disp,
         fix=args.fix_order)

    input_path = os.path.join(args.data, args.tdataName)
    [_, test_dataset] = Datasets.__dict__[args.tdataName](split=0,  # all to be tested
                                                          root=input_path,
                                                          disp=True,
                                                          of=False,
                                                          shuffle_test=False,
                                                          transform=input_transform,
                                                          target_transform=target_transform)

    # Create Torch Data Loaders
    train_loader0 = torch.utils.data.DataLoader(train_dataset0, batch_size=args.batch_size,
                                                num_workers=args.workers,
                                                pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.tbatch_size, num_workers=args.workers,
                                             pin_memory=False, shuffle=False)

    # Create models
    fixed_model = None
    if args.fixed_disp:
        fixed_network_data = torch.load(args.fixed_disp)
        fixed_model = fixed_network_data['m_model']
        print("=> using pre-trained model '{}'".format(fixed_model))
        fixed_model = models.__dict__[fixed_model](fixed_network_data).cuda()
        fixed_model = torch.nn.DataParallel(fixed_model).cuda()
        print("=> Number of parameters fixed-model '{}'".format(utils.get_n_params(fixed_model)))
        fixed_model.eval()
        for param in fixed_model.module.parameters():
            param.requires_grad = False

    # Create models
    if args.pretrained_disp:
        network_data = torch.load(args.pretrained_disp)
        args.m_model = network_data['m_model']
        print("=> using pre-trained model '{}'".format(args.m_model))
    else:
        network_data = None
        print("=> creating m model '{}'".format(args.m_model))
    m_model = models.__dict__[args.m_model](network_data).cuda()
    m_model = torch.nn.DataParallel(m_model).cuda()
    cudnn.benchmark = True
    print("=> Number of parameters m-model '{}'".format(utils.get_n_params(m_model)))

    if args.pretrained_pose:
        network_data = torch.load(args.pretrained_pose)
        args.pose_model = network_data['pose_model']
        print("=> using pre-trained pose model '{}'".format(args.pose_model))
    else:
        network_data = None
        print("=> creating pose model '{}'".format(args.pose_model))
    pose_model = models.__dict__[args.pose_model](network_data).cuda()
    pose_model = torch.nn.DataParallel(pose_model).cuda()
    print("=> Number of parameters pose-model '{}'".format(utils.get_n_params(pose_model)))

    # Create optimizer
    print('Setting {} Optimizer'.format(args.optimizer))
    param_groups = [{'params': pose_model.module.bias_parameters(), 'weight_decay': args.bias_decay, 'lr': args.lr},
                    {'params': pose_model.module.weight_parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                    {'params': m_model.module.bias_parameters(), 'weight_decay': args.bias_decay, 'lr': args.lr},
                    {'params': m_model.module.weight_parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}]
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=param_groups, betas=(args.momentum, args.beta))
    g_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    # If we are resuming training from previous run, update the scheduler
    # It is supposed there is a function for this, but it didnt work as expected when I tried it
    for epoch in range(args.start_epoch):
        g_scheduler.step()

    # Folder for debugging
    deb_path = os.path.join(save_path, 'Debug')
    if not os.path.exists(deb_path):
        os.makedirs(deb_path)

    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch
        print('Learning rate {}'.format(g_scheduler.get_last_lr()))
        train_loss = train(train_loader0, m_model, fixed_model, pose_model, optimizer, epoch, deb_path)
        train_writer.add_scalar('train_loss', train_loss, epoch)

        # Update learning rate
        g_scheduler.step()

        # Evaluate on validation set
        a_1 = validate(val_loader, m_model, epoch, output_writers)
        test_writer.add_scalar('Mean a_1', a_1, epoch)

        # Check if mean a1 acc is best and save model (best or checkpoint)
        if best_a_1 < 0:
            best_a_1 = a_1
        is_best = a_1 > best_a_1
        best_a_1 = max(a_1, best_a_1)

        utils.save_checkpoint_disp({
            'epoch': epoch + 1,
            'm_model': args.m_model,
            'state_dict': m_model.module.state_dict(),
            'best_a_1': best_a_1,
        }, is_best, save_path)

        utils.save_checkpoint_pose({
            'epoch': epoch + 1,
            'pose_model': args.pose_model,
            'state_dict': pose_model.module.state_dict(),
            'best_a_1': best_a_1,
        }, is_best, save_path)


def train(train_loader, m_model, fixed_model, pose_model, optimizer, epoch, deb_path):
    global args
    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    rec_losses = utils.AverageMeter()
    losses = utils.AverageMeter()

    # switch to train mode
    m_model.train()

    end = time.time()
    for i, input_data0 in enumerate(train_loader):
        # Read training data (cpu -> gpu int transfer is faster than float)
        full_prev_frame = input_data0[0][0].cuda().float()
        full_curr_frame = input_data0[0][1].cuda().float()
        full_next_frame = input_data0[0][2].cuda().float()
        prev_frame = input_data0[1][0].cuda().float()
        curr_frame = input_data0[1][1].cuda().float()
        next_frame = input_data0[1][2].cuda().float()

        # Convert to cuda, float, and normalize
        B, C, H, W = curr_frame.shape
        mean_shift = torch.zeros(B, C, 1, 1).type(curr_frame.type())
        mean_shift[:, 0, :, :] = 0.411
        mean_shift[:, 1, :, :] = 0.432
        mean_shift[:, 2, :, :] = 0.45
        full_prev_frame = full_prev_frame / 255 - mean_shift
        full_curr_frame = full_curr_frame / 255 - mean_shift
        full_next_frame = full_next_frame / 255 - mean_shift
        prev_frame = prev_frame / 255 - mean_shift
        curr_frame = curr_frame / 255 - mean_shift
        next_frame = next_frame / 255 - mean_shift

        full_ref_frames = [full_prev_frame, full_next_frame]
        ref_frames = [prev_frame, next_frame]
        intrinsics = input_data0[2].cuda()
        intrinsics_inv = torch.inverse(intrinsics)
        in_grid = input_data0[3].cuda()
        max_disp = input_data0[4].unsqueeze(1).unsqueeze(1).type(curr_frame.type())
        max_disp = torch.abs(max_disp)
        img_h = input_data0[5].type(curr_frame.type()).view(B, 1, 1, 1)
        img_w = input_data0[6].type(curr_frame.type()).view(B, 1, 1, 1)
        ign_mask_flag = input_data0[9].type(curr_frame.type())
        random_scale = input_data0[7].type_as(curr_frame).view(B, 1, 1)

        # Get the appropiate min disp
        min_disp = max_disp * args.min_disp / args.max_disp

        # measure data loading/preparation time
        data_time.update(time.time() - end)

        #  Avoid car hoods in lateral views (used for DDAD only)
        ign_mask_flag = ign_mask_flag.view(B, 1, 1, 1)

        # Reset gradients
        optimizer.zero_grad()

        # Compute Camera extrinsics and intrinsics
        if args.train_pose:
            camera_pose = pose_model(full_curr_frame, full_ref_frames)
        else:
            with torch.no_grad():
                camera_pose = pose_model(full_curr_frame, full_ref_frames)
                camera_pose = camera_pose.detach()

        stat_mask_c = None
        proj_views = None
        disp = None
        # Try to kickstart pose net in epoch 0 with global prior (naive disparity assumption)
        if epoch < 5 and args.train_pose:
            ones = torch.ones(B, 1, H, W).type(curr_frame.type())

            # For DDAD, ignore car hoods in lateral views
            ign_mask = (in_grid[:, 1, :, :].unsqueeze(1).clone() > (0.8 * 2 - 1)).type_as(curr_frame)
            ign_mask = ones - ign_mask * ign_mask_flag

            # Build naive disp for kick-starting pose net
            disp = in_grid[:, 1, :, :]
            disp[disp < 0] = 0
            disp = (disp * (max_disp - min_disp) + min_disp).unsqueeze(1)

            photo_warp_loss, proj_views = pose_loss(curr_frame, ref_frames, intrinsics, intrinsics_inv,
                                                1 / disp, ign_mask, -camera_pose, rotation_mode='euler',
                                                padding_mode='zeros')
            loss = photo_warp_loss
        else:
            # Run forward pass once, but generate two synthetic views
            this_camera_pose = torch.cat((camera_pose[:, 0, :].detach(), camera_pose[:, 1, :].detach()), 0)

            # Scale disp proportional to img scale
            if args.depth_scale == 'Direct':
                this_camera_pose[:,0:3] = this_camera_pose[:,0:3] / \
                                          torch.cat((random_scale, random_scale), 0).view(2*B, 1)
            # Scale disp inverse proportional to img scale
            if args.depth_scale == 'Inverse':
                this_camera_pose[:, 0:3] = this_camera_pose[:, 0:3] * \
                                           torch.cat((random_scale, random_scale), 0).view(2 * B, 1)

            # Get SPIMO masks
            if fixed_model is not None:
                stat_mask_c, stat_mask_p, stat_mask_n = get_spimo_masks(fixed_model, curr_frame, prev_frame,
                                                                        next_frame, in_grid, min_disp, max_disp,
                                                                        intrinsics)

            ########### Forward pass #############
            disp = m_model(curr_frame, in_grid, min_disp, max_disp, intrinsics=intrinsics)
            depth = 1 / disp.squeeze(1)

            # Warp next and prev SPIMO masks to current frame and combine with center mask
            if fixed_model is not None:
                stat_mask_p, stat_mask_n = project_and_fuse_spimo(depth, stat_mask_c, stat_mask_p, stat_mask_n,
                                                                  in_grid, this_camera_pose, intrinsics, intrinsics_inv)
            else:
                stat_mask_p, stat_mask_n = 1, 1

            #  Compute smooth losses
            disp_sm_loss = 0
            if args.a_disp_sm > 0 and epoch >= args.disp_smooth_sch:
                disp_sm_loss = smoothness(curr_frame, disp, gamma=2)

            # Get photometric warping error
            photo_warp_loss, proj_views = pose_loss_masked(curr_frame, ref_frames, intrinsics, intrinsics_inv,
                                                           depth, [stat_mask_p, stat_mask_n], -camera_pose,
                                                           rotation_mode='euler', padding_mode='zeros', alpha=0)
            rec_losses.update(photo_warp_loss.detach().cpu(), args.batch_size)

            # Aggregate all loss terms
            loss = photo_warp_loss + args.a_disp_sm * disp_sm_loss

        if loss == 0:
            continue
        losses.update(loss.detach().cpu(), args.batch_size)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log usefull information
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] Time {3}  Data {4}  Loss {5} PhotoLoss {6}'
                  .format(epoch, i, epoch_size, batch_time, data_time, losses, rec_losses))

            # Print intermediate outputs
            with torch.no_grad():
                denormalize = np.array([0.411, 0.432, 0.45])
                denormalize = denormalize[:, np.newaxis, np.newaxis]
                idx = 0  # left_view.shape[0] - 1

                p_im = curr_frame[idx].detach().squeeze().cpu().numpy() + denormalize
                im = Image.fromarray(np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8))
                im.save(os.path.join(deb_path, 'e{}_t_view.png'.format(epoch)))
                p_im = prev_frame[idx].detach().squeeze().cpu().numpy() + denormalize
                im = Image.fromarray(np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8))
                im.save(os.path.join(deb_path, 'e{}_t0_view.png'.format(epoch)))
                p_im = next_frame[idx].detach().squeeze().cpu().numpy() + denormalize
                im = Image.fromarray(np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8))
                im.save(os.path.join(deb_path, 'e{}_t1_view.png'.format(epoch)))

                p_im = full_curr_frame[idx].detach().squeeze().cpu().numpy() + denormalize
                im = Image.fromarray(np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8))
                im.save(os.path.join(deb_path, 'e{}_t_fullview.png'.format(epoch)))

                with open(os.path.join(deb_path, 'e{}_poses.txt'.format(epoch)), 'w+') as f:
                    f.write('\nCamera pose prev {}\n'.format(camera_pose[idx, 0, :].squeeze().cpu().numpy()))
                    f.write('\nCamera pose next {}\n'.format(camera_pose[idx, 1, :].squeeze().cpu().numpy()))

                if proj_views is not None:
                    for n_projv in range(len(proj_views)):
                        p_im = proj_views[n_projv][idx].detach().squeeze().cpu().numpy() + denormalize
                        im = Image.fromarray(np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8))
                        im.save(os.path.join(deb_path, 'e{}_proj_{}_view.png'.format(epoch, n_projv)))

                if disp is not None:
                    the_max_disp = torch.max_pool2d(disp, kernel_size=(H, W))
                    ndisp = disp / the_max_disp
                    p_im = ndisp[idx].detach().squeeze().cpu().numpy()
                    im = Image.fromarray(np.rint(255 * p_im).astype(np.uint8))
                    im.save(os.path.join(deb_path, 'e{}_disp.png'.format(epoch)))

                if stat_mask_c is not None:
                    p_im = stat_mask_c[idx].detach().squeeze().cpu().numpy()
                    im = Image.fromarray(np.rint(255 * p_im).astype(np.uint8))
                    im.save(os.path.join(deb_path, 'e{}_stat_mask_c.png'.format(epoch)))

                    p_im = stat_mask_p[idx].detach().squeeze().cpu().numpy()
                    im = Image.fromarray(np.rint(255 * p_im).astype(np.uint8))
                    im.save(os.path.join(deb_path, 'e{}_stat_mask_p.png'.format(epoch)))

                    p_im = stat_mask_n[idx].detach().squeeze().cpu().numpy()
                    im = Image.fromarray(np.rint(255 * p_im).astype(np.uint8))
                    im.save(os.path.join(deb_path, 'e{}_stat_mask_n.png'.format(epoch)))

        if i >= epoch_size:
            break

    return losses.avg


def get_spimo_masks(fixed_model, curr_frame, prev_frame, next_frame, in_grid, min_disp, max_disp, intrinsics):
    B, C, H, W = curr_frame.shape

    with torch.no_grad():
        thres = args.var_gamma  # variance threshold
        dfactor = args.mask_scale
        no_pass = args.no_fix_pass

        dcurr_frame = F.interpolate(curr_frame, size=(int(H * dfactor), int(W * dfactor)),
                                    mode='bilinear', align_corners=True)
        dprev_frame = F.interpolate(prev_frame, size=(int(H * dfactor), int(W * dfactor)),
                                    mode='bilinear', align_corners=True)
        dnext_frame = F.interpolate(next_frame, size=(int(H * dfactor), int(W * dfactor)),
                                    mode='bilinear', align_corners=True)
        din_grid = F.interpolate(in_grid, size=(int(H * dfactor), int(W * dfactor)),
                                 mode='bilinear', align_corners=True)
        curr_frame_x4 = torch.cat((dcurr_frame.repeat(no_pass, 1, 1, 1),
                                   dprev_frame.repeat(no_pass, 1, 1, 1),
                                   dnext_frame.repeat(no_pass, 1, 1, 1)), 0)
        min_disp_x4 = min_disp.clone().repeat(no_pass * 3, 1, 1)
        max_disp_x4 = max_disp.clone().repeat(no_pass * 3, 1, 1)
        intrinsics_x4 = intrinsics.clone().repeat(no_pass * 3, 1, 1)
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
        in_grid_x4 = in_grid_x4.repeat(3, 1, 1, 1)

        disp_x4x3 = fixed_model(curr_frame_x4, in_grid_x4, min_disp_x4, max_disp_x4,
                                focal_factor=dfactor, intrinsics=intrinsics_x4)
        disp_x4x3 = disp_x4x3[0]

        # Get mask at current frame
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
        disp_volume_c = F.interpolate(disp_volume_c, size=(H, W))  # back to target resolution

        # Get mask at prev frame
        # For vertical shift, we must take care of change in depth distribution
        disp_x4 = disp_x4x3[B * no_pass:B * no_pass * 2]
        if no_pass == 4:
            mean_disp = (disp_x4[0:B] + disp_x4[B:2 * B] + disp_x4[2 * B:3 * B]) / 3
            disp_mean, disp_std = F.avg_pool2d(mean_disp, kernel_size=(h, w)), \
                                  torch.var(mean_disp.view(B, 1 * h * w), dim=1).view(B, 1, 1, 1) ** (1 / 2)
            disp_v = (disp_x4[3 * B::] - F.avg_pool2d(disp_x4[3 * B::], kernel_size=(h, w))) / \
                     torch.var(disp_x4[3 * B::].view(B, 1 * h * w), dim=1).view(B, 1, 1, 1) ** (1 / 2)
            disp_v = disp_v * disp_std + disp_mean
            # Get potentially dynamic object masks (exclude top of image)
            disp_volume_p = torch.cat((disp_x4[0:B], disp_x4[B:2 * B],
                                       disp_x4[2 * B:3 * B], disp_v), 1)
        else:
            disp_volume_p = torch.cat((disp_x4[0:B], disp_x4[B:2 * B], disp_x4[2 * B:3 * B]), 1)
        disp_volume_p = F.interpolate(disp_volume_p, size=(H, W))  # back to target resolution

        # Get mask at next frame
        # For vertical shift, we must take care of change in depth distribution
        disp_x4 = disp_x4x3[B * no_pass * 2::]
        if no_pass == 4:
            mean_disp = (disp_x4[0:B] + disp_x4[B:2 * B] + disp_x4[2 * B:3 * B]) / 3
            disp_mean, disp_std = F.avg_pool2d(mean_disp, kernel_size=(h, w)), \
                                  torch.var(mean_disp.view(B, 1 * h * w), dim=1).view(B, 1, 1, 1) ** (1 / 2)
            disp_v = (disp_x4[3 * B::] - F.avg_pool2d(disp_x4[3 * B::], kernel_size=(h, w))) / \
                     torch.var(disp_x4[3 * B::].view(B, 1 * h * w), dim=1).view(B, 1, 1, 1) ** (1 / 2)
            disp_v = disp_v * disp_std + disp_mean
            # Get potentially dynamic object masks (exclude top of image)
            disp_volume_n = torch.cat((disp_x4[0:B], disp_x4[B:2 * B],
                                       disp_x4[2 * B:3 * B], disp_v), 1)
        else:
            disp_volume_n = torch.cat((disp_x4[0:B], disp_x4[B:2 * B], disp_x4[2 * B:3 * B]), 1)
        disp_volume_n = F.interpolate(disp_volume_n, size=(H, W))  # back to target resolution

        # Get stat masks from depth volumes
        depth_volume = 1 / disp_volume_c
        mean_depth = torch.mean(depth_volume, dim=1, keepdim=True)
        var_depth = torch.var(depth_volume, dim=1, keepdim=True)
        dispersion = var_depth / (mean_depth ** 2)
        stat_mask_c = (dispersion < thres).type_as(disp_volume_c)

        depth_volume = 1 / disp_volume_p
        mean_depth = torch.mean(depth_volume, dim=1, keepdim=True)
        var_depth = torch.var(depth_volume, dim=1, keepdim=True)
        dispersion = var_depth / (mean_depth ** 2)
        stat_mask_p = (dispersion < thres).type_as(disp_volume_p)

        depth_volume = 1 / disp_volume_n
        mean_depth = torch.mean(depth_volume, dim=1, keepdim=True)
        var_depth = torch.var(depth_volume, dim=1, keepdim=True)
        dispersion = var_depth / (mean_depth ** 2)
        stat_mask_n = (dispersion < thres).type_as(disp_volume_n)

        return stat_mask_c, stat_mask_p, stat_mask_n


def project_and_fuse_spimo(depth, stat_mask_c, stat_mask_p, stat_mask_n, in_grid, this_camera_pose,
                           intrinsics, intrinsics_inv):
    B, _, _, _ = depth.shape

    this_depth = depth.detach()
    prev_pose = this_camera_pose[:B, :]
    next_pose = this_camera_pose[B:, :]
    with torch.no_grad():
        stat_mask_p = torch.cat((stat_mask_p, stat_mask_p, stat_mask_p), 1).detach()
        warped_stat_mask_p = inverse_warp(stat_mask_p, this_depth, -prev_pose.detach(), intrinsics, intrinsics_inv)[:,
                             0, :, :]
        stat_mask_p = stat_mask_c * warped_stat_mask_p.unsqueeze(1)

        stat_mask_n = torch.cat((stat_mask_n, stat_mask_n, stat_mask_n), 1).detach()
        warped_stat_mask_n = inverse_warp(stat_mask_n, this_depth, -next_pose.detach(), intrinsics, intrinsics_inv)[:,
                             0, :, :]
        stat_mask_n = stat_mask_c * warped_stat_mask_n.unsqueeze(1)

        if args.ignore_top:
            ignore_region = (in_grid[:, 1, :, :].unsqueeze(1) < -0.25).type_as(in_grid)
            stat_mask_p = stat_mask_p + ignore_region
            stat_mask_p[stat_mask_p > 1] = 1
            stat_mask_n = stat_mask_n + ignore_region
            stat_mask_n[stat_mask_n > 1] = 1

    return stat_mask_p, stat_mask_n


def validate(val_loader, m_model, epoch, output_writers):
    global args

    test_time = utils.AverageMeter()
    flow2_EPEs = utils.AverageMeter()
    kitti_erros = utils.multiAverageMeter(utils.kitti_error_names)

    # switch to evaluate mode
    m_model.eval()

    # Disable gradients to save memory
    with torch.no_grad():
        for i, input_data in enumerate(val_loader):
            left_view = input_data[0][0].cuda().float()
            # input_right = input_data[0][1].cuda()
            target = input_data[1][0].cuda()
            max_disp = torch.Tensor([args.max_disp]).unsqueeze(1).unsqueeze(1).type(left_view.type())
            B, C, H, W = left_view.shape

            # Prepare input data
            end = time.time()
            min_disp = max_disp * args.min_disp / args.max_disp

            # Normalize input view
            mean_shift = torch.zeros(B, C, 1, 1).type(left_view.type())
            mean_shift[:, 0, :, :] = 0.411
            mean_shift[:, 1, :, :] = 0.432
            mean_shift[:, 2, :, :] = 0.45
            left_view = left_view / 255 - mean_shift

            # Prepare identity grid
            i_tetha = torch.zeros(B, 2, 3).cuda()
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            a_grid = F.affine_grid(i_tetha, [B, 3, H, W], align_corners=True)
            grid = torch.zeros(B, 2, H, W).type(a_grid.type())
            grid[:, 0, :, :] = a_grid[:, :, :, 0]
            grid[:, 1, :, :] = a_grid[:, :, :, 1]

            # Run inference
            disp = m_model(left_view, grid, min_disp, max_disp)

            # Ensure disp is the same size as input
            if disp.shape[2] != H:
                disp = F.interpolate(disp, size=(H, W), mode='bilinear', align_corners=True)

            # Measure elapsed time
            test_time.update(time.time() - end)

            # Measure EPE (end point error or mean absolute error)
            flow2_EPE = realEPE(disp, target, sparse=args.sparse)
            flow2_EPEs.update(flow2_EPE.detach(), target.size(0))

            # Record kitti metrics (error and accuracy metrics in meters)
            target_depth, pred_depth = utils.disps_to_depths_kitti2015(target.detach().squeeze(1).cpu().numpy(),
                                                                       disp.detach().squeeze(1).cpu().numpy())
            kitti_erros.update(utils.compute_kitti_errors(target_depth[0], pred_depth[0], use_median=True), B)

            if i < len(output_writers):  # log first output of first batches
                denormalize = np.array([0.411, 0.432, 0.45])
                denormalize = denormalize[:, np.newaxis, np.newaxis]

                # Plot input image
                if epoch == 0:
                    output_writers[i].add_image('Inputs', left_view[0].cpu().numpy() + denormalize, 0)

                # Plot disp
                output_writers[i].add_image('Left disp', utils.disp2rgb(disp[0].cpu().numpy(), None), epoch)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'.format(i, len(val_loader), test_time, flow2_EPEs))

    # Print errors
    print(kitti_erros)
    return kitti_erros.avg[4]


if __name__ == '__main__':
    import os

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no

    # Torch related packages are placed here to prevent windows to re-import them randomly
    # (used to happen in previous versions of pytorch, not sure about newer ones)
    import torch
    import torch.utils.data
    import torchvision.transforms as transforms
    from tensorboardX import SummaryWriter
    import torch.backends.cudnn as cudnn
    import torch.nn.functional as F
    from math import exp

    import myUtils as utils
    import data_transforms
    from loss_functions import realEPE, smoothness, pose_loss, pose_loss_masked
    from inverse_warp import *
    main()
