# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/main_visibility_CALIB.py

import math
import os
import random
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# import apex
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from DatasetLidarCamera import DatasetLidarCameraKittiOdometry
from losses import (CombinedLoss, DistancePoints3D, GeometricLoss, L1Loss,
                    ProposedLoss, WeakSelfSupervisedLoss)
from models.LCCNet import LCCNet

from quaternion_distances import quaternion_distance

from tensorboardX import SummaryWriter
from utils import (compose_pose_from_prediction_batched, mat2xyzrpy,
                   merge_inputs, overlay_imgs, pad_point_clouds,
                   project_point_clouds_to_depth_batched, quat2mat,
                   quaternion_from_matrix, rotate_forward,
                   transform_point_cloud_homogeneous_batched, tvector2mat)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

ex = Experiment("LCCNet")
ex.captured_out_filter = apply_backspaces_and_linefeeds


# noinspection PyUnusedLocal
@ex.config
def config():
    checkpoints = './checkpoints/'
    dataset = 'kitti/odom' # 'kitti/raw'
    data_folder = '/home/zmy/datasets/kitti/dataset'
    use_reflectance = False
    val_sequence = 0
    epochs = 120
    BASE_LEARNING_RATE = 3e-4  # 1e-4
    loss = 'combined'
    max_t = 0.1 # 1.5, 1.0,  0.5,  0.2,  0.1
    max_r = 1. # 20.0, 10.0, 5.0,  2.0,  1.0
    batch_size = 8  # conservative default for single-GPU fine-tuning
    num_worker = 2
    network = 'Res_f1'
    optimizer = 'adam'
    resume = False
    weights = './pretrained/kitti_iter5.tar'
    rescale_rot = 1.0
    rescale_transl = 2.0
    precision = "O0"
    norm = 'bn'
    dropout = 0.0
    max_depth = 80.
    weight_point_cloud = 0.5
    log_frequency = 10
    print_frequency = 50
    starting_epoch = 0

    # Phase B scaffold: keep baseline default unchanged.
    training_mode = 'supervised_baseline'  # ['supervised_baseline', 'weak_self_supervised']
    selfsup_recipe = 'a_weak_v1'
    lambda_depth = 2.0
    lambda_edge = 0.5
    lambda_mask = 0.05
    lambda_pose_prior = 0.01
    lambda_sup_aux = 0.3
    selfsup_warmup_epochs = 8
    sup_aux_decay_epochs = 35
    depth_loss_type = 'charbonnier'
    min_valid_points = 128
    detach_reference_depth = True
    max_train_batches = -1
    max_val_batches = -1
    use_amp = False
    amp_dtype = 'fp16'  # ['bf16', 'fp16']
    enable_image_logging = False
    dataloader_prefetch_factor = 4
    dataloader_persistent_workers = True
    diagnostic_frequency = 50
    finite_check_frequency = 20


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCH = 1
def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH*100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _losses_are_finite(loss_dict):
    for key, value in loss_dict.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return False, key
    return True, None


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv


def prepare_projected_batch(sample, img_shape, max_depth):
    rgb_input = []
    shape_pad_input = []
    real_shape_input = []
    pc_lidar_input = []
    pc_rotated_input = []

    target_device = sample['tr_error'].device

    for idx in range(len(sample['rgb'])):
        real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

        sample['point_cloud'][idx] = sample['point_cloud'][idx].to(device=target_device, non_blocking=True)
        pc_lidar = sample['point_cloud'][idx].clone()
        if max_depth < 80.:
            pc_lidar = pc_lidar[:, pc_lidar[0, :] < max_depth].clone()

        rgb = sample['rgb'][idx].to(device=target_device, non_blocking=True)
        shape_pad = [0, 0, 0, 0]
        shape_pad[3] = img_shape[0] - rgb.shape[1]
        shape_pad[1] = img_shape[1] - rgb.shape[2]
        rgb = F.pad(rgb, shape_pad)

        rgb_input.append(rgb)
        shape_pad_input.append(shape_pad)
        real_shape_input.append(real_shape)
        pc_lidar_input.append(pc_lidar)

    rgb_input = torch.stack(rgb_input)
    calib_batch = sample['calib'].to(device=rgb_input.device, dtype=rgb_input.dtype)

    pc_lidar_batch, pc_lidar_mask = pad_point_clouds(pc_lidar_input)

    rt_matrices = compose_pose_from_prediction_batched(
        sample['tr_error'].to(device=rgb_input.device, dtype=rgb_input.dtype),
        sample['rot_error'].to(device=rgb_input.device, dtype=rgb_input.dtype),
    )
    pc_rotated_batch = transform_point_cloud_homogeneous_batched(pc_lidar_batch, rt_matrices, inverse=False)
    pc_rotated_mask = pc_lidar_mask.clone()
    if max_depth < 80.:
        pc_rotated_mask = pc_rotated_mask & (pc_rotated_batch[:, 0, :] < max_depth)

    for idx in range(pc_rotated_batch.shape[0]):
        pc_rotated_input.append(pc_rotated_batch[idx, :, pc_rotated_mask[idx]].contiguous())

    lidar_gt = project_point_clouds_to_depth_batched(
        pc_lidar_batch,
        calib_batch,
        img_shape,
        point_mask=pc_lidar_mask,
        min_depth=1e-3,
        max_depth=max_depth,
    )['depth'] / max_depth

    lidar_input = project_point_clouds_to_depth_batched(
        pc_rotated_batch,
        calib_batch,
        img_shape,
        point_mask=pc_rotated_mask,
        min_depth=1e-3,
        max_depth=max_depth,
    )['depth'] / max_depth

    return {
        'rgb_input': rgb_input,
        'lidar_input': lidar_input,
        'lidar_gt': lidar_gt,
        'real_shape_input': real_shape_input,
        'shape_pad_input': shape_pad_input,
        'pc_rotated_input': pc_rotated_input,
    }


# CCN training
@ex.capture
def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss,
          training_mode='supervised_baseline', point_clouds_miscalib=None,
          reference_depths=None, calibs=None, image_shapes=None, epoch=0,
          max_depth=80.0, selfsup_warmup_epochs=5, sup_aux_decay_epochs=15,
          use_amp=True, amp_dtype='bf16', grad_scaler=None):
    model.train()

    optimizer.zero_grad(set_to_none=True)

    autocast_enabled = bool(use_amp and torch.cuda.is_available())
    autocast_dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16

    # Run model and compute loss with autocast for faster Tensor Core execution.
    with torch.autocast(device_type='cuda', dtype=autocast_dtype, enabled=autocast_enabled):
        transl_err, rot_err = model(rgb_img, refl_img)

        if training_mode == 'supervised_baseline':
            if loss == 'points_distance' or loss == 'combined':
                losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
            else:
                losses = loss_fn(target_transl, target_rot, transl_err, rot_err)
        else:
            if epoch < selfsup_warmup_epochs:
                sup_aux_weight = 1.0
            elif sup_aux_decay_epochs <= 0:
                sup_aux_weight = 0.0
            else:
                decay_epoch = min(epoch - selfsup_warmup_epochs, sup_aux_decay_epochs)
                sup_aux_weight = max(0.0, 1.0 - float(decay_epoch) / float(sup_aux_decay_epochs))

            losses = loss_fn(
                point_clouds_miscalib=point_clouds_miscalib,
                reference_depths=reference_depths,
                calibs=calibs,
                target_transl=target_transl,
                target_rot=target_rot,
                transl_err=transl_err,
                rot_err=rot_err,
                image_shape=image_shapes,
                max_depth=max_depth,
                sup_aux_weight=sup_aux_weight,
            )
            losses['sup_aux_weight'] = transl_err.new_tensor(sup_aux_weight)

    if autocast_enabled and amp_dtype == 'fp16' and grad_scaler is not None:
        grad_scaler.scale(losses['total_loss']).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        losses['total_loss'].backward()
        optimizer.step()

    return losses, rot_err, transl_err


# CNN test
@ex.capture
def val(model, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss,
    training_mode='supervised_baseline', point_clouds_miscalib=None,
    reference_depths=None, calibs=None, image_shapes=None, epoch=0,
    max_depth=80.0, selfsup_warmup_epochs=5, sup_aux_decay_epochs=15,
    use_amp=True, amp_dtype='bf16'):
    model.eval()

    # Run model
    autocast_enabled = bool(use_amp and torch.cuda.is_available())
    autocast_dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=autocast_dtype, enabled=autocast_enabled):
            transl_err, rot_err = model(rgb_img, refl_img)

    if training_mode == 'supervised_baseline':
        if loss == 'points_distance' or loss == 'combined':
            losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
        else:
            losses = loss_fn(target_transl, target_rot, transl_err, rot_err)
    else:
        if epoch < selfsup_warmup_epochs:
            sup_aux_weight = 1.0
        elif sup_aux_decay_epochs <= 0:
            sup_aux_weight = 0.0
        else:
            decay_epoch = min(epoch - selfsup_warmup_epochs, sup_aux_decay_epochs)
            sup_aux_weight = max(0.0, 1.0 - float(decay_epoch) / float(sup_aux_decay_epochs))

        losses = loss_fn(
            point_clouds_miscalib=point_clouds_miscalib,
            reference_depths=reference_depths,
            calibs=calibs,
            target_transl=target_transl,
            target_rot=target_rot,
            transl_err=transl_err,
            rot_err=rot_err,
            image_shape=image_shapes,
            max_depth=max_depth,
            sup_aux_weight=sup_aux_weight,
        )
        losses['sup_aux_weight'] = transl_err.new_tensor(sup_aux_weight)

    # if loss != 'points_distance':
    #     total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)
    # else:
    #     total_loss = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)

    total_trasl_error = torch.tensor(0.0, device=target_transl.device)
    total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi
    for j in range(rgb_img.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.

    # # output image: The overlay image of the input rgb image and the projected lidar pointcloud depth image
    # cam_intrinsic = camera_model[0]
    # rotated_point_cloud =
    # R_predicted = quat2mat(R_predicted[0])
    # T_predicted = tvector2mat(T_predicted[0])
    # RT_predicted = torch.mm(T_predicted, R_predicted)
    # rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

    return losses, total_trasl_error.item(), total_rot_error.sum().item(), rot_err, transl_err


@ex.automain
def main(_config, _run, seed):
    global EPOCH
    print('Loss Function Choice: {}'.format(_config['loss']))
    print('Training Mode: {}'.format(_config['training_mode']))

    if _config['training_mode'] not in ['supervised_baseline', 'weak_self_supervised']:
        raise ValueError("Unknown training_mode: {}".format(_config['training_mode']))

    # Phase A guard: keep baseline behavior strict and explicit.
    if _config['training_mode'] == 'supervised_baseline':
        if _config['loss'] not in ['simple', 'geometric', 'points_distance', 'L1', 'combined']:
            raise ValueError("Unsupported baseline loss: {}".format(_config['loss']))

    if _config['val_sequence'] is None:
        raise TypeError('val_sequences cannot be None')
    else:
        val_sequence = f"{_config['val_sequence']:02d}"
        print("Val Sequence: ", val_sequence)
        dataset_class = DatasetLidarCameraKittiOdometry
    img_shape = (384, 1280) # 网络的输入尺度
    input_size = (256, 512)
    checkpoints_root = os.path.join(_config["checkpoints"], _config['dataset'])

    dataset_train = dataset_class(
        _config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
        split='train', use_reflectance=_config['use_reflectance'],
        val_sequence=val_sequence)
    dataset_val = dataset_class(
        _config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
        split='val', use_reflectance=_config['use_reflectance'],
        val_sequence=val_sequence)
    model_savepath = os.path.join(checkpoints_root, 'val_seq_' + val_sequence, 'models')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    log_savepath = os.path.join(checkpoints_root, 'val_seq_' + val_sequence, 'log')
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    train_writer = SummaryWriter(os.path.join(log_savepath, 'train'))
    val_writer = SummaryWriter(os.path.join(log_savepath, 'val'))

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    train_dataset_size = len(dataset_train)
    val_dataset_size = len(dataset_val)
    print('Number of the train dataset: {}'.format(train_dataset_size))
    print('Number of the val dataset: {}'.format(val_dataset_size))

    # Training and validation set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    loader_kwargs = {
        'worker_init_fn': init_fn,
        'collate_fn': merge_inputs,
        'drop_last': False,
        'pin_memory': True,
    }
    if num_worker > 0:
        loader_kwargs['persistent_workers'] = bool(_config['dataloader_persistent_workers'])
        loader_kwargs['prefetch_factor'] = int(_config['dataloader_prefetch_factor'])

    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=num_worker,
                                                 **loader_kwargs)

    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                **loader_kwargs)

    print(len(TrainImgLoader))
    print(len(ValImgLoader))

    # loss function choice
    if _config['training_mode'] == 'supervised_baseline':
        if _config['loss'] == 'simple':
            loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
        elif _config['loss'] == 'geometric':
            loss_fn = GeometricLoss()
            loss_fn = loss_fn.cuda()
        elif _config['loss'] == 'points_distance':
            loss_fn = DistancePoints3D()
        elif _config['loss'] == 'L1':
            loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])
        elif _config['loss'] == 'combined':
            loss_fn = CombinedLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])
        else:
            raise ValueError("Unknown Loss Function")
    else:
        loss_fn = WeakSelfSupervisedLoss(
            lambda_depth=_config['lambda_depth'],
            lambda_edge=_config['lambda_edge'],
            lambda_mask=_config['lambda_mask'],
            lambda_pose_prior=_config['lambda_pose_prior'],
            lambda_sup_aux=_config['lambda_sup_aux'],
            depth_loss_type=_config['depth_loss_type'],
            min_valid_points=_config['min_valid_points'],
            detach_reference_depth=_config['detach_reference_depth'],
        )

    #runs = datetime.now().strftime('%b%d_%H-%M-%S') + "/"
    # train_writer = SummaryWriter('./logs/' + runs)
    #ex.info["tensorflow"] = {}
    #ex.info["tensorflow"]["logdirs"] = ['./logs/' + runs]

    # network choice and settings
    if _config['network'].startswith('Res'):
        feat = 1
        md = 4
        split = _config['network'].split('_')
        for item in split[1:]:
            if item.startswith('f'):
                feat = int(item[-1])
            elif item.startswith('md'):
                md = int(item[2:])
        assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
        assert 0 < md, "md must be positive"
        model = LCCNet(input_size, use_feat_from=feat, md=md,
                         use_reflectance=_config['use_reflectance'], dropout=_config['dropout'],
                         Action_Func='leakyrelu', attention=False, res_num=18)
    else:
        raise TypeError("Network unknown")
    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        if len(saved_state_dict) > 0 and list(saved_state_dict.keys())[0].startswith('module.'):
            saved_state_dict = {k[7:]: v for k, v in saved_state_dict.items()}
        model.load_state_dict(saved_state_dict)

        # original saved file with DataParallel
        # state_dict = torch.load(model_path)
        # create new OrderedDict that does not contain `module.`
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # model.load_state_dict(new_state_dict)

    # model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()

    amp_dtype = _config['amp_dtype']
    if _config['use_amp'] and amp_dtype == 'bf16':
        print("BF16 is not supported by correlation CUDA op in this repo; fallback to FP16 AMP.")
        amp_dtype = 'fp16'

    grad_scaler = torch.cuda.amp.GradScaler(enabled=bool(_config['use_amp'] and amp_dtype == 'fp16'))

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if _config['loss'] == 'geometric':
        parameters += list(loss_fn.parameters())
    if _config['optimizer'] == 'adam':
        optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
        # Probably this scheduler is not used
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)
    else:
        optimizer = optim.SGD(parameters, lr=_config['BASE_LEARNING_RATE'], momentum=0.9,
                              weight_decay=5e-6, nesterov=True)

    starting_epoch = _config['starting_epoch']
    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        if 'optimizer' in checkpoint:
            opt_state_dict = checkpoint['optimizer']
            optimizer.load_state_dict(opt_state_dict)
        if starting_epoch == 0 and 'epoch' in checkpoint:
            starting_epoch = checkpoint['epoch'] + 1

    # Allow mixed-precision if needed
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level=_config["precision"])

    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.
    old_save_filename = None

    train_iter = 0
    val_iter = 0
    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        print('This is %d-th epoch' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        local_loss = 0.
        local_preprocess_time = 0.0
        local_optimize_time = 0.0
        local_iter_time = 0.0
        if _config['optimizer'] != 'adam':
            _run.log_scalar("LR", _config['BASE_LEARNING_RATE'] *
                            math.exp((1 - epoch) * 4e-2), epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = _config['BASE_LEARNING_RATE'] * \
                                    math.exp((1 - epoch) * 4e-2)
        else:
            #scheduler.step(epoch%100)
            _run.log_scalar("LR", scheduler.get_lr()[0])


        ## Training ##
        time_for_50ep = time.time()
        for batch_idx, sample in enumerate(TrainImgLoader):
            if _config['max_train_batches'] > 0 and batch_idx >= _config['max_train_batches']:
                break
            #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            start_preprocess = time.time()
            prepared = prepare_projected_batch(sample, img_shape, _config['max_depth'])
            rgb_input = prepared['rgb_input']
            lidar_input = prepared['lidar_input']
            lidar_gt = prepared['lidar_gt']
            real_shape_input = prepared['real_shape_input']
            shape_pad_input = prepared['shape_pad_input']
            pc_rotated_input = prepared['pc_rotated_input']

            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
            lidar_input = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")
            rgb_input = rgb_input.contiguous(memory_format=torch.channels_last)
            lidar_input = lidar_input.contiguous(memory_format=torch.channels_last)
            end_preprocess = time.time()

            start_optimize = time.time()
            loss, R_predicted,  T_predicted = train(
                model, optimizer, rgb_input, lidar_input,
                sample['tr_error'], sample['rot_error'],
                loss_fn, sample['point_cloud'], _config['loss'], _config['training_mode'],
                point_clouds_miscalib=pc_rotated_input,
                reference_depths=lidar_gt,
                calibs=sample['calib'],
                image_shapes=real_shape_input,
                epoch=epoch,
                max_depth=_config['max_depth'],
                selfsup_warmup_epochs=_config['selfsup_warmup_epochs'],
                sup_aux_decay_epochs=_config['sup_aux_decay_epochs'],
                use_amp=_config['use_amp'],
                amp_dtype=amp_dtype,
                grad_scaler=grad_scaler,
            )
            end_optimize = time.time()

            if _config['finite_check_frequency'] > 0 and batch_idx % _config['finite_check_frequency'] == 0:
                is_finite, failed_key = _losses_are_finite(loss)
                if not is_finite:
                    raise ValueError("Loss {} is NaN/Inf".format(failed_key))

            if batch_idx % _config['log_frequency'] == 0:
                show_idx = 0
                # output image: The overlay image of the input rgb image
                # and the projected lidar pointcloud depth image
                if _config['enable_image_logging']:
                    rotated_point_cloud = pc_rotated_input[show_idx]
                    R_predicted = quat2mat(R_predicted[show_idx])
                    T_predicted = tvector2mat(T_predicted[show_idx])
                    RT_predicted = torch.mm(T_predicted, R_predicted)
                    rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

                    depth_pred, uv = lidar_project_depth(rotated_point_cloud,
                                                        sample['calib'][show_idx],
                                                        real_shape_input[show_idx]) # or image_shape
                    depth_pred /= _config['max_depth']
                    depth_pred = F.pad(depth_pred, shape_pad_input[show_idx])

                    pred_show = overlay_imgs(rgb_show[show_idx], depth_pred.unsqueeze(0))
                    input_show = overlay_imgs(rgb_show[show_idx], lidar_show[show_idx].unsqueeze(0))
                    gt_show = overlay_imgs(rgb_show[show_idx], lidar_gt[show_idx].unsqueeze(0))

                    pred_show = torch.from_numpy(pred_show)
                    pred_show = pred_show.permute(2, 0, 1)
                    input_show = torch.from_numpy(input_show)
                    input_show = input_show.permute(2, 0, 1)
                    gt_show = torch.from_numpy(gt_show)
                    gt_show = gt_show.permute(2, 0, 1)

                    train_writer.add_image("input_proj_lidar", input_show, train_iter)
                    train_writer.add_image("gt_proj_lidar", gt_show, train_iter)
                    train_writer.add_image("pred_proj_lidar", pred_show, train_iter)

                train_writer.add_scalar("Loss_Total", loss['total_loss'].item(), train_iter)
                train_writer.add_scalar("Loss_Translation", loss['transl_loss'].item(), train_iter)
                train_writer.add_scalar("Loss_Rotation", loss['rot_loss'].item(), train_iter)
                if _config['loss'] == 'combined' and 'point_clouds_loss' in loss:
                    train_writer.add_scalar("Loss_Point_clouds", loss['point_clouds_loss'].item(), train_iter)
                if 'depth_loss' in loss:
                    train_writer.add_scalar("Loss_Depth", loss['depth_loss'].item(), train_iter)
                if 'edge_loss' in loss:
                    train_writer.add_scalar("Loss_Edge", loss['edge_loss'].item(), train_iter)
                if 'mask_loss' in loss:
                    train_writer.add_scalar("Loss_Mask", loss['mask_loss'].item(), train_iter)
                if 'pose_prior_loss' in loss:
                    train_writer.add_scalar("Loss_PosePrior", loss['pose_prior_loss'].item(), train_iter)
                if 'sup_aux_loss' in loss:
                    train_writer.add_scalar("Loss_SupAux", loss['sup_aux_loss'].item(), train_iter)
                if 'sup_aux_weight' in loss:
                    train_writer.add_scalar("Weight_SupAux", loss['sup_aux_weight'].item(), train_iter)
                if 'valid_ratio' in loss:
                    train_writer.add_scalar("Valid_Ratio", loss['valid_ratio'].item(), train_iter)
                if 'valid_points_mean' in loss:
                    train_writer.add_scalar("Valid_Points_Mean", loss['valid_points_mean'].item(), train_iter)
                if 'pred_transl_norm' in loss:
                    train_writer.add_scalar("Pred_Transl_Norm", loss['pred_transl_norm'].item(), train_iter)
                if 'pred_rot_norm_dev' in loss:
                    train_writer.add_scalar("Pred_Rot_Norm_Dev", loss['pred_rot_norm_dev'].item(), train_iter)

            local_loss += loss['total_loss'].item()
            preprocess_time = end_preprocess - start_preprocess
            optimize_time = end_optimize - start_optimize
            iter_time = time.time() - start_time
            local_preprocess_time += preprocess_time
            local_optimize_time += optimize_time
            local_iter_time += iter_time

            if batch_idx % 50 == 0 and batch_idx != 0:
                avg_pre = local_preprocess_time / 50.0
                avg_opt = local_optimize_time / 50.0
                avg_iter = local_iter_time / 50.0
                avg_pre_ratio = avg_pre / max(avg_iter, 1e-8)
                avg_opt_ratio = avg_opt / max(avg_iter, 1e-8)

                print(f'Iter {batch_idx}/{len(TrainImgLoader)} training loss = {local_loss/50:.3f}, '
                      f'time = {(time.time() - start_time)/lidar_input.shape[0]:.4f}, '
                      f'time_preprocess = {avg_pre/lidar_input.shape[0]:.4f}, '
                      f'time_optimize = {avg_opt/lidar_input.shape[0]:.4f}, '
                      f'pre_ratio = {avg_pre_ratio:.2f}, '
                      f'opt_ratio = {avg_opt_ratio:.2f}, '
                      f'time for 50 iter: {time.time()-time_for_50ep:.4f}')
                time_for_50ep = time.time()
                _run.log_scalar("Loss", local_loss/50, train_iter)
                if _config['training_mode'] == 'weak_self_supervised' and _config['diagnostic_frequency'] > 0 and batch_idx % _config['diagnostic_frequency'] == 0:
                    _run.log_scalar("Diag_Train_Preprocess_Sec", avg_pre, train_iter)
                    _run.log_scalar("Diag_Train_Optimize_Sec", avg_opt, train_iter)
                    _run.log_scalar("Diag_Train_Iter_Sec", avg_iter, train_iter)
                local_loss = 0.
                local_preprocess_time = 0.0
                local_optimize_time = 0.0
                local_iter_time = 0.0
            total_train_loss += loss['total_loss'].item() * len(sample['rgb'])
            train_iter += 1
            # total_iter += len(sample['rgb'])

        print("------------------------------------")
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(dataset_train)))
        print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
        print("------------------------------------")
        _run.log_scalar("Total training loss", total_train_loss / len(dataset_train), epoch)

        ## Validation ##
        total_val_loss = 0.
        total_val_t = 0.
        total_val_r = 0.

        local_loss = 0.0
        local_val_preprocess_time = 0.0
        local_val_optimize_time = 0.0
        local_val_iter_time = 0.0
        for batch_idx, sample in enumerate(ValImgLoader):
            if _config['max_val_batches'] > 0 and batch_idx >= _config['max_val_batches']:
                break
            #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            prepared = prepare_projected_batch(sample, img_shape, _config['max_depth'])
            rgb_input = prepared['rgb_input']
            lidar_input = prepared['lidar_input']
            lidar_gt = prepared['lidar_gt']
            real_shape_input = prepared['real_shape_input']
            shape_pad_input = prepared['shape_pad_input']
            pc_rotated_input = prepared['pc_rotated_input']

            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
            lidar_input = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")
            rgb_input = rgb_input.contiguous(memory_format=torch.channels_last)
            lidar_input = lidar_input.contiguous(memory_format=torch.channels_last)
            end_preprocess = time.time()

            start_optimize = time.time()
            loss, trasl_e, rot_e, R_predicted,  T_predicted = val(
                model, rgb_input, lidar_input,
                sample['tr_error'], sample['rot_error'],
                loss_fn, sample['point_cloud'], _config['loss'], _config['training_mode'],
                point_clouds_miscalib=pc_rotated_input,
                reference_depths=lidar_gt,
                calibs=sample['calib'],
                image_shapes=real_shape_input,
                epoch=epoch,
                max_depth=_config['max_depth'],
                selfsup_warmup_epochs=_config['selfsup_warmup_epochs'],
                sup_aux_decay_epochs=_config['sup_aux_decay_epochs'],
                use_amp=_config['use_amp'],
                amp_dtype=amp_dtype,
            )
            end_optimize = time.time()

            if _config['finite_check_frequency'] > 0 and batch_idx % _config['finite_check_frequency'] == 0:
                is_finite, failed_key = _losses_are_finite(loss)
                if not is_finite:
                    raise ValueError("Loss {} is NaN/Inf".format(failed_key))

            if batch_idx % _config['log_frequency'] == 0:
                show_idx = 0
                # output image: The overlay image of the input rgb image
                # and the projected lidar pointcloud depth image
                if _config['enable_image_logging']:
                    rotated_point_cloud = pc_rotated_input[show_idx]
                    R_predicted = quat2mat(R_predicted[show_idx])
                    T_predicted = tvector2mat(T_predicted[show_idx])
                    RT_predicted = torch.mm(T_predicted, R_predicted)
                    rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

                    depth_pred, uv = lidar_project_depth(rotated_point_cloud,
                                                        sample['calib'][show_idx],
                                                        real_shape_input[show_idx]) # or image_shape
                    depth_pred /= _config['max_depth']
                    depth_pred = F.pad(depth_pred, shape_pad_input[show_idx])

                    pred_show = overlay_imgs(rgb_show[show_idx], depth_pred.unsqueeze(0))
                    input_show = overlay_imgs(rgb_show[show_idx], lidar_show[show_idx].unsqueeze(0))
                    gt_show = overlay_imgs(rgb_show[show_idx], lidar_gt[show_idx].unsqueeze(0))

                    pred_show = torch.from_numpy(pred_show)
                    pred_show = pred_show.permute(2, 0, 1)
                    input_show = torch.from_numpy(input_show)
                    input_show = input_show.permute(2, 0, 1)
                    gt_show = torch.from_numpy(gt_show)
                    gt_show = gt_show.permute(2, 0, 1)

                    val_writer.add_image("input_proj_lidar", input_show, val_iter)
                    val_writer.add_image("gt_proj_lidar", gt_show, val_iter)
                    val_writer.add_image("pred_proj_lidar", pred_show, val_iter)

                val_writer.add_scalar("Loss_Total", loss['total_loss'].item(), val_iter)
                val_writer.add_scalar("Loss_Translation", loss['transl_loss'].item(), val_iter)
                val_writer.add_scalar("Loss_Rotation", loss['rot_loss'].item(), val_iter)
                if _config['loss'] == 'combined' and 'point_clouds_loss' in loss:
                    val_writer.add_scalar("Loss_Point_clouds", loss['point_clouds_loss'].item(), val_iter)
                if 'depth_loss' in loss:
                    val_writer.add_scalar("Loss_Depth", loss['depth_loss'].item(), val_iter)
                if 'edge_loss' in loss:
                    val_writer.add_scalar("Loss_Edge", loss['edge_loss'].item(), val_iter)
                if 'mask_loss' in loss:
                    val_writer.add_scalar("Loss_Mask", loss['mask_loss'].item(), val_iter)
                if 'pose_prior_loss' in loss:
                    val_writer.add_scalar("Loss_PosePrior", loss['pose_prior_loss'].item(), val_iter)
                if 'sup_aux_loss' in loss:
                    val_writer.add_scalar("Loss_SupAux", loss['sup_aux_loss'].item(), val_iter)
                if 'sup_aux_weight' in loss:
                    val_writer.add_scalar("Weight_SupAux", loss['sup_aux_weight'].item(), val_iter)
                if 'valid_ratio' in loss:
                    val_writer.add_scalar("Valid_Ratio", loss['valid_ratio'].item(), val_iter)
                if 'valid_points_mean' in loss:
                    val_writer.add_scalar("Valid_Points_Mean", loss['valid_points_mean'].item(), val_iter)
                if 'pred_transl_norm' in loss:
                    val_writer.add_scalar("Pred_Transl_Norm", loss['pred_transl_norm'].item(), val_iter)
                if 'pred_rot_norm_dev' in loss:
                    val_writer.add_scalar("Pred_Rot_Norm_Dev", loss['pred_rot_norm_dev'].item(), val_iter)


            total_val_t += trasl_e
            total_val_r += rot_e
            local_loss += loss['total_loss'].item()
            val_preprocess_time = end_preprocess - start_time
            val_optimize_time = end_optimize - start_optimize
            val_iter_time = time.time() - start_time
            local_val_preprocess_time += val_preprocess_time
            local_val_optimize_time += val_optimize_time
            local_val_iter_time += val_iter_time

            if batch_idx % 50 == 0 and batch_idx != 0:
                avg_val_pre = local_val_preprocess_time / 50.0
                avg_val_opt = local_val_optimize_time / 50.0
                avg_val_iter = local_val_iter_time / 50.0
                print('Iter %d val loss = %.3f , time = %.2f' % (batch_idx, local_loss/50.,
                                                                  (time.time() - start_time)/lidar_input.shape[0]))
                print('Val diag: pre=%.4f opt=%.4f iter=%.4f pre_ratio=%.2f opt_ratio=%.2f' % (
                    avg_val_pre / lidar_input.shape[0],
                    avg_val_opt / lidar_input.shape[0],
                    avg_val_iter / lidar_input.shape[0],
                    avg_val_pre / max(avg_val_iter, 1e-8),
                    avg_val_opt / max(avg_val_iter, 1e-8),
                ))
                local_loss = 0.0
                local_val_preprocess_time = 0.0
                local_val_optimize_time = 0.0
                local_val_iter_time = 0.0
            total_val_loss += loss['total_loss'].item() * len(sample['rgb'])
            val_iter += 1

        print("------------------------------------")
        print('total val loss = %.3f' % (total_val_loss / len(dataset_val)))
        print(f'total traslation error: {total_val_t / len(dataset_val)} cm')
        print(f'total rotation error: {total_val_r / len(dataset_val)} °')
        print("------------------------------------")

        _run.log_scalar("Val_Loss", total_val_loss / len(dataset_val), epoch)
        _run.log_scalar("Val_t_error", total_val_t / len(dataset_val), epoch)
        _run.log_scalar("Val_r_error", total_val_r / len(dataset_val), epoch)

        # SAVE
        val_loss = total_val_loss / len(dataset_val)
        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            #_run.result = BEST_VAL_LOSS
            if _config['rescale_transl'] > 0:
                _run.result = total_val_t / len(dataset_val)
            else:
                _run.result = total_val_r / len(dataset_val)
            savefilename = f'{model_savepath}/checkpoint_r{_config["max_r"]:.2f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}.tar'
            torch.save({
                'config': _config,
                'epoch': epoch,
                'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': total_train_loss / len(dataset_train),
                'val_loss': total_val_loss / len(dataset_val),
            }, savefilename)
            print(f'Model saved as {savefilename}')
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    return _run.result
