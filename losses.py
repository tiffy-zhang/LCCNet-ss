# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/losses.pyy

import torch
from torch import nn as nn
import torch.nn.functional as F

from quaternion_distances import quaternion_distance
from utils import (differentiable_reproject_depth_batched, pad_point_clouds,
                   quat2mat, rotate_back, rotate_forward, tvector2mat,
                   quaternion_from_matrix)


class GeometricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sx = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.sq = torch.nn.Parameter(torch.Tensor([-3.0]), requires_grad=True)
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = torch.exp(-self.sx) * loss_transl + self.sx
        total_loss += torch.exp(-self.sq) * loss_rot + self.sq
        return total_loss


class ProposedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(ProposedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.losses = {}

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean() * 100
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        self.losses['total_loss'] = total_loss
        self.losses['transl_loss'] = loss_transl
        self.losses['rot_loss'] = loss_rot
        return self.losses


class L1Loss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(L1Loss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = self.transl_loss(rot_err, target_rot).sum(1).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        return total_loss


class DistancePoints3D(nn.Module):
    def __init__(self):
        super(DistancePoints3D, self).__init__()

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The mean distance between 3D points
        """
        #start = time.time()
        total_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            total_loss += error.mean()

        #end = time.time()
        #print("3D Distance Time: ", end-start)

        return total_loss/target_transl.shape[0]


# The combination of L1 loss of translation part,
# quaternion angle loss of rotation part,
# distance loss of the pointclouds
class CombinedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(CombinedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot

        #start = time.time()
        point_clouds_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            point_clouds_loss += error.mean()

        #end = time.time()
        #print("3D Distance Time: ", end-start)
        total_loss = (1 - self.weight_point_cloud) * pose_loss +\
                     self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0])
        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss/target_transl.shape[0]

        return self.loss


class WeakSelfSupervisedLoss(nn.Module):
    def __init__(self, lambda_depth=1.0, lambda_edge=0.2, lambda_mask=0.05,
                 lambda_pose_prior=0.05, lambda_sup_aux=0.2,
                 depth_loss_type='smooth_l1', min_valid_points=128,
                 detach_reference_depth=True):
        super(WeakSelfSupervisedLoss, self).__init__()
        self.lambda_depth = lambda_depth
        self.lambda_edge = lambda_edge
        self.lambda_mask = lambda_mask
        self.lambda_pose_prior = lambda_pose_prior
        self.lambda_sup_aux = lambda_sup_aux
        self.depth_loss_type = depth_loss_type
        self.min_valid_points = min_valid_points
        self.detach_reference_depth = detach_reference_depth
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.loss = {}

    @staticmethod
    def _edge_maps(depth):
        if depth.dim() == 4:
            dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
            dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        elif depth.dim() == 3:
            dx = depth[:, :, 1:] - depth[:, :, :-1]
            dy = depth[:, 1:, :] - depth[:, :-1, :]
        else:
            raise TypeError("depth must have shape [B,1,H,W] or [1,H,W]")
        return dx, dy

    def _depth_loss(self, pred_depth, ref_depth, mask):
        mask_f = mask.float()
        denom = mask_f.sum().clamp(min=1.0)
        if self.depth_loss_type == 'charbonnier':
            diff = pred_depth - ref_depth
            charbonnier = torch.sqrt(diff * diff + 1e-6)
            return (charbonnier * mask_f).sum() / denom
        smooth_l1 = F.smooth_l1_loss(pred_depth, ref_depth, reduction='none')
        return (smooth_l1 * mask_f).sum() / denom

    def forward(self, point_clouds_miscalib, reference_depths, calibs,
                target_transl, target_rot, transl_err, rot_err,
                image_shape=None, max_depth=80.0, sup_aux_weight=1.0):
        device = transl_err.device
        batch_size = transl_err.shape[0]

        padded_point_clouds, point_mask = pad_point_clouds([
            point_cloud.to(device) for point_cloud in point_clouds_miscalib
        ])

        if isinstance(reference_depths, list):
            reference_depths = torch.stack(reference_depths, dim=0)
        reference_depths = reference_depths.to(device)
        if reference_depths.dim() == 3:
            reference_depths = reference_depths.unsqueeze(1)
        if self.detach_reference_depth:
            reference_depths = reference_depths.detach()

        if isinstance(calibs, list):
            calibs = torch.stack(calibs, dim=0)
        calibs = calibs.to(device)

        cur_h, cur_w = int(reference_depths.shape[-2]), int(reference_depths.shape[-1])
        reproj = differentiable_reproject_depth_batched(
            point_clouds=padded_point_clouds,
            pred_transl=transl_err,
            pred_rot=rot_err,
            cam_intrinsics=calibs,
            image_shape=(cur_h, cur_w),
            point_mask=point_mask,
            min_depth=1e-3,
            max_depth=max_depth,
            mode='bilinear',
            apply_inverse_pose=True,
        )

        pred_depth = reproj['depth'] / max_depth
        valid_mask = reproj['valid_mask'] & (reference_depths > 0)

        depth_loss = self._depth_loss(pred_depth, reference_depths, valid_mask)

        pdx, pdy = self._edge_maps(pred_depth)
        rdx, rdy = self._edge_maps(reference_depths)
        mdx = valid_mask[:, :, :, 1:] & valid_mask[:, :, :, :-1]
        mdy = valid_mask[:, :, 1:, :] & valid_mask[:, :, :-1, :]

        mdx_f = mdx.float()
        mdy_f = mdy.float()
        edge_x = ((pdx - rdx).abs() * mdx_f).sum() / mdx_f.sum().clamp(min=1.0)
        edge_y = ((pdy - rdy).abs() * mdy_f).sum() / mdy_f.sum().clamp(min=1.0)
        edge_loss = edge_x + edge_y

        valid_count = reproj['valid_points'].sum(dim=1).float()
        mask_loss = torch.clamp(
            (self.min_valid_points - valid_count) / float(self.min_valid_points),
            min=0.0
        ).mean()
        valid_ratio = valid_mask.float().mean()

        pose_prior_loss = transl_err.norm(dim=1).mean() + (rot_err.norm(dim=1) - 1.0).abs().mean()

        sup_transl_loss = self.transl_loss(transl_err, target_transl).sum(1).mean()
        sup_rot_loss = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        sup_aux_loss = sup_transl_loss + sup_rot_loss

        total_loss = self.lambda_depth * depth_loss + \
                     self.lambda_edge * edge_loss + \
                     self.lambda_mask * mask_loss + \
                     self.lambda_pose_prior * pose_prior_loss + \
                     self.lambda_sup_aux * sup_aux_weight * sup_aux_loss

        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = sup_transl_loss
        self.loss['rot_loss'] = sup_rot_loss
        self.loss['depth_loss'] = depth_loss
        self.loss['edge_loss'] = edge_loss
        self.loss['mask_loss'] = mask_loss
        self.loss['pose_prior_loss'] = pose_prior_loss
        self.loss['sup_aux_loss'] = sup_aux_loss
        self.loss['valid_ratio'] = valid_ratio
        self.loss['valid_points_mean'] = valid_count.mean()
        self.loss['pred_transl_norm'] = transl_err.norm(dim=1).mean()
        self.loss['pred_rot_norm_dev'] = (rot_err.norm(dim=1) - 1.0).abs().mean()
        return self.loss

