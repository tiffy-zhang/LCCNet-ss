# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/utils.py

import math

import mathutils
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from torch.utils.data.dataloader import default_collate


def rotate_points(PC, R, T=None, inverse=True):
    if T is not None:
        R = R.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        RT = T*R
    else:
        RT=R.copy()
    if inverse:
        RT.invert_safe()
    RT = torch.tensor(RT, device=PC.device, dtype=torch.float)

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_points_torch(PC, R, T=None, inverse=True):
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_forward(PC, R, T=None):
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (torch.Tensor/mathutils.Euler): can be either:
            * (mathutils.Euler) euler angles of the rotation part, in this case T cannot be None
            * (torch.Tensor shape [4]) quaternion representation of the rotation part, in this case T cannot be None
            * (mathutils.Matrix shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
            * (torch.Tensor shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
        T (torch.Tensor/mathutils.Vector): Translation of the new pose, shape [3], or None (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    else:
        return rotate_points(PC, R, T, inverse=True)


def rotate_back(PC_ROTATED, R, T=None):
    """
    Inverse of :func:`~utils.rotate_forward`.
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC_ROTATED, R, T, inverse=False)
    else:
        return rotate_points(PC_ROTATED, R, T, inverse=False)


def invert_pose(R, T):
    """
    Given the 'sampled pose' (aka H_init), we want CMRNet to predict inv(H_init).
    inv(T*R) will be used as ground truth for the network.
    Args:
        R (mathutils.Euler): Rotation of 'sampled pose'
        T (mathutils.Vector): Translation of 'sampled pose'

    Returns:
        (R_GT, T_GT) = (mathutils.Quaternion, mathutils.Vector)
    """
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    RT = T * R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT


def merge_inputs(queries):
    point_clouds = []
    imgs = []
    reflectances = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'rgb' and key != 'reflectance'}
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
        if 'reflectance' in input:
            reflectances.append(input['reflectance'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = imgs
    if len(reflectances) > 0:
        returns['reflectance'] = reflectances
    return returns


def quaternion_from_matrix(matrix):
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr+1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()


def quatmultiply(q, r):
    """
    Multiply two quaternions
    Args:
        q (torch.Tensor/nd.ndarray): shape=[4], first quaternion
        r (torch.Tensor/nd.ndarray): shape=[4], second quaternion

    Returns:
        torch.Tensor: shape=[4], normalized quaternion q*r
    """
    t = torch.zeros(4, device=q.device)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t / t.norm()


def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    mat[3, 3] = 1.
    return mat


def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat


def quat2mat_batched(q):
    """
    Convert batched quaternions to batched homogeneous rotation matrices.
    Args:
        q (torch.Tensor): shape [B,4], quaternion in [w, x, y, z]

    Returns:
        torch.Tensor: shape [B,4,4]
    """
    if q.dim() != 2 or q.shape[1] != 4:
        raise TypeError("q must have shape [B,4]")

    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    mat = torch.zeros((q.shape[0], 4, 4), device=q.device, dtype=q.dtype)
    mat[:, 0, 0] = 1 - 2 * (y * y + z * z)
    mat[:, 0, 1] = 2 * (x * y - z * w)
    mat[:, 0, 2] = 2 * (x * z + y * w)

    mat[:, 1, 0] = 2 * (x * y + z * w)
    mat[:, 1, 1] = 1 - 2 * (x * x + z * z)
    mat[:, 1, 2] = 2 * (y * z - x * w)

    mat[:, 2, 0] = 2 * (x * z - y * w)
    mat[:, 2, 1] = 2 * (y * z + x * w)
    mat[:, 2, 2] = 1 - 2 * (x * x + y * y)
    mat[:, 3, 3] = 1.0
    return mat


def tvector2mat_batched(t):
    """
    Convert batched translation vectors to homogeneous transform matrices.
    Args:
        t (torch.Tensor): shape [B,3]

    Returns:
        torch.Tensor: shape [B,4,4]
    """
    if t.dim() != 2 or t.shape[1] != 3:
        raise TypeError("t must have shape [B,3]")

    mat = torch.eye(4, device=t.device, dtype=t.dtype).unsqueeze(0).repeat(t.shape[0], 1, 1)
    mat[:, 0, 3] = t[:, 0]
    mat[:, 1, 3] = t[:, 1]
    mat[:, 2, 3] = t[:, 2]
    return mat


def mat2xyzrpy(rotmatrix):
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (torch.Tensor/np.ndarray): [4x4] transformation matrix

    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin ( rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw], device=rotmatrix.device, dtype=rotmatrix.dtype)


def to_rotation_matrix(R, T):
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT


def compose_pose_from_prediction(transl_err, rot_err):
    """
    Compose homogeneous pose matrix/matrices from predicted translation and quaternion.
    Args:
        transl_err (torch.Tensor): [3] or [B,3]
        rot_err (torch.Tensor): [4] or [B,4]

    Returns:
        torch.Tensor: [4,4] or [B,4,4]
    """
    single_sample = transl_err.dim() == 1
    if single_sample:
        transl_err = transl_err.unsqueeze(0)
        rot_err = rot_err.unsqueeze(0)

    pose_mats = []
    for i in range(transl_err.shape[0]):
        r_mat = quat2mat(rot_err[i])
        t_mat = tvector2mat(transl_err[i])
        pose_mats.append(torch.mm(t_mat, r_mat))

    pose_mats = torch.stack(pose_mats, dim=0)
    return pose_mats[0] if single_sample else pose_mats


def compose_pose_from_prediction_batched(transl_err, rot_err):
    """
    Compose homogeneous pose matrices from batched translation and quaternion.
    Args:
        transl_err (torch.Tensor): [B,3]
        rot_err (torch.Tensor): [B,4]

    Returns:
        torch.Tensor: [B,4,4]
    """
    if transl_err.dim() != 2 or transl_err.shape[1] != 3:
        raise TypeError("transl_err must have shape [B,3]")
    if rot_err.dim() != 2 or rot_err.shape[1] != 4:
        raise TypeError("rot_err must have shape [B,4]")

    r_mat = quat2mat_batched(rot_err)
    t_mat = tvector2mat_batched(transl_err)
    return torch.bmm(t_mat, r_mat)


def transform_point_cloud_homogeneous(point_cloud, rt_matrix, inverse=False):
    """
    Apply rigid transform to homogeneous point cloud.
    Args:
        point_cloud (torch.Tensor): [4,N] or [N,4]
        rt_matrix (torch.Tensor): [4,4]
        inverse (bool): if True, apply inverse transform

    Returns:
        torch.Tensor: transformed point cloud in the same layout as input
    """
    if rt_matrix.shape != torch.Size([4, 4]):
        raise TypeError("rt_matrix must have shape [4,4]")

    if inverse:
        rt_matrix = rt_matrix.inverse()

    if point_cloud.shape[0] == 4:
        return torch.mm(rt_matrix, point_cloud)
    if point_cloud.shape[1] == 4:
        return torch.mm(rt_matrix, point_cloud.t()).t()
    raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")


def pad_point_clouds(point_clouds):
    """
    Pad a list of homogeneous point clouds to a single batched tensor.
    Args:
        point_clouds (list[torch.Tensor]): each tensor is [4,N] or [N,4]

    Returns:
        tuple: (padded_point_clouds [B,4,Nmax], valid_mask [B,Nmax])
    """
    if len(point_clouds) == 0:
        raise ValueError("point_clouds cannot be empty")

    normalized = []
    max_points = 0
    for point_cloud in point_clouds:
        if point_cloud.shape[0] == 4:
            current = point_cloud
        elif point_cloud.shape[1] == 4:
            current = point_cloud.t()
        else:
            raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
        normalized.append(current)
        max_points = max(max_points, current.shape[1])

    batch_size = len(normalized)
    device = normalized[0].device
    dtype = normalized[0].dtype
    padded = torch.zeros((batch_size, 4, max_points), device=device, dtype=dtype)
    valid_mask = torch.zeros((batch_size, max_points), device=device, dtype=torch.bool)

    for i, point_cloud in enumerate(normalized):
        num_points = point_cloud.shape[1]
        padded[i, :, :num_points] = point_cloud
        valid_mask[i, :num_points] = True

    return padded, valid_mask


def transform_point_cloud_homogeneous_batched(point_clouds, rt_matrices, inverse=False):
    """
    Apply rigid transforms to a batch of homogeneous point clouds.
    Args:
        point_clouds (torch.Tensor): [B,4,N] or [B,N,4]
        rt_matrices (torch.Tensor): [B,4,4]
        inverse (bool): if True, apply inverse transforms

    Returns:
        torch.Tensor: transformed point clouds with same layout as input
    """
    if rt_matrices.dim() != 3 or rt_matrices.shape[1:] != torch.Size([4, 4]):
        raise TypeError("rt_matrices must have shape [B,4,4]")

    if inverse:
        rt_matrices = torch.inverse(rt_matrices)

    if point_clouds.dim() != 3:
        raise TypeError("point_clouds must be a 3D tensor")

    if point_clouds.shape[1] == 4:
        return torch.bmm(rt_matrices, point_clouds)
    if point_clouds.shape[2] == 4:
        transformed = torch.bmm(rt_matrices, point_clouds.transpose(1, 2))
        return transformed.transpose(1, 2)
    raise TypeError("Point clouds must have shape [B,4,N] or [B,N,4]")


def project_points_torch(point_cloud, cam_intrinsic, image_shape, min_depth=1e-3, max_depth=None, eps=1e-8):
    """
    Project 3D points to image plane using pure torch operations.
    Args:
        point_cloud (torch.Tensor): [4,N], [N,4], [3,N], or [N,3]
        cam_intrinsic (torch.Tensor): [3,3]
        image_shape (tuple/list): [H, W] or [H, W, C]
        min_depth (float): minimum valid depth
        max_depth (float or None): optional maximum depth
        eps (float): numerical guard for division

    Returns:
        tuple: (uv [N,2], z [N], valid_mask [N])
    """
    if len(image_shape) < 2:
        raise TypeError("image_shape must contain at least [H, W]")
    h, w = int(image_shape[0]), int(image_shape[1])

    if not torch.is_tensor(cam_intrinsic):
        cam_intrinsic = torch.tensor(cam_intrinsic, dtype=point_cloud.dtype, device=point_cloud.device)
    else:
        cam_intrinsic = cam_intrinsic.to(device=point_cloud.device, dtype=point_cloud.dtype)

    if point_cloud.shape[0] in [3, 4]:
        xyz = point_cloud[:3, :].t()
    elif point_cloud.shape[1] in [3, 4]:
        xyz = point_cloud[:, :3]
    else:
        raise TypeError("Point cloud must have shape [4xN], [Nx4], [3xN], or [Nx3]")

    projected = torch.mm(xyz, cam_intrinsic.t())
    z = projected[:, 2]
    inv_z = 1.0 / (z + eps)
    uv = projected[:, :2] * inv_z.unsqueeze(1)

    valid = z > min_depth
    if max_depth is not None:
        valid = valid & (z < max_depth)
    valid = valid & (uv[:, 0] >= 0) & (uv[:, 0] < (w - 1))
    valid = valid & (uv[:, 1] >= 0) & (uv[:, 1] < (h - 1))

    return uv, z, valid


def project_points_torch_batched(point_clouds, cam_intrinsics, image_shape, point_mask=None,
                                 min_depth=1e-3, max_depth=None, eps=1e-8):
    """
    Project a batch of 3D point clouds to the image plane.
    Args:
        point_clouds (torch.Tensor): [B,4,N], [B,N,4], [B,3,N], or [B,N,3]
        cam_intrinsics (torch.Tensor): [B,3,3] or [3,3]
        image_shape (tuple/list): [H, W] or [H, W, C]
        point_mask (torch.Tensor or None): [B,N] valid points mask
        min_depth (float): minimum valid depth
        max_depth (float or None): optional maximum depth
        eps (float): numerical guard for division

    Returns:
        tuple: (uv [B,N,2], z [B,N], valid_mask [B,N])
    """
    if len(image_shape) < 2:
        raise TypeError("image_shape must contain at least [H, W]")
    h, w = int(image_shape[0]), int(image_shape[1])

    if point_clouds.dim() != 3:
        raise TypeError("point_clouds must be a 3D tensor")

    if point_clouds.shape[1] in [3, 4]:
        xyz = point_clouds[:, :3, :].transpose(1, 2)
    elif point_clouds.shape[2] in [3, 4]:
        xyz = point_clouds[:, :, :3]
    else:
        raise TypeError("Point clouds must have shape [B,4,N], [B,N,4], [B,3,N], or [B,N,3]")

    if cam_intrinsics.dim() == 2:
        cam_intrinsics = cam_intrinsics.unsqueeze(0).expand(xyz.shape[0], -1, -1)
    cam_intrinsics = cam_intrinsics.to(device=xyz.device, dtype=xyz.dtype)

    projected = torch.bmm(xyz, cam_intrinsics.transpose(1, 2))
    z = projected[:, :, 2]
    inv_z = 1.0 / (z + eps)
    uv = projected[:, :, :2] * inv_z.unsqueeze(-1)

    valid = z > min_depth
    if max_depth is not None:
        valid = valid & (z < max_depth)
    valid = valid & (uv[:, :, 0] >= 0) & (uv[:, :, 0] < (w - 1))
    valid = valid & (uv[:, :, 1] >= 0) & (uv[:, :, 1] < (h - 1))
    if point_mask is not None:
        valid = valid & point_mask

    return uv, z, valid


def rasterize_depth_torch(uv, z, image_shape, mode='bilinear', eps=1e-8):
    """
    Rasterize sparse projected depth to image grid with differentiable accumulation.
    Args:
        uv (torch.Tensor): [N,2] projected coordinates in pixels
        z (torch.Tensor): [N] depth values
        image_shape (tuple/list): [H, W] or [H, W, C]
        mode (str): only 'bilinear' is currently supported
        eps (float): numerical guard

    Returns:
        tuple: (depth_map [1,H,W], weight_map [1,H,W])
    """
    if mode != 'bilinear':
        raise ValueError("Only 'bilinear' mode is supported")
    if len(image_shape) < 2:
        raise TypeError("image_shape must contain at least [H, W]")

    h, w = int(image_shape[0]), int(image_shape[1])
    device = uv.device
    dtype = uv.dtype

    if uv.numel() == 0:
        depth_map = torch.zeros((1, h, w), device=device, dtype=dtype)
        weight_map = torch.zeros((1, h, w), device=device, dtype=dtype)
        return depth_map, weight_map

    x = uv[:, 0]
    y = uv[:, 1]

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    wx1 = x - x0
    wy1 = y - y0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    x0l = x0.long()
    y0l = y0.long()
    x1l = x1.long()
    y1l = y1.long()

    acc_depth = torch.zeros(h * w, device=device, dtype=dtype)
    acc_weight = torch.zeros(h * w, device=device, dtype=dtype)

    corners = [
        (x0l, y0l, wx0 * wy0),
        (x0l, y1l, wx0 * wy1),
        (x1l, y0l, wx1 * wy0),
        (x1l, y1l, wx1 * wy1),
    ]

    for xc, yc, weight in corners:
        valid = (xc >= 0) & (xc < w) & (yc >= 0) & (yc < h)
        if valid.any():
            idx = yc[valid] * w + xc[valid]
            wv = weight[valid]
            zv = z[valid]
            acc_weight.scatter_add_(0, idx, wv)
            acc_depth.scatter_add_(0, idx, wv * zv)

    depth = acc_depth / (acc_weight + eps)
    depth_map = depth.view(1, h, w)
    weight_map = acc_weight.view(1, h, w)
    return depth_map, weight_map


def rasterize_depth_torch_batched(uv, z, image_shape, valid_mask=None, mode='bilinear', eps=1e-8):
    """
    Rasterize a batch of sparse projected depths to image grids.
    Args:
        uv (torch.Tensor): [B,N,2] projected coordinates in pixels
        z (torch.Tensor): [B,N] depth values
        image_shape (tuple/list): [H, W] or [H, W, C]
        valid_mask (torch.Tensor or None): [B,N] valid points mask
        mode (str): only 'bilinear' is currently supported
        eps (float): numerical guard

    Returns:
        tuple: (depth_map [B,1,H,W], weight_map [B,1,H,W])
    """
    if mode != 'bilinear':
        raise ValueError("Only 'bilinear' mode is supported")
    if len(image_shape) < 2:
        raise TypeError("image_shape must contain at least [H, W]")

    h, w = int(image_shape[0]), int(image_shape[1])
    batch_size = uv.shape[0]
    device = uv.device
    dtype = uv.dtype

    if valid_mask is None:
        valid_mask = torch.ones_like(z, dtype=torch.bool)

    x = uv[:, :, 0]
    y = uv[:, :, 1]

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    wx1 = x - x0
    wy1 = y - y0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    x0l = x0.long()
    y0l = y0.long()
    x1l = x1.long()
    y1l = y1.long()

    acc_depth = torch.zeros(batch_size * h * w, device=device, dtype=dtype)
    acc_weight = torch.zeros(batch_size * h * w, device=device, dtype=dtype)
    batch_offsets = (torch.arange(batch_size, device=device).view(-1, 1) * (h * w)).expand(-1, uv.shape[1])

    corners = [
        (x0l, y0l, wx0 * wy0),
        (x0l, y1l, wx0 * wy1),
        (x1l, y0l, wx1 * wy0),
        (x1l, y1l, wx1 * wy1),
    ]

    for xc, yc, weight in corners:
        valid = valid_mask & (xc >= 0) & (xc < w) & (yc >= 0) & (yc < h)
        if valid.any():
            flat_idx = batch_offsets[valid] + yc[valid] * w + xc[valid]
            acc_weight.scatter_add_(0, flat_idx, weight[valid])
            acc_depth.scatter_add_(0, flat_idx, weight[valid] * z[valid])

    depth = acc_depth / (acc_weight + eps)
    depth_map = depth.view(batch_size, 1, h, w)
    weight_map = acc_weight.view(batch_size, 1, h, w)
    return depth_map, weight_map


def project_point_clouds_to_depth_batched(point_clouds, cam_intrinsics, image_shape,
                                          point_mask=None, min_depth=1e-3,
                                          max_depth=None, mode='bilinear'):
    """
    Project a batch of point clouds to depth maps.
    Returns depth, weight, valid_mask, uv, z, valid_points.
    """
    uv, z, valid_points = project_points_torch_batched(
        point_clouds, cam_intrinsics, image_shape,
        point_mask=point_mask, min_depth=min_depth, max_depth=max_depth
    )
    depth_map, weight_map = rasterize_depth_torch_batched(
        uv, z, image_shape, valid_mask=valid_points, mode=mode
    )
    valid_mask = build_validity_mask(
        depth_map,
        min_depth=min_depth,
        max_depth=max_depth,
        weight_map=weight_map,
        min_weight=1e-6,
    )
    return {
        'depth': depth_map,
        'weight': weight_map,
        'valid_mask': valid_mask,
        'uv': uv,
        'z': z,
        'valid_points': valid_points,
    }


def differentiable_reproject_depth_batched(point_clouds, pred_transl, pred_rot, cam_intrinsics,
                                           image_shape, point_mask=None, min_depth=1e-3,
                                           max_depth=None, mode='bilinear',
                                           apply_inverse_pose=True):
    """
    End-to-end differentiable batched depth reprojection utility.
    """
    rt_matrices = compose_pose_from_prediction(pred_transl, pred_rot)
    projected_point_clouds = transform_point_cloud_homogeneous_batched(
        point_clouds, rt_matrices, inverse=apply_inverse_pose
    )
    outputs = project_point_clouds_to_depth_batched(
        projected_point_clouds, cam_intrinsics, image_shape,
        point_mask=point_mask, min_depth=min_depth, max_depth=max_depth,
        mode=mode,
    )
    outputs['point_cloud_projected'] = projected_point_clouds
    return outputs


def build_validity_mask(depth_map, min_depth=1e-3, max_depth=None, weight_map=None, min_weight=0.0):
    """
    Build validity mask for rasterized depth map.
    Args:
        depth_map (torch.Tensor): [1,H,W] or [H,W]
        min_depth (float): lower depth bound
        max_depth (float or None): upper depth bound
        weight_map (torch.Tensor or None): [1,H,W] or [H,W]
        min_weight (float): optional occupancy threshold

    Returns:
        torch.Tensor: boolean mask with same spatial size as depth_map
    """
    mask = depth_map > min_depth
    if max_depth is not None:
        mask = mask & (depth_map < max_depth)
    if weight_map is not None:
        mask = mask & (weight_map > min_weight)
    return mask


def differentiable_reproject_depth(point_cloud, pred_transl, pred_rot, cam_intrinsic, image_shape,
                                   min_depth=1e-3, max_depth=None, mode='bilinear',
                                   apply_inverse_pose=True):
    """
    End-to-end differentiable depth reprojection utility.
    Args:
        point_cloud (torch.Tensor): [4,N] or [N,4] homogeneous points
        pred_transl (torch.Tensor): [3]
        pred_rot (torch.Tensor): [4]
        cam_intrinsic (torch.Tensor): [3,3]
        image_shape (tuple/list): [H,W] or [H,W,C]
        min_depth (float): lower depth bound
        max_depth (float or None): upper depth bound
        mode (str): rasterization mode
        apply_inverse_pose (bool): keep consistency with rotate_forward usage

    Returns:
        dict: depth, weight, valid_mask, uv, z, point_cloud_projected
    """
    rt = compose_pose_from_prediction(pred_transl, pred_rot)
    projected_pc = transform_point_cloud_homogeneous(point_cloud, rt, inverse=apply_inverse_pose)

    uv, z, valid_points = project_points_torch(
        projected_pc, cam_intrinsic, image_shape,
        min_depth=min_depth, max_depth=max_depth
    )

    depth_map, weight_map = rasterize_depth_torch(uv[valid_points], z[valid_points], image_shape, mode=mode)
    valid_mask = build_validity_mask(
        depth_map,
        min_depth=min_depth,
        max_depth=max_depth,
        weight_map=weight_map,
        min_weight=1e-6
    )

    return {
        'depth': depth_map,
        'weight': weight_map,
        'valid_mask': valid_mask,
        'uv': uv,
        'z': z,
        'valid_points': valid_points,
        'point_cloud_projected': projected_pc,
    }


def overlay_imgs(rgb, lidar, idx=0):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean
    lidar = lidar.clone()

    lidar[lidar == 0] = 1000.
    lidar = -lidar
    #lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    #lidar = lidar.squeeze()
    lidar = lidar[0][0]
    lidar = (lidar*255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
    #io.imshow(blended_img)
    #io.show()
    #plt.figure()
    #plt.imshow(blended_img)
    #io.imsave(f'./IMGS/{idx:06d}.png', blended_img)
    return blended_img
