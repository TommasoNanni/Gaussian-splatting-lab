import numpy as np
import torch

Z_NEAR = 1e-4

def to_cam_coordinates(mean, covariance, cam_to_world):
    """
    Transforms 3D points and their covariances to camera coordinates.
    mean: N x 3
    covariance: N x 3 x 3
    cam_to_world: V x 4 x 4
    """
    R_c = cam_to_world[:, :3, :3] # V x 3 x 3
    t_c = cam_to_world[:, :3, 3]  # V x 3
    mean_cam = torch.einsum("vij, vni -> vnj", R_c, mean[None, :, :] - t_c[:, None, :])  # V x 3
    covariance_cam = torch.einsum("vji, njk, vkl -> vnil", R_c, covariance, R_c)  # V x N x 3 x 3
    return mean_cam, covariance_cam

def gaussians_to_2d(means_cam, covariances, intrinsics):
    """
    Takes in input gaussians already in camera i coordinates
    means: V X N X 3
    covariances: V X N X 3 X 3
    intrinsics: V X 3 X 3
    """
    fx, fy = intrinsics[:, 0, 0, None], intrinsics[:, 1, 1, None]  # V x 1
    cx, cy = intrinsics[:, 0, 2, None], intrinsics[:, 1, 2, None]  # V x 1
    x, y, z = means_cam[..., 0], means_cam[..., 1], means_cam[..., 2]          # V x N
    # clamp with maximum, not abs: points behind the camera must not survive with a flipped sign
    z = torch.clamp(z, min=Z_NEAR)
    inv_z = 1.0 / z
    means_2d = torch.stack([fx * x * inv_z + cx,
                         fy * y * inv_z + cy], dim=-1)            # V x N x 2
    visible = z > Z_NEAR                                           # V x N, carry forward to cull

    # now covariances
    jac = torch.zeros((means_cam.shape[0], means_cam.shape[1], 2, 3), dtype=means_cam.dtype, device=means_cam.device)  # V x N x 2 x 3
    jac[:, :, 0, 0] = fx * inv_z
    jac[:, :, 0, 2] = -fx * x * inv_z ** 2
    jac[:, :, 1, 1] = fy * inv_z
    jac[:, :, 1, 2] = -fy * y * inv_z ** 2
    cov_2d = torch.einsum("vnik, vnkl, vnjl -> vnij", jac, covariances, jac)  # V x N x 2 x 2

    # also add a convolution signal for the rasterizer
    cov_2d[..., 0, 0] += 0.3
    cov_2d[..., 1, 1] += 0.3
    return means_2d, cov_2d, visible

def invert_2d(cov_2d):
    det = cov_2d[..., 0, 0]*cov_2d[..., 1, 1] - cov_2d[..., 0, 1]*cov_2d[..., 1, 0]
    inv_cov_2d = torch.zeros_like(cov_2d)
    inv_cov_2d[..., 0, 0] = cov_2d[..., 1,1] / det
    inv_cov_2d[..., 1, 1] = cov_2d[..., 0,0] / det
    inv_cov_2d[..., 0, 1] = -cov_2d[..., 0,1] / det
    inv_cov_2d[..., 1, 0] = -cov_2d[..., 1,0] / det
    return inv_cov_2d

def compute_gaussian_radius(cov_2d):
    """
    compute the radius of a 2D gaussian from its covariance matrix as the sqrt of the largest eigenvalue
    """
    mid = (cov_2d[..., 0, 0] + cov_2d[..., 1, 1])/2
    d = torch.sqrt(((cov_2d[..., 0, 0] - cov_2d[..., 1, 1])/2)**2 + cov_2d[..., 0, 1]*cov_2d[..., 1, 0])
    lambda_max = mid + d
    radius = torch.ceil(3*torch.sqrt(lambda_max))
    return radius
    
def obtain_bboxes(means_2d, radius, visible, images):
    """
    means_2d: V x N x 2
    radius: V x N
    visible: V x N
    images: V x H x W x 3
    """
    H, W = images.shape[1:3]
    # x is a column index -> clip to W; y is a row index -> clip to H
    x0 = torch.clamp(torch.ceil(means_2d[..., 0] - radius), 0, W).long()
    y0 = torch.clamp(torch.ceil(means_2d[..., 1] - radius), 0, H).long()
    x1 = torch.clamp(torch.floor(means_2d[..., 0] + radius) + 1, 0, W).long()
    y1 = torch.clamp(torch.floor(means_2d[..., 1] + radius) + 1, 0, H).long()
    alive = visible & (x0 < x1) & (y0 < y1)
    bboxes = torch.stack([x0, y0, x1, y1], dim=-1)  # V x N x 4, packed as (x0, y0, x1, y1)
    return bboxes, alive

def sort_gaussians(means_cam):
    z_axis = means_cam[..., 2]  # V x N
    order = torch.argsort(z_axis, dim=-1)  # V x N
    return order