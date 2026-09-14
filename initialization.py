import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

DATA = np.load("data/tiny_nerf_data.npz")
N = 10000 # number of initial gaussians
C0 = 0.28209479177387814
CAMS = 106

def visualize_images():
    images = DATA["images"]
    fig, ax = plt.subplots(1, CAMS, figsize=(12, 3))
    for i in range(CAMS):
        ax[i].imshow(images[i], interpolation="nearest")
        ax[i].set_title(f"Image {i + 1}")
        ax[i].axis("off")
    plt.tight_layout(); 
    plt.show()

def to_covariance(quaternions, log_scales):
    quaternions = quaternions / torch.norm(quaternions, dim = -1, keepdim=True) # N x 4
    s = torch.exp(log_scales)                                    # N x 3
    rotmat = torch.zeros((quaternions.shape[0], 3, 3), dtype=quaternions.dtype, device=quaternions.device)
    rotmat[:, 0, 0] = 1-2*(quaternions[:, 2]**2 + quaternions[:, 3]**2)
    rotmat[:, 1, 1] = 1-2*(quaternions[:, 1]**2 + quaternions[:, 3]**2)
    rotmat[:, 2, 2] = 1-2*(quaternions[:, 1]**2 + quaternions[:, 2]**2)
    rotmat[:, 0, 1] = 2*(quaternions[:, 1]*quaternions[:, 2] - quaternions[:, 0]*quaternions[:, 3])
    rotmat[:, 1, 0] = 2*(quaternions[:, 1]*quaternions[:, 2] + quaternions[:, 0]*quaternions[:, 3])
    rotmat[:, 0, 2] = 2*(quaternions[:, 1]*quaternions[:, 3] + quaternions[:, 0]*quaternions[:, 2])
    rotmat[:, 2, 0] = 2*(quaternions[:, 1]*quaternions[:, 3] - quaternions[:, 0]*quaternions[:, 2])
    rotmat[:, 1, 2] = 2*(quaternions[:, 2]*quaternions[:, 3] - quaternions[:, 0]*quaternions[:, 1])
    rotmat[:, 2, 1] = 2*(quaternions[:, 2]*quaternions[:, 3] + quaternions[:, 0]*quaternions[:, 1])

    M = rotmat * s[:, None, :]                                # R @ diag(s), column scaling
    cov = M @ M.transpose(1, 2)                            # R S S^T R^T
    return cov # N X 3 X 3

def compute_c2w_opencv(data):
    poses = data["poses"][:CAMS, ...]
    poses_opencv = poses.copy() 
    poses_opencv[..., :3, 1:3] *= -1
    return poses_opencv # CAMS x 4 x 4

def compute_intrinsics_opencv(data):
    intrinsics = data["focal"]
    images = data["images"][:CAMS,...]
    intrinsics_opencv = np.zeros((CAMS, 3, 3))
    intrinsics_opencv[..., 0, 0] = intrinsics
    intrinsics_opencv[..., 1, 1] = intrinsics
    intrinsics_opencv[..., 0, 2] = images.shape[2] / 2
    intrinsics_opencv[..., 1, 2] = images.shape[1] / 2
    intrinsics_opencv[..., 2, 2] = 1.0
    return intrinsics_opencv # CAMS x 3 x 3

def project(points, c2w, intrinsics):
    R, t = c2w[:, :3, :3], c2w[:, :3, 3]
    # world -> camera: X_cam = R^T (X_world - t). "vji" is the transpose of R.
    X = np.einsum("vji, vnj -> vni", R, points - t[:, None, :])
    z = X[..., 2]
    inv_z = 1.0 / np.where(np.abs(z) > 1e-7, z, 1e-7)
    fx, fy = intrinsics[:, 0, 0, None], intrinsics[:, 1, 1, None]
    cx, cy = intrinsics[:, 0, 2, None], intrinsics[:, 1, 2, None]
    u = fx * X[...,0]*inv_z + cx
    v = fy * X[...,1]*inv_z + cy
    return np.stack([u, v], axis=-1), z # CAMS x N x 2, CAMS x N

def initialize_gaussians():
    R_max = 1.3
    # initializing the position of gaussians
    g = np.random.normal(size=(N, 3))
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    r = R_max * np.random.uniform(size=(N, 1)) ** (1 / 3)
    means = g * r
    # initializing the covariance of gaussians
    d, _ = cKDTree(means).query(means, k=4)
    sigma = np.mean(d[:, 1:], axis = 1, keepdims=True)
    s_hat = np.log(np.clip(sigma, 1e-7, None)).repeat(3, axis=1)
    quats = np.zeros((N, 4))
    quats[:, 0] = 1.0
    # initializing the opacity of gaussians
    alpha = np.full((N,1), np.log(0.1/0.9))
    # init colors
    uv, z = project(
        means[None, ...],
        compute_c2w_opencv(DATA),
        compute_intrinsics_opencv(DATA)
    )
    # we need a pixel index to sample the color from the image, so we floor the uv coordinates to get the pixel index
    ui = np.floor(uv[..., 0]).astype(int)
    vi = np.floor(uv[..., 1]).astype(int)
    # verify that the projected points are within the image bounds and in front of the camera
    ok = (z > 0) & (ui >= 0) & (ui < DATA["images"].shape[2]) & (vi >= 0) & (vi < DATA["images"].shape[1])
    # accumulator that sums colors and counter to check how many views partecipateed
    acc = np.zeros((N,3))
    cnt = np.zeros((N,1))
    for i in range(CAMS):
        m = ok[i]
        acc[m] += DATA["images"][i, vi[i,m], ui[i,m], :3]
        cnt[m] += 1.0
    rgb = np.where(cnt>0, acc / np.maximum(cnt,1) , 0.5)
    sh_dc = ((rgb - 0.5)/C0)[:, None, :]
    return {
        "means": means,     # N x 3      world coords
        "log_scales": s_hat, # N x 3     exp() at use time
        "quats": quats,     # N x 4      wxyz, normalize at use time
        "logit_opacity": alpha, # N x 1  sigmoid() at use time
        "sh_dc": sh_dc,     # N x 1 x 3  color = 0.5 + C0 * sh_dc
    }