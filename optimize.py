import torch
from tqdm import tqdm
from initialization import (
    initialize_gaussians, 
    compute_c2w_opencv, 
    compute_intrinsics_opencv, 
    to_covariance,
    DATA,
    C0,
)
from gaussian_operations import (
    to_cam_coordinates,
    gaussians_to_2d,
    invert_2d,
    sort_gaussians,
    compute_gaussian_radius,
    obtain_bboxes,
)
from rasterizer import rasterize
import torch.optim as optimzer

def optimize(
    iters: int = 1000,      
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    G = initialize_gaussians()
    params = convert_to_params(G, device)
    # fixed data: moved to the device once, not every iteration
    intrinsics_opencv = torch.tensor(compute_intrinsics_opencv(DATA), dtype=torch.float32, device=device)
    poses_opencv = torch.tensor(compute_c2w_opencv(DATA), dtype=torch.float32, device=device)
    images = torch.tensor(DATA["images"][:len(poses_opencv)], dtype=torch.float32, device=device)
    optimizer = optimzer.Adam([
        {"params": [params["means"]],         "lr": 1.6e-4},
        {"params": [params["log_scales"]],    "lr": 5e-3},
        {"params": [params["quats"]],         "lr": 1e-3},
        {"params": [params["logit_opacity"]], "lr": 5e-2},
        {"params": [params["sh_dc"]],         "lr": 2.5e-3},
    ])
    pbar = tqdm(range(iters), desc="Optimizing")
    for it in pbar:
        optimizer.zero_grad()
        view = torch.randint(0, len(poses_opencv), (1,)).item()
        gt_image = images[view:view+1]
        cov_3d = to_covariance(params["quats"], params["log_scales"])
        mean_cam, covariance_cam = to_cam_coordinates(
            params["means"], cov_3d, poses_opencv[view:view+1]
        )
        mean_2d, cov_2d, visible = gaussians_to_2d(
            mean_cam, covariance_cam, intrinsics_opencv[view:view+1]
        )
        # bookkeeping needs no gradients: compute it in torch on the GPU, untracked
        with torch.no_grad():
            ordered = sort_gaussians(mean_cam)
            radius = compute_gaussian_radius(cov_2d)
            bboxes, alive = obtain_bboxes(mean_2d, radius, visible, gt_image)
        inv_cov_2d = invert_2d(cov_2d)
        # activations: raw params -> actual colours in [0, inf) and opacities in (0, 1).
        # recomputed every iteration so gradients flow back into the raw params
        colors = torch.clamp(0.5 + C0 * params["sh_dc"][:, 0, :], min=0.0)   # N x 3
        opacities = torch.sigmoid(params["logit_opacity"][:, 0])              # N
        # the rasterizer renders a single view: [0] drops the V=1 axis
        image = rasterize(
            means_2d = mean_2d[0],
            inv_covariances_2d = inv_cov_2d[0],
            colors = colors,
            opacities = opacities,
            order = ordered[0],
            alive = alive[0],
            bboxes = bboxes[0],
            height = gt_image.shape[1],
            width = gt_image.shape[2],
            background = None,
        )
        loss = compute_loss(image, gt_image[0])
        if it % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        loss.backward()
        optimizer.step()
    torch.save({k: v.detach().cpu() for k, v in params.items()}, "optimized_gaussians.pt")


def convert_to_params(G, device):
    params = {}
    params["means"] = torch.tensor(G["means"], device=device, dtype=torch.float32, requires_grad=True)
    params["log_scales"] = torch.tensor(G["log_scales"], device=device, dtype=torch.float32, requires_grad=True)
    params["quats"] = torch.tensor(G["quats"], device=device, dtype=torch.float32, requires_grad=True)
    params["logit_opacity"] = torch.tensor(G["logit_opacity"], device=device, dtype=torch.float32, requires_grad=True)
    params["sh_dc"] = torch.tensor(G["sh_dc"], device=device, dtype=torch.float32, requires_grad=True)
    return params

def compute_loss(rendered, target):
    return torch.nn.functional.l1_loss(rendered, target)



if __name__ == "__main__":
    optimize(iters=3000)