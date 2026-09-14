import time
import numpy as np
import torch
import viser
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation
from initialization import C0, CAMS, DATA, to_covariance, compute_c2w_opencv

def load_gaussians(path):
    """Loads the saved raw params and applies the same activations as training."""
    params = torch.load(path, weights_only=True)
    means = params["means"].numpy()                                                 # N x 3
    # clipped to 1 as well: matplotlib rejects colours above 1
    colors = torch.clamp(0.5 + C0 * params["sh_dc"][:, 0, :], 0.0, 1.0).numpy()    # N x 3
    opacity = torch.sigmoid(params["logit_opacity"][:, 0]).numpy()                  # N
    scales = torch.exp(params["log_scales"]).numpy()                                # N x 3
    quaternions = torch.nn.functional.normalize(params["quats"], p=2, dim=1).numpy()  # N x 4
    return means, colors, opacity, scales, quaternions

def visualize(path="optimized_gaussians.pt", min_opacity=0.05, size=1.0):
    means, colors, opacity, scales, _ = load_gaussians(path)
    # camera centres: the translation column is the same in OpenGL and OpenCV poses
    cams = DATA["poses"][:, :3, 3]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(bottom=0.18)

    # equal aspect: same limits on every axis, otherwise the object looks squashed
    half = np.abs(cams).max() * 1.05
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    ax.scatter(*cams[CAMS:].T, c="lightgray", s=8, label="unused cameras")
    ax.scatter(*cams[:CAMS].T, c="red", s=30, label=f"training cameras ({CAMS})")
    ax.legend(loc="upper left")

    # s is in screen points^2, not world units: convert the largest axis of each
    # gaussian with a rough points-per-world-unit factor. Sizes won't follow zoom.
    points_per_unit = fig.get_figwidth() * 72 * 0.6 / (2 * half)
    diameter = 2 * scales.max(axis=1) * points_per_unit                            # N, in points

    state = {"scatter": None}
    def draw(min_opacity, size):
        if state["scatter"] is not None:
            state["scatter"].remove()
        keep = opacity > min_opacity
        # per-point transparency goes in the 4th column of c; alpha= is one value for all
        rgba = np.concatenate([colors[keep], opacity[keep, None]], axis=1)
        state["scatter"] = ax.scatter(
            *means[keep].T, c=rgba, s=(size * diameter[keep]) ** 2, depthshade=False
        )
        ax.set_title(f"{keep.sum()} / {len(means)} gaussians with opacity > {min_opacity:.2f}")
        fig.canvas.draw_idle()

    opacity_slider = Slider(fig.add_axes([0.2, 0.08, 0.6, 0.03]), "min opacity", 0.0, 1.0, valinit=min_opacity)
    size_slider = Slider(fig.add_axes([0.2, 0.03, 0.6, 0.03]), "size", 0.1, 3.0, valinit=size)
    opacity_slider.on_changed(lambda _: draw(opacity_slider.val, size_slider.val))
    size_slider.on_changed(lambda _: draw(opacity_slider.val, size_slider.val))

    draw(min_opacity, size)
    plt.show()
    return fig

def viser_visualize(path="optimized_gaussians.pt", min_opacity=0.05):
    """Real splat rendering in the browser. Open the URL printed in the terminal."""
    means, colors, opacity, scales, quaternions = load_gaussians(path)
    covariances = to_covariance(torch.from_numpy(quaternions), torch.from_numpy(np.log(scales))).numpy()  # N x 3 x 3

    server = viser.ViserServer()

    # adding a node under an existing name replaces it: that's how the slider updates the splats
    def show_splats(min_opacity):
        keep = opacity > min_opacity
        server.scene.add_gaussian_splats(
            "/gaussians",
            centers=means[keep],
            covariances=covariances[keep],
            rgbs=colors[keep],
            opacities=opacity[keep, None],  # viser wants N x 1
        )

    slider = server.gui.add_slider("min opacity", min=0.0, max=1.0, step=0.01, initial_value=min_opacity)
    slider.on_update(lambda _: show_splats(slider.value))
    show_splats(min_opacity)

    # training cameras as frustums showing their image; viser cameras are OpenCV, like our poses
    H, W = DATA["images"].shape[1:3]
    fov = 2 * np.arctan(H / (2 * float(DATA["focal"])))  # vertical, radians
    for i, c2w in enumerate(compute_c2w_opencv(DATA)):
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=fov,
            aspect=W / H,
            scale=0.15,
            image=(DATA["images"][i] * 255).astype(np.uint8),
            wxyz=Rotation.from_matrix(c2w[:3, :3]).as_quat(scalar_first=True),
            position=c2w[:3, 3],
        )

    # the viewer lives as long as this script does
    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    viser_visualize()
