# Gaussian Splatting Lab

A from-scratch implementation of [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), written to understand how it works. No CUDA kernels and no splatting libraries: the whole pipeline, rasterizer included, is plain PyTorch.

Trained on the `tiny_nerf` Lego scene (106 views, 100×100).

## Pipeline

1. **Init** (`initialization.py`): random Gaussians in a sphere. Scale from the distance to the 3 nearest neighbours, identity rotation, opacity 0.1, colour from reprojecting each Gaussian into the training images.
2. **Covariance**: quaternion + log-scales → `Σ = R S Sᵀ Rᵀ`.
3. **Projection** (`gaussian_operations.py`): world → camera (OpenCV convention), then the EWA Jacobian `Σ₂D = J Σ Jᵀ`, plus a 0.3 px low-pass dilation.
4. **Bookkeeping**: conic (inverse 2D covariance), 3σ radius, pixel bounding boxes, depth sort.
5. **Rasterizer** (`rasterizer.py`): fully vectorized front-to-back alpha compositing. Every (splat, pixel) pair goes into one flat list sorted by pixel and depth, and transmittance comes from a cumulative sum of `log(1 − α)`, computed in float64. No Python loop over splats and no in-place updates, so autograd can differentiate through it.
6. **Training** (`optimize.py`): L1 loss, Adam with a separate learning rate per parameter group, one random view per step.
7. **Viewer** (`visualizer.py`): interactive splat rendering in the browser with [viser](https://viser.studio), plus the training cameras shown as frustums.

Parameters are stored unconstrained (`log_scales`, `logit_opacity`, `quats` in wxyz, `sh_dc`) and activated at use time.

## Setup

```bash
pip install torch numpy scipy matplotlib tqdm viser
```

Download [`tiny_nerf_data.npz`](http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz) into `data/`.

On Windows with conda, PyTorch and NumPy can load two copies of OpenMP and crash with `OMP: Error #15`. Workaround:

```bash
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
```

## Usage

Run both from the project folder:

```bash
python optimize.py      # trains for 3000 steps, saves optimized_gaussians.pt
python visualizer.py    # opens the viewer at http://localhost:8080
```

## Results

- Training on only 4 views memorizes those images (L1 0.009) but fails on unseen views (L1 0.090).
- Training on all 106 views reaches L1 ≈ 0.025 uniformly across views.
- The vectorized rasterizer matches a reference per-splat loop to ~1e-6 in both image and gradients, at ~0.07 s per forward + backward pass instead of ~29 s.

## Not implemented (yet)

- Densification and pruning (clone, split, opacity reset)
- Higher-order spherical harmonics (view-dependent colour)
- SSIM term in the loss, learning-rate schedules
