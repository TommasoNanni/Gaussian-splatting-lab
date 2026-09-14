import torch

def rasterize(
        means_2d,
        inv_covariances_2d,
        colors,
        opacities,
        order,
        alive,
        bboxes,
        height,
        width,
        background = None,
    ):
    """
    Renders ONE view. No python loop over splats and no in-place updates,
    so autograd can differentiate through it.
    means_2d: N x 2
    inv_covariances_2d: N x 2 x 2
    colors: N x 3
    opacities: N
    order: N, nearest first
    alive: N
    bboxes: N x 4, packed as (x0, y0, x1, y1)
    background: 3 (or H x W x 3), None for black
    returns: H x W x 3
    """
    # step 1: one entry per (splat, pixel) pair, already in depth order
    gid, px, py, pix = expand_pairs(order, alive, bboxes, width)
    # step 2: group pairs by pixel, keeping depth order inside each pixel
    gid, px, py, pix = sort_pairs(gid, px, py, pix)
    # step 3: gaussian opacity of every pair
    alpha = pair_alpha(means_2d, inv_covariances_2d, opacities, gid, px, py)
    # step 4: transmittance in front of every pair
    T, T_final = pair_transmittance(alpha, pix, height * width)
    # step 5: sum alpha * T * color into the pixels
    image = composite(alpha, T, colors, gid, pix, height * width).view(height, width, 3)
    if background is not None:
        image = image + T_final.view(height, width, 1) * background
    return image

def expand_pairs(order, alive, bboxes, width):
    idx = order[alive[order]]                                   # M, alive gaussian ids, nearest first
    x0, y0, x1, y1 = bboxes[idx].unbind(-1)                     # M each
    w, h = x1 - x0, y1 - y0
    sizes = w * h                                               # pixels per splat
    # rank = position in idx = depth rank; repeated once per pixel of the splat
    rank = torch.repeat_interleave(torch.arange(idx.shape[0], device=idx.device), sizes)  # P
    # local pixel index inside the bbox, row by row
    start = torch.cumsum(sizes, 0) - sizes
    k = torch.arange(rank.shape[0], device=idx.device) - start[rank]
    px = x0[rank] + k % w[rank]                                 # column
    py = y0[rank] + k // w[rank]                                # row
    pix = py * width + px                                       # flat pixel index
    return idx[rank], px, py, pix

def sort_pairs(gid, px, py, pix):
    # pairs come out of expand_pairs in depth order; a STABLE sort by pixel keeps
    # that order among pairs of the same pixel
    perm = torch.argsort(pix, stable=True)
    return gid[perm], px[perm], py[perm], pix[perm]

def pair_alpha(means_2d, inv_covariances_2d, opacities, gid, px, py):
    dx = px.float() - means_2d[gid, 0]
    dy = py.float() - means_2d[gid, 1]
    a = inv_covariances_2d[gid, 0, 0]
    b = inv_covariances_2d[gid, 0, 1]
    c = inv_covariances_2d[gid, 1, 1]
    power = -0.5 * (a * dx**2 + 2 * b * dx * dy + c * dy**2)
    return torch.clamp(opacities[gid] * torch.exp(torch.clamp(power, max = 0.0)), max = 0.99)

def pair_transmittance(alpha, pix, num_pixels):
    # T_i = prod_{j<i} (1 - alpha_j) = exp(sum_{j<i} log(1 - alpha_j))
    # float64: the cumsum runs over millions of pairs, float32 loses ~1% of the gradient
    log_1m = torch.log1p(-alpha.double())
    cs = torch.cumsum(log_1m, 0)
    # reset the running sum at the first pair of every pixel
    _, counts = torch.unique_consecutive(pix, return_counts = True)
    first = torch.cumsum(counts, 0) - counts
    offset = torch.repeat_interleave(cs[first] - log_1m[first], counts)
    T = torch.exp(cs - log_1m - offset).float()                 # excludes the pair itself
    # light left after every splat of the pixel; pixels with no pairs keep T = 1
    log_T_final = torch.zeros(num_pixels, dtype = torch.float64, device = alpha.device).index_add(0, pix, log_1m)
    return T, torch.exp(log_T_final).float()

def composite(alpha, T, colors, gid, pix, num_pixels):
    weights = (alpha * T)[:, None] * colors[gid]                # P x 3
    return torch.zeros(num_pixels, 3, dtype = weights.dtype, device = weights.device).index_add(0, pix, weights)
