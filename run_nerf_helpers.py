import json
import os
import sys

import cv2 as cv
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# from hash_encoding import HashEmbedder, SHEncoder

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]).cuda())
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return torch.mean(torch.square(pred - exact))


def batchify_query(inputs, query_function, batch_size=2**22):
    """
    args:
        inputs: [..., input_dim]
    return:
        outputs: [..., output_dim]
    """
    input_dim = inputs.shape[-1]
    input_shape = inputs.shape
    inputs = inputs.view(-1, input_dim)  # flatten all but last dim
    N = inputs.shape[0]
    outputs = []
    for i in range(0, N, batch_size):
        output = query_function(inputs[i : i + batch_size])
        if isinstance(output, tuple):
            output = output[0]
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    return outputs.view(*input_shape[:-1], -1)  # unflatten


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SirenNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=4, skips=[4], first_omega_0=30, hidden_omega_0=1):
        """ """
        super(SirenNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0, is_first=True)]
            + [SineLayer(W, W, omega_0=hidden_omega_0) for i in range(D - 1)]
        )

        self.output_linear = nn.Linear(W, output_ch)
        # with torch.no_grad():
        #     self.output_linear.weight.uniform_(-np.sqrt(6 / W) / hidden_omega_0,
        #                                         np.sqrt(6 / W) / hidden_omega_0)

    def forward(self, x):
        input_pts = x  # torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

        outputs = self.output_linear(h)
        return outputs


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, args, i=0):
    if i == -1:
        return nn.Identity(), 3
    elif i == 0:
        embed_kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }

        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj: eo.embed(x)
        out_dim = embedder_obj.out_dim
    # elif i == 1:
    #     embed = HashEmbedder(bounding_box=args.bounding_box, \
    #                          log2_hashmap_size=args.log2_hashmap_size, \
    #                          finest_resolution=args.finest_res)
    #     out_dim = embed.out_dim
    # elif i == 2:
    #     embed = SHEncoder()
    #     out_dim = embed.out_dim
    return embed, out_dim


# Small NeRF for Hash embeddings
class NeRFSmall(nn.Module):
    def __init__(
        self,
        num_layers=3,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=2,
        hidden_dim_color=16,
        input_ch=3,
        output_ch=1,
    ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.rgb = torch.nn.Parameter(torch.tensor([0.0]))

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = output_ch  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # self.color_net = []
        # for l in range(num_layers_color):
        #     if l == 0:
        #         in_dim = 1
        #     else:
        #         in_dim = hidden_dim_color

        #     if l == num_layers_color - 1:
        #         out_dim = 1
        #     else:
        #         out_dim = hidden_dim_color

        #     self.color_net.append(nn.Linear(in_dim, out_dim, bias=True))

    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = F.relu(h, inplace=True)

        sigma = h
        return sigma


class NeRFSmall_c(nn.Module):
    def __init__(
        self,
        num_layers=3,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=2,
        hidden_dim_color=16,
        input_ch=3,
    ):
        super(NeRFSmall_c, self).__init__()

        self.input_ch = input_ch
        self.rgb = torch.nn.Parameter(torch.tensor([0.0]))

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        self.color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 1
            else:
                out_dim = hidden_dim_color

            self.color_net.append(nn.Linear(in_dim, out_dim, bias=True))
        self.color_net = nn.ModuleList(self.color_net)

    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = F.relu(h, inplace=True)

        sigma = h
        color = self.color_net[0](sigma[..., 1:])
        for l in range(1, self.num_layers_color):
            color = F.relu(color, inplace=True)
            color = self.color_net[l](color)
        return sigma[..., :1], color


class NeRFSmall_bg(nn.Module):
    def __init__(
        self,
        num_layers=3,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=2,
        hidden_dim_color=16,
        input_ch=3,
    ):
        super(NeRFSmall_bg, self).__init__()

        self.input_ch = input_ch
        self.rgb = torch.nn.Parameter(torch.tensor([0.0]))

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = input_ch + geo_feat_dim  # 1 for sigma, 15 for SH features
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3  # RGB color channels
            else:
                out_dim = hidden_dim_color

            self.color_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.color_net = nn.ModuleList(self.color_net)

    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = F.relu(h, inplace=True)

        sigma = h[..., :1]
        geo_feat = h[..., 1:]

        # color network
        h_color = torch.cat([geo_feat, x], dim=-1)  # concatenate sigma and SH features
        for l in range(len(self.color_net)):
            h_color = self.color_net[l](h_color)
            if l < len(self.color_net) - 1:
                h_color = F.relu(h_color, inplace=True)

        color = torch.sigmoid(h_color)  # apply sigmoid activation to get color values in range [0, 1]

        return sigma, color


class NeRFSmallPotential(nn.Module):
    def __init__(
        self,
        num_layers=3,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=2,
        hidden_dim_color=16,
        input_ch=3,
        use_f=False,
    ):
        super(NeRFSmallPotential, self).__init__()

        self.input_ch = input_ch
        self.rgb = torch.nn.Parameter(torch.tensor([0.0]))

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = hidden_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.sigma_net = nn.ModuleList(sigma_net)
        self.out = nn.Linear(hidden_dim, 3, bias=True)
        self.use_f = use_f
        if use_f:
            self.out_f = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.out_f2 = nn.Linear(hidden_dim, 3, bias=True)

    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = F.relu(h, True)

        v = self.out(h)
        if self.use_f:
            f = self.out_f(h)
            f = F.relu(f, True)
            f = self.out_f2(f)
        else:
            f = v * 0
        return v, f


def save_quiver_plot(u, v, res, save_path, scale=0.00000002):
    """
    Args:
        u: [H, W], vel along x (W)
        v: [H, W], vel along y (H)
        res: resolution of the plot along the longest axis; if None, let step = 1
        save_path:
    """
    import matplotlib
    import matplotlib.pyplot as plt

    H, W = u.shape
    y, x = np.mgrid[0:H, 0:W]
    axis_len = max(H, W)
    step = 1 if res is None else axis_len // res
    xq = [i[::step] for i in x[::step]]
    yq = [i[::step] for i in y[::step]]
    uq = [i[::step] for i in u[::step]]
    vq = [i[::step] for i in v[::step]]

    uv_norm = np.sqrt(np.array(uq) ** 2 + np.array(vq) ** 2).max()
    short_len = min(H, W)
    matplotlib.rcParams["font.size"] = 10 / short_len * axis_len
    fig, ax = plt.subplots(figsize=(10 / short_len * W, 10 / short_len * H))
    q = ax.quiver(xq, yq, uq, vq, pivot="tail", angles="uv", scale_units="xy", scale=scale / step)
    ax.invert_yaxis()
    plt.quiverkey(q, X=0.6, Y=1.05, U=uv_norm, label=f"Max arrow length = {uv_norm:.2g}", labelpos="E")
    plt.savefig(save_path)
    plt.close()
    return


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="ij"
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_np_continuous(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    random_offset_i = np.random.uniform(0, 1, size=(H, W))
    random_offset_j = np.random.uniform(0, 1, size=(H, W))
    i = i + random_offset_i
    j = j + random_offset_j
    i = np.clip(i, 0, W - 1)
    j = np.clip(j, 0, H - 1)

    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d, i, j


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def sample_bilinear(img, xy):
    """
    Sample image with bilinear interpolation
    :param img: (T, V, H, W, 3)
    :param xy: (V, 2, H, W)
    :return: img: (T, V, H, W, 3)
    """
    T, V, H, W, _ = img.shape
    u, v = xy[:, 0], xy[:, 1]

    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    u_floor, v_floor = np.floor(u).astype(int), np.floor(v).astype(int)
    u_ceil, v_ceil = np.ceil(u).astype(int), np.ceil(v).astype(int)

    u_ratio, v_ratio = u - u_floor, v - v_floor
    u_ratio, v_ratio = u_ratio[None, ..., None], v_ratio[None, ..., None]

    bottom_left = img[:, np.arange(V)[:, None, None], v_floor, u_floor]
    bottom_right = img[:, np.arange(V)[:, None, None], v_floor, u_ceil]
    top_left = img[:, np.arange(V)[:, None, None], v_ceil, u_floor]
    top_right = img[:, np.arange(V)[:, None, None], v_ceil, u_ceil]

    bottom = (1 - u_ratio) * bottom_left + u_ratio * bottom_right
    top = (1 - u_ratio) * top_left + u_ratio * top_right

    interpolated = (1 - v_ratio) * bottom + v_ratio * top

    return interpolated


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def jacobian3D(x):
    # x, (b,)d,h,w,ch, pytorch tensor
    # return jacobian and curl

    dudx = x[:, :, :, 1:, 0] - x[:, :, :, :-1, 0]
    dvdx = x[:, :, :, 1:, 1] - x[:, :, :, :-1, 1]
    dwdx = x[:, :, :, 1:, 2] - x[:, :, :, :-1, 2]
    dudy = x[:, :, 1:, :, 0] - x[:, :, :-1, :, 0]
    dvdy = x[:, :, 1:, :, 1] - x[:, :, :-1, :, 1]
    dwdy = x[:, :, 1:, :, 2] - x[:, :, :-1, :, 2]
    dudz = x[:, 1:, :, :, 0] - x[:, :-1, :, :, 0]
    dvdz = x[:, 1:, :, :, 1] - x[:, :-1, :, :, 1]
    dwdz = x[:, 1:, :, :, 2] - x[:, :-1, :, :, 2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = torch.cat((dudx, torch.unsqueeze(dudx[:, :, :, -1], 3)), 3)
    dvdx = torch.cat((dvdx, torch.unsqueeze(dvdx[:, :, :, -1], 3)), 3)
    dwdx = torch.cat((dwdx, torch.unsqueeze(dwdx[:, :, :, -1], 3)), 3)

    dudy = torch.cat((dudy, torch.unsqueeze(dudy[:, :, -1, :], 2)), 2)
    dvdy = torch.cat((dvdy, torch.unsqueeze(dvdy[:, :, -1, :], 2)), 2)
    dwdy = torch.cat((dwdy, torch.unsqueeze(dwdy[:, :, -1, :], 2)), 2)

    dudz = torch.cat((dudz, torch.unsqueeze(dudz[:, -1, :, :], 1)), 1)
    dvdz = torch.cat((dvdz, torch.unsqueeze(dvdz[:, -1, :, :], 1)), 1)
    dwdz = torch.cat((dwdz, torch.unsqueeze(dwdz[:, -1, :, :], 1)), 1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy

    j = torch.stack([dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz], -1)
    c = torch.stack([u, v, w], -1)

    return j, c


def curl2D(x, data_format="NHWC"):
    assert data_format == "NHWC"
    u = x[:, 1:, :, 0] - x[:, :-1, :, 0]  # ds/dy
    v = x[:, :, :-1, 0] - x[:, :, 1:, 0]  # -ds/dx,
    u = torch.cat([u, u[:, -1:, :]], dim=1)
    v = torch.cat([v, v[:, :, -1:]], dim=2)
    c = tf.stack([u, v], dim=-1)  # type: ignore
    return c


def curl3D(x, data_format="NHWC"):
    assert data_format == "NHWC"
    # x: bzyxc
    # dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:, :, :, 1:, 1] - x[:, :, :, :-1, 1]  #
    dwdx = x[:, :, :, 1:, 2] - x[:, :, :, :-1, 2]  #
    dudy = x[:, :, 1:, :, 0] - x[:, :, :-1, :, 0]  #
    # dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:, :, 1:, :, 2] - x[:, :, :-1, :, 2]  #
    dudz = x[:, 1:, :, :, 0] - x[:, :-1, :, :, 0]  #
    dvdz = x[:, 1:, :, :, 1] - x[:, :-1, :, :, 1]  #
    # dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # dudx = torch.cat((dudx, dudx[:,:,:,-1]), dim=3)
    dvdx = torch.cat((dvdx, dvdx[:, :, :, -1:]), dim=3)  #
    dwdx = torch.cat((dwdx, dwdx[:, :, :, -1:]), dim=3)  #

    dudy = torch.cat((dudy, dudy[:, :, -1:, :]), dim=2)  #
    # dvdy = torch.cat((dvdy, dvdy[:,:,-1:,:]), dim=2)
    dwdy = torch.cat((dwdy, dwdy[:, :, -1:, :]), dim=2)  #

    dudz = torch.cat((dudz, dudz[:, -1:, :, :]), dim=1)  #
    dvdz = torch.cat((dvdz, dvdz[:, -1:, :, :]), dim=1)  #
    # dwdz = torch.cat((dwdz, dwdz[:,-1:,:,:]), dim=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy

    # j = tf.stack([
    #       dudx,dudy,dudz,
    #       dvdx,dvdy,dvdz,
    #       dwdx,dwdy,dwdz
    # ], dim=-1)
    # curl = dwdy-dvdz,dudz-dwdx,dvdx-dudy
    c = torch.stack([u, v, w], dim=-1)

    return c


def jacobian3D_np(x):
    # x, (b,)d,h,w,ch
    # return jacobian and curl

    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:, :, :, 1:, 0] - x[:, :, :, :-1, 0]
    dvdx = x[:, :, :, 1:, 1] - x[:, :, :, :-1, 1]
    dwdx = x[:, :, :, 1:, 2] - x[:, :, :, :-1, 2]
    dudy = x[:, :, 1:, :, 0] - x[:, :, :-1, :, 0]
    dvdy = x[:, :, 1:, :, 1] - x[:, :, :-1, :, 1]
    dwdy = x[:, :, 1:, :, 2] - x[:, :, :-1, :, 2]
    dudz = x[:, 1:, :, :, 0] - x[:, :-1, :, :, 0]
    dvdz = x[:, 1:, :, :, 1] - x[:, :-1, :, :, 1]
    dwdz = x[:, 1:, :, :, 2] - x[:, :-1, :, :, 2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = np.concatenate((dudx, np.expand_dims(dudx[:, :, :, -1], axis=3)), axis=3)
    dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:, :, :, -1], axis=3)), axis=3)
    dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:, :, :, -1], axis=3)), axis=3)

    dudy = np.concatenate((dudy, np.expand_dims(dudy[:, :, -1, :], axis=2)), axis=2)
    dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:, :, -1, :], axis=2)), axis=2)
    dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:, :, -1, :], axis=2)), axis=2)

    dudz = np.concatenate((dudz, np.expand_dims(dudz[:, -1, :, :], axis=1)), axis=1)
    dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:, -1, :, :], axis=1)), axis=1)
    dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:, -1, :, :], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy

    j = np.stack([dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz], axis=-1)
    c = np.stack([u, v, w], axis=-1)

    return j, c


def velLegendHSV(hsvin, is3D, lw=-1, constV=255):
    # hsvin: (b), h, w, 3
    # always overwrite hsvin borders [lw], please pad hsvin before hand
    # or fill whole hsvin (lw < 0)
    ih, iw = hsvin.shape[-3:-1]
    if lw <= 0:  # fill whole
        a_list, b_list = [range(ih)], [range(iw)]
    else:  # fill border
        a_list = [range(ih), range(lw), range(ih), range(ih - lw, ih)]
        b_list = [range(lw), range(iw), range(iw - lw, iw), range(iw)]
    for a, b in zip(a_list, b_list):
        for _fty in a:
            for _ftx in b:
                fty = _fty - ih // 2
                ftx = _ftx - iw // 2
                ftang = np.arctan2(fty, ftx) + np.pi
                ftang = ftang * (180 / np.pi / 2)
                # print("ftang,min,max,mean", ftang.min(), ftang.max(), ftang.mean())
                # ftang,min,max,mean 0.7031249999999849 180.0 90.3515625
                hsvin[..., _fty, _ftx, 0] = np.expand_dims(ftang, axis=-1)  # 0-360
                # hsvin[...,_fty,_ftx,0] = ftang
                hsvin[..., _fty, _ftx, 2] = constV
                if (not is3D) or (lw == 1):
                    hsvin[..., _fty, _ftx, 1] = 255
                else:
                    thetaY1 = 1.0 - ((ih // 2) - abs(fty)) / float(lw if (lw > 1) else (ih // 2))
                    thetaY2 = 1.0 - ((iw // 2) - abs(ftx)) / float(lw if (lw > 1) else (iw // 2))
                    fthetaY = max(thetaY1, thetaY2) * (0.5 * np.pi)
                    ftxY, ftyY = np.cos(fthetaY), np.sin(fthetaY)
                    fangY = np.arctan2(ftyY, ftxY)
                    fangY = fangY * (240 / np.pi * 2)  # 240 - 0
                    hsvin[..., _fty, _ftx, 1] = 255 - fangY
                    # print("fangY,min,max,mean", fangY.min(), fangY.max(), fangY.mean())
    # finished velLegendHSV.


def cubecenter(cube, axis, half=0):
    # cube: (b,)h,h,h,c
    # axis: 1 (z), 2 (y), 3 (x)
    reduce_axis = [a for a in [1, 2, 3] if a != axis]
    pack = np.mean(cube, axis=tuple(reduce_axis))  # (b,)h,c
    pack = np.sqrt(np.sum(np.square(pack), axis=-1) + 1e-6)  # (b,)h

    length = cube.shape[axis - 5]  # h
    weights = np.arange(0.5 / length, 1.0, 1.0 / length)
    if half == 1:  # first half
        weights = np.where(weights < 0.5, weights, np.zeros_like(weights))
        pack = np.where(weights < 0.5, pack, np.zeros_like(pack))
    elif half == 2:  # second half
        weights = np.where(weights > 0.5, weights, np.zeros_like(weights))
        pack = np.where(weights > 0.5, pack, np.zeros_like(pack))

    weighted = pack * weights  # (b,)h
    weiAxis = np.sum(weighted, axis=-1) / np.sum(pack, axis=-1) * length  # (b,)

    return weiAxis.astype(np.int32)  # a ceiling is included


def vel_uv2hsv(vel, scale=160, is3D=False, logv=False, mix=False):
    # vel: a np.float32 array, in shape of (?=b,) d,h,w,3 for 3D and (?=b,)h,w, 2 or 3 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good.
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use more slices to get a volumetric visualization if True, which is slow

    ori_shape = list(vel.shape[:-1]) + [3]  # (?=b,) d,h,w,3
    if is3D:
        new_range = list(range(len(ori_shape)))
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXvel = np.transpose(vel, z_new_range)

        _xm, _ym, _zm = (ori_shape[-2] - 1) // 2, (ori_shape[-3] - 1) // 2, (ori_shape[-4] - 1) // 2

        if mix:
            _xlist = [cubecenter(vel, 3, 1), _xm, cubecenter(vel, 3, 2)]
            _ylist = [cubecenter(vel, 2, 1), _ym, cubecenter(vel, 2, 2)]
            _zlist = [cubecenter(vel, 1, 1), _zm, cubecenter(vel, 1, 2)]
        else:
            _xlist, _ylist, _zlist = [_xm], [_ym], [_zm]

        hsv = []
        for _x, _y, _z in zip(_xlist, _ylist, _zlist):
            # print(_x, _y, _z)
            _x, _y, _z = np.clip([_x, _y, _z], 0, ori_shape[-2:-5:-1])
            _yz = YZXvel[..., _x, :]
            _yz = np.stack([_yz[..., 2], _yz[..., 0], _yz[..., 1]], axis=-1)
            _yx = YZXvel[..., _z, :, :]
            _yx = np.stack([_yx[..., 0], _yx[..., 2], _yx[..., 1]], axis=-1)
            _zx = YZXvel[..., _y, :, :, :]
            _zx = np.stack([_zx[..., 0], _zx[..., 1], _zx[..., 2]], axis=-1)
            # print(_yx.shape, _yz.shape, _zx.shape)

            # in case resolution is not a cube, (res,res,res)
            _yxz = np.concatenate([_yx, _yz], axis=-2)  # yz, yx, zx  # (?=b,),h,w+zdim,3

            if ori_shape[-3] < ori_shape[-4]:
                pad_shape = list(_yxz.shape)  # (?=b,),h,w+zdim,3
                pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
                _pad = np.zeros(pad_shape, dtype=np.float32)
                _yxz = np.concatenate([_yxz, _pad], axis=-3)
            elif ori_shape[-3] > ori_shape[-4]:
                pad_shape = list(_zx.shape)  # (?=b,),h,w+zdim,3
                pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

                _zx = np.concatenate([_zx, np.zeros(pad_shape, dtype=np.float32)], axis=-3)

            midVel = np.concatenate([_yxz, _zx], axis=-2)  # yz, yx, zx  # (?=b,),h,w*3,3
            hsv += [vel2hsv(midVel, True, logv, scale)]
        # remove depth dim, increase with zyx slices
        ori_shape[-3] = 3 * ori_shape[-2]
        ori_shape[-2] = ori_shape[-1]
        ori_shape = ori_shape[:-1]
    else:
        hsv = [vel2hsv(vel, False, logv, scale)]

    bgr = []
    for _hsv in hsv:
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape([-1] + ori_shape[-2:])
        if is3D:
            velLegendHSV(_hsv, is3D, lw=max(1, min(6, int(0.025 * ori_shape[-2]))), constV=255)
        _hsv = cv.cvtColor(_hsv, cv.COLOR_HSV2BGR)
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape(ori_shape)
        bgr += [_hsv]
    if len(bgr) == 1:
        bgr = bgr[0]
    else:
        bgr = bgr[0] * 0.2 + bgr[1] * 0.6 + bgr[2] * 0.2
    return bgr.astype(np.uint8)[::-1]  # flip Y


def vel2hsv(velin, is3D, logv, scale=None):  # 2D
    fx, fy = velin[..., 0], velin[..., 1]
    ori_shape = list(velin.shape[:-1]) + [3]
    if is3D:
        fz = velin[..., 2]
        ang = np.arctan2(fz, fx) + np.pi  # angXZ
        zxlen2 = fx * fx + fz * fz
        angY = np.arctan2(np.abs(fy), np.sqrt(zxlen2))
        v = np.sqrt(zxlen2 + fy * fy)
    else:
        v = np.sqrt(fx * fx + fy * fy)
        ang = np.arctan2(fy, fx) + np.pi

    if logv:
        v = np.log10(v + 1)

    hsv = np.zeros(ori_shape, np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    if is3D:
        hsv[..., 1] = 255 - angY * (240 / np.pi * 2)
    else:
        hsv[..., 1] = 255
    if scale is not None:
        hsv[..., 2] = np.minimum(v * scale, 255)
    else:
        hsv[..., 2] = v / max(v.max(), 1e-6) * 255.0
    return hsv


def den_scalar2rgb(den, scale=160, is3D=False, logv=False, mix=True):
    # den: a np.float32 array, in shape of (?=b,) d,h,w,1 for 3D and (?=b,)h,w,1 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good.
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use averaged value as a volumetric visualization if True, else show middle slice

    ori_shape = list(den.shape)
    if ori_shape[-1] != 1:
        ori_shape.append(1)
        den = np.reshape(den, ori_shape)

    if is3D:
        new_range = list(range(len(ori_shape)))
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXden = np.transpose(den, z_new_range)

        if not mix:
            _yz = YZXden[..., (ori_shape[-2] - 1) // 2, :]
            _yx = YZXden[..., (ori_shape[-4] - 1) // 2, :, :]
            _zx = YZXden[..., (ori_shape[-3] - 1) // 2, :, :, :]
        else:
            _yz = np.average(YZXden, axis=-2)
            _yx = np.average(YZXden, axis=-3)
            _zx = np.average(YZXden, axis=-4)
            # print(_yx.shape, _yz.shape, _zx.shape)

        # in case resolution is not a cube, (res,res,res)
        _yxz = np.concatenate([_yx, _yz], axis=-2)  # yz, yx, zx  # (?=b,),h,w+zdim,1

        if ori_shape[-3] < ori_shape[-4]:
            pad_shape = list(_yxz.shape)  # (?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
            _pad = np.zeros(pad_shape, dtype=np.float32)
            _yxz = np.concatenate([_yxz, _pad], axis=-3)
        elif ori_shape[-3] > ori_shape[-4]:
            pad_shape = list(_zx.shape)  # (?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

            _zx = np.concatenate([_zx, np.zeros(pad_shape, dtype=np.float32)], axis=-3)

        midDen = np.concatenate([_yxz, _zx], axis=-2)  # yz, yx, zx  # (?=b,),h,w*3,1
    else:
        midDen = den

    if logv:
        midDen = np.log10(midDen + 1)
    if scale is None:
        midDen = midDen / max(midDen.max(), 1e-6) * 255.0
    else:
        midDen = midDen * scale
    grey = np.clip(midDen, 0, 255)

    return grey.astype(np.uint8)[::-1]  # flip y



# functions to transfer between 4. world space and 2. simulation space,
# velocity are further scaled according to resolution as in mantaflow
def vel_world2smoke(Vworld, w2s, scale_vector, st_factor):
    _st_factor = torch.Tensor(st_factor).expand((3,))
    vel_rot = Vworld[..., None, :] * (w2s[:3, :3])
    vel_rot = torch.sum(vel_rot, -1)  # 4.world to 3.target
    vel_scale = vel_rot / (scale_vector) * _st_factor  # 3.target to 2.simulation
    return vel_scale


def vel_smoke2world(Vsmoke, s2w, scale_vector, st_factor):
    _st_factor = torch.Tensor(st_factor).expand((3,))
    vel_scale = Vsmoke * (scale_vector) / _st_factor  # 2.simulation to 3.target
    vel_rot = torch.sum(vel_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    return vel_rot


def pos_world2smoke(Pworld, w2s, scale_vector):
    pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3, :3]), -1)  # 4.world to 3.target
    pos_off = (w2s[:3, -1]).expand(pos_rot.shape)  # 4.world to 3.target
    new_pose = pos_rot + pos_off
    pos_scale = new_pose / (scale_vector)  # 3.target to 2.simulation
    return pos_scale


def off_smoke2world(Offsmoke, s2w, scale_vector):
    off_scale = Offsmoke * (scale_vector)  # 2.simulation to 3.target
    off_rot = torch.sum(off_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    return off_rot


def pos_smoke2world(Psmoke, s2w, scale_vector):
    pos_scale = Psmoke * (scale_vector)  # 2.simulation to 3.target
    pos_rot = torch.sum(pos_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    pos_off = (s2w[:3, -1]).expand(pos_rot.shape)  # 3.target to 4.world
    return pos_rot + pos_off


def get_voxel_pts(H, W, D, s2w, scale_vector, n_jitter=0, r_jitter=0.8):
    """Get voxel positions."""

    i, j, k = torch.meshgrid(torch.linspace(0, D - 1, D), torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
    pts = torch.stack([(k + 0.5) / W, (j + 0.5) / H, (i + 0.5) / D], -1)
    # shape D*H*W*3, value [(x,y,z)] , range [0,1]

    jitter_r = torch.Tensor([r_jitter / W, r_jitter / H, r_jitter / D]).float().expand(pts.shape)
    for i_jitter in range(n_jitter):
        off_i = torch.rand(pts.shape, dtype=torch.float) - 0.5
        # shape D*H*W*3, value [(x,y,z)] , range [-0.5,0.5]

        pts = pts + off_i * jitter_r

    return pos_smoke2world(pts, s2w, scale_vector)


def get_voxel_pts_offset(H, W, D, s2w, scale_vector, r_offset=0.8):
    """Get voxel positions."""

    i, j, k = torch.meshgrid(torch.linspace(0, D - 1, D), torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
    pts = torch.stack([(k + 0.5) / W, (j + 0.5) / H, (i + 0.5) / D], -1)
    # shape D*H*W*3, value [(x,y,z)] , range [0,1]

    jitter_r = torch.Tensor([r_offset / W, r_offset / H, r_offset / D]).expand(pts.shape)
    off_i = torch.rand([1, 1, 1, 3], dtype=torch.float) - 0.5
    # shape 1*1*1*3, value [(x,y,z)] , range [-0.5,0.5]
    pts = pts + off_i * jitter_r

    return pos_smoke2world(pts, s2w, scale_vector)


# from FFJORD github code
def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


class Voxel_Tool(object):

    def __get_tri_slice(self, _xm, _ym, _zm, _n=1):
        _yz = torch.reshape(self.pts[..., _xm : _xm + _n, :], (-1, 3))
        _zx = torch.reshape(self.pts[:, _ym : _ym + _n, ...], (-1, 3))
        _xy = torch.reshape(self.pts[_zm : _zm + _n, ...], (-1, 3))

        pts_mid = torch.cat([_yz, _zx, _xy], dim=0)
        npMaskXYZ = [np.zeros([self.D, self.H, self.W, 1], dtype=np.float32) for _ in range(3)]
        npMaskXYZ[0][..., _xm : _xm + _n, :] = 1.0
        npMaskXYZ[1][:, _ym : _ym + _n, ...] = 1.0
        npMaskXYZ[2][_zm : _zm + _n, ...] = 1.0
        return pts_mid, torch.tensor(np.clip(npMaskXYZ[0] + npMaskXYZ[1] + npMaskXYZ[2], 1e-6, 3.0))

    def __pad_slice_to_volume(self, _slice, _n, mode=0):
        # mode: 0, x_slice, 1, y_slice, 2, z_slice
        tar_shape = [self.D, self.H, self.W]
        in_shape = tar_shape[:]
        in_shape[-1 - mode] = _n
        fron_shape = tar_shape[:]
        fron_shape[-1 - mode] = (tar_shape[-1 - mode] - _n) // 2
        back_shape = tar_shape[:]
        back_shape[-1 - mode] = tar_shape[-1 - mode] - _n - fron_shape[-1 - mode]

        cur_slice = _slice.view(in_shape + [-1])
        front_0 = torch.zeros(fron_shape + [cur_slice.shape[-1]])
        back_0 = torch.zeros(back_shape + [cur_slice.shape[-1]])

        volume = torch.cat([front_0, cur_slice, back_0], dim=-2 - mode)
        return volume

    def __init__(self, smoke_tran, smoke_tran_inv, smoke_scale, D, H, W, middleView=None):
        self.s_s2w = torch.Tensor(smoke_tran).expand([4, 4])
        self.s_w2s = torch.Tensor(smoke_tran_inv).expand([4, 4])
        self.s_scale = torch.Tensor(smoke_scale).expand([3])
        self.D = D
        self.H = H
        self.W = W
        self.pts = get_voxel_pts(H, W, D, self.s_s2w, self.s_scale)
        self.pts_mid = None
        self.npMaskXYZ = None
        self.middleView = middleView
        if middleView is not None:
            _n = 1 if self.middleView == "mid" else 3
            _xm, _ym, _zm = (W - _n) // 2, (H - _n) // 2, (D - _n) // 2
            self.pts_mid, self.npMaskXYZ = self.__get_tri_slice(_xm, _ym, _zm, _n)

    def get_raw_at_pts(self, cur_pts, chunk=1024 * 32, use_viewdirs=False, network_query_fn=None, network_fn=None):
        input_shape = list(cur_pts.shape[0:-1])

        pts_flat = cur_pts.view(-1, 4)
        pts_N = pts_flat.shape[0]
        # Evaluate model
        all_raw = []
        viewdir_zeros = torch.zeros([chunk, 3], dtype=torch.float) if use_viewdirs else None
        for i in range(0, pts_N, chunk):
            pts_i = pts_flat[i : i + chunk]
            viewdir_i = viewdir_zeros[: pts_i.shape[0]] if use_viewdirs else None

            raw_i = network_query_fn(pts_i, viewdir_i, network_fn)
            all_raw.append(raw_i)

        raw = torch.cat(all_raw, 0).view(input_shape + [-1])
        return raw

    def get_density_flat(
        self, cur_pts, chunk=1024 * 32, use_viewdirs=False, network_query_fn=None, network_fn=None, getStatic=True
    ):
        flat_raw = self.get_raw_at_pts(cur_pts, chunk, use_viewdirs, network_query_fn, network_fn)
        den_raw = F.relu(flat_raw[..., -1:])
        returnStatic = getStatic and (flat_raw.shape[-1] > 4)
        if returnStatic:
            static_raw = F.relu(flat_raw[..., 3:4])
            return [den_raw, static_raw]
        return [den_raw]

    def get_velocity_flat(self, cur_pts, batchify_fn, chunk=1024 * 32, vel_model=None):
        pts_N = cur_pts.shape[0]
        world_v = []
        for i in range(0, pts_N, chunk):
            input_i = cur_pts[i : i + chunk]
            vel_i = batchify_fn(vel_model, chunk)(input_i)
            world_v.append(vel_i)
        world_v = torch.cat(world_v, 0)
        return world_v

    def get_density_and_derivatives(
        self, cur_pts, chunk=1024 * 32, use_viewdirs=False, network_query_fn=None, network_fn=None
    ):
        _den = self.get_density_flat(cur_pts, chunk, use_viewdirs, network_query_fn, network_fn, False)[0]
        # requires 1 backward passes
        # The minibatch Jacobian matrix of shape (N, D_y=1, D_x=4)
        jac = _get_minibatch_jacobian(_den, cur_pts)
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]  # (N,1)
        return _den, _d_x, _d_y, _d_z, _d_t

    def get_velocity_and_derivatives(self, cur_pts, chunk=1024 * 32, batchify_fn=None, vel_model=None):
        _vel = self.get_velocity_flat(cur_pts, batchify_fn, chunk, vel_model)
        # requires 3 backward passes
        # The minibatch Jacobian matrix of shape (N, D_y=3, D_x=4)
        jac = _get_minibatch_jacobian(_vel, cur_pts)
        _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]  # (N,3)
        return _vel, _u_x, _u_y, _u_z, _u_t

    def get_voxel_density_list(
        self, t=None, chunk=1024 * 32, use_viewdirs=False, network_query_fn=None, network_fn=None, middle_slice=False
    ):
        D, H, W = self.D, self.H, self.W
        # middle_slice, only for fast visualization of the middle slice
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]
        if t is not None:
            input_t = torch.ones([pts_N, 1]) * float(t)
            pts_flat = torch.cat([pts_flat, input_t], dim=-1)

        den_list = self.get_density_flat(pts_flat, chunk, use_viewdirs, network_query_fn, network_fn)

        return_list = []
        for den_raw in den_list:
            if middle_slice:
                # only for fast visualization of the middle slice
                _n = 1 if self.middleView == "mid" else 3
                _yzV, _zxV, _xyV = torch.split(den_raw, [D * H * _n, D * W * _n, H * W * _n], dim=0)
                mixV = (
                    self.__pad_slice_to_volume(_yzV, _n, 0)
                    + self.__pad_slice_to_volume(_zxV, _n, 1)
                    + self.__pad_slice_to_volume(_xyV, _n, 2)
                )
                return_list.append(mixV / self.npMaskXYZ)
            else:
                return_list.append(den_raw.view(D, H, W, 1))
        return return_list

    def get_voxel_velocity(self, deltaT, t, batchify_fn, chunk=1024 * 32, vel_model=None, middle_slice=False):
        # middle_slice, only for fast visualization of the middle slice
        D, H, W = self.D, self.H, self.W
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]
        if t is not None:
            input_t = torch.ones([pts_N, 1]) * float(t)
            pts_flat = torch.cat([pts_flat, input_t], dim=-1)

        world_v = self.get_velocity_flat(pts_flat, batchify_fn, chunk, vel_model)
        reso_scale = [self.W * deltaT, self.H * deltaT, self.D * deltaT]
        target_v = vel_world2smoke(world_v, self.s_w2s, self.s_scale, reso_scale)

        if middle_slice:
            _n = 1 if self.middleView == "mid" else 3
            _yzV, _zxV, _xyV = torch.split(target_v, [D * H * _n, D * W * _n, H * W * _n], dim=0)
            mixV = (
                self.__pad_slice_to_volume(_yzV, _n, 0)
                + self.__pad_slice_to_volume(_zxV, _n, 1)
                + self.__pad_slice_to_volume(_xyV, _n, 2)
            )
            target_v = mixV / self.npMaskXYZ
        else:
            target_v = target_v.view(D, H, W, 3)

        return target_v

    def save_voxel_den_npz(
        self,
        den_path,
        t,
        use_viewdirs=False,
        network_query_fn=None,
        network_fn=None,
        chunk=1024 * 32,
        save_npz=True,
        save_jpg=False,
        jpg_mix=True,
        noStatic=False,
    ):
        voxel_den_list = self.get_voxel_density_list(
            t, chunk, use_viewdirs, network_query_fn, network_fn, middle_slice=not (jpg_mix or save_npz)
        )
        head_tail = os.path.split(den_path)
        namepre = ["", "static_"]
        for voxel_den, npre in zip(voxel_den_list, namepre):
            voxel_den = voxel_den.detach().cpu().numpy()
            if save_jpg:
                jpg_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0] + ".jpg")
                imageio.imwrite(jpg_path, den_scalar2rgb(voxel_den, scale=None, is3D=True, logv=False, mix=jpg_mix))
            if save_npz:
                # to save some space
                npz_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0] + ".npz")
                voxel_den = np.float16(voxel_den)
                np.savez_compressed(npz_path, vel=voxel_den)
            if noStatic:
                break

    def save_voxel_vel_npz(
        self,
        vel_path,
        deltaT,
        t,
        batchify_fn,
        chunk=1024 * 32,
        vel_model=None,
        save_npz=True,
        save_jpg=False,
        save_vort=False,
    ):
        vel_scale = 160
        voxel_vel = (
            self.get_voxel_velocity(deltaT, t, batchify_fn, chunk, vel_model, middle_slice=not save_npz)
            .detach()
            .cpu()
            .numpy()
        )

        if save_jpg:
            jpg_path = os.path.splitext(vel_path)[0] + ".jpg"
            imageio.imwrite(jpg_path, vel_uv2hsv(voxel_vel, scale=vel_scale, is3D=True, logv=False))
        if save_npz:
            if save_vort and save_jpg:
                _, NETw = jacobian3D_np(voxel_vel)
                head_tail = os.path.split(vel_path)
                imageio.imwrite(
                    os.path.join(head_tail[0], "vort" + os.path.splitext(head_tail[1])[0] + ".jpg"),
                    vel_uv2hsv(NETw[0], scale=vel_scale * 5.0, is3D=True),
                )
            # to save some space
            voxel_vel = np.float16(voxel_vel)
            np.savez_compressed(vel_path, vel=voxel_vel)


# Velocity Model
class SIREN_vel(nn.Module):
    def __init__(self, D=6, W=128, input_ch=4, output_ch=3, skips=[], fading_fin_step=0, bbox_model=None):
        """
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.fading_step = 0
        self.fading_fin_step = fading_fin_step if fading_fin_step > 0 else 0
        self.bbox_model = bbox_model

        first_omega_0 = 30.0
        hidden_omega_0 = 1.0

        self.hid_linears = nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0)]
            + [
                (
                    SineLayer(W, W, omega_0=hidden_omega_0)
                    if i not in self.skips
                    else SineLayer(W + input_ch, W, omega_0=hidden_omega_0)
                )
                for i in range(D - 1)
            ]
        )

        final_vel_linear = nn.Linear(W, output_ch)

        self.vel_linear = final_vel_linear

    def update_fading_step(self, fading_step):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - vel_in_step)
        if fading_step >= 0:
            self.fading_step = fading_step

    def fading_wei_list(self):
        # try print(fading_wei_list()) for debug
        step_ratio = np.clip(float(self.fading_step) / float(max(1e-8, self.fading_fin_step)), 0, 1)
        ma = 1 + (self.D - 2) * step_ratio  # in range of 1 to self.D-1
        fading_wei_list = [np.clip(1 + ma - m, 0, 1) * np.clip(1 + m - ma, 0, 1) for m in range(self.D)]
        return fading_wei_list

    def print_fading(self):
        # w_list = self.fading_wei_list()
        # _str = ["h%d:%0.03f" % (i, w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        # print("; ".join(_str))
        pass

    def forward(self, x):
        h = x
        h_layers = []
        for i, l in enumerate(self.hid_linears):
            h = self.hid_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([x, h], -1)

        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step:
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w, y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w * y + h

        vel_out = self.vel_linear(h)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(x[..., :3])
            vel_out = torch.reshape(bbox_mask, [-1, 1]) * vel_out

        return vel_out
