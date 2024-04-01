import json
import os

import imageio
import numpy as np
import torch

from load_scalarflow import pose_spherical


info_json_path = "data/ScalarReal/info.json"

split = "train"

all_poses = []

with open(info_json_path, "r") as fp:
    # read render settings
    meta = json.load(fp)
    near = float(meta["near"])
    far = float(meta["far"])
    radius = (near + far) * 0.5
    phi = float(meta["phi"])
    rotZ = meta["rot"] == "Z"
    r_center = np.float32(meta["render_center"])

    # read scene data
    voxel_tran = np.float32(meta["voxel_matrix"])
    # swap_zx
    voxel_tran = np.stack([voxel_tran[:, 2], voxel_tran[:, 1], voxel_tran[:, 0], voxel_tran[:, 3]], axis=1)
    voxel_scale = np.broadcast_to(meta["voxel_scale"], [3])

    # read video frames
    # all videos should be synchronized, having the same frame_rate and frame_num

    video_list = meta[split + "_videos"] if (split + "_videos") in meta else meta["train_videos"][0:1]

    for video_id, train_video in enumerate(video_list):

        camera_hw = train_video["camera_hw"]
        H, W = camera_hw
        camera_angle_x = float(train_video["camera_angle_x"])
        Focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        all_poses.append(np.array(train_video["transform_matrix"]).astype(np.float32))


poses = np.stack(all_poses, 0)  # [V, 4, 4]
hwf = np.float32([H, W, Focal])

# set render settings:
sp_n = 20  # an even number!
sp_poses = [
    pose_spherical(angle, phi, radius, rotZ, r_center[0], r_center[1], r_center[2])
    for angle in np.linspace(-180, 180, sp_n + 1)[:-1]
]
render_poses = torch.stack(sp_poses, 0)  # [sp_poses[36]]*sp_n, for testing a single pose
render_timesteps = np.arange(sp_n) / (sp_n - 1)

# poses, hwf, render_poses, render_timesteps, voxel_tran, voxel_scale, near, far
print("poses", poses.shape)
print("hwf", hwf.shape)
print("render_poses", render_poses.shape)
print("render_timesteps", render_timesteps.shape)
print("voxel_tran", voxel_tran.shape)
print("voxel_scale", voxel_scale.shape)
print("near", near)
print("far", far)

poses_path = "data/ScalarReal/poses.npy"
inv_poses_path = "data/ScalarReal/inv_poses.npy"
sp_poses_path = "data/ScalarReal/sp_poses.npy"
np.save(poses_path, poses)
np.save(sp_poses_path, render_poses.numpy())

inv_poses = np.linalg.inv(poses)
np.save(inv_poses_path, inv_poses)

print(poses[0, :, :])
# print(poses[0, -1, :])

print(inv_poses[0, :, :])

poses_bounds_hwf = hwf.reshape([3, 1])
poses_bounds_poses = poses[:, :3, :4]  # [V, 3, 4]

poses_bounds_poses = np.concatenate(
    [poses_bounds_poses, np.tile(poses_bounds_hwf[np.newaxis, ...], [poses_bounds_poses.shape[0], 1, 1])], 2
)

print(poses_bounds_poses.shape)

# must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
# must switch to [-u, r, -t] from [r, u, -t]
poses_bounds_poses = np.concatenate(
    [
        poses_bounds_poses[:, 1:2, :],
        poses_bounds_poses[:, 0:1, :],
        poses_bounds_poses[:, 2:3, :],
        poses_bounds_poses[:, 3:4, :],
        poses_bounds_poses[:, 4:5, :],
    ],
    1,
)
print(poses_bounds_poses.shape)
poses_bounds_poses = poses_bounds_poses.reshape([poses_bounds_poses.shape[0], 15])
near_depth = near
far_depth = far
poses_bounds_depth = np.array([near_depth, far_depth])
poses_bounds = np.concatenate(
    [poses_bounds_poses, np.tile(poses_bounds_depth[np.newaxis, :], [poses_bounds_poses.shape[0], 1])], 1
)

print(poses_bounds.shape)

poses_bounds_path = "data/ScalarReal/poses_bounds.npy"
np.save(poses_bounds_path, poses_bounds)
