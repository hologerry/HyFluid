import json
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F


# fmt: off
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()
# fmt: on


def pose_spherical(theta, phi, radius, rotZ=True, wx=0.0, wy=0.0, wz=0.0):
    # spherical, rotZ=True: theta rotate around Z; rotZ=False: theta rotate around Y
    # wx,wy,wz, additional translation, normally the center coord.
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    if rotZ:  # swap yz, and keep right-hand
        c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w

    ct = torch.Tensor([[1, 0, 0, wx], [0, 1, 0, wy], [0, 0, 1, wz], [0, 0, 0, 1]]).float()
    c2w = ct @ c2w

    return c2w


def load_real_capture_frame_data(basedir, half_res=False, split="train", test_view="2"):
    # frame data
    all_imgs = []
    all_poses = []

    with open(os.path.join(basedir, "transforms_aligned.json"), "r") as fp:
        # read render settings
        meta = json.load(fp)
    near = float(meta["near"]) / 2.0
    far = float(meta["far"]) * 2.0
    radius = (near + far) * 0.5
    phi = 20.0
    rotZ = False
    r_center = np.array([0.3382070094283088, 0.38795384153014023, -0.2609209839653898]).astype(np.float32)

    # read scene data
    # x,y,z
    voxel_tran = np.array(
        [
            [0.0, 0.0, 1.0, -0.21816665828228],
            [0.0, 1.0, 0.0, -0.044627271592617035 * 5.0],
            [-1.0, 0.0, 0.0, -0.004908999893814325],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # swap_zx
    voxel_tran = np.stack([voxel_tran[:, 2], voxel_tran[:, 1], voxel_tran[:, 0], voxel_tran[:, 3]], axis=1)
    voxel_scale = np.broadcast_to([0.4909 * 2.5, 0.73635 * 2.0, 0.4909 * 2.0], [3])

    # read video frames
    # all videos should be synchronized, having the same frame_rate and frame_num

    frames = meta["frames"]
    if split == "train":
        target_cam_names = ["2"]
    else:
        # target_cam_names = ["0", "1", "3", "4"]
        # current code only support single view testing
        target_cam_names = [test_view]

    frame_nums = 120
    if "red" in basedir.lower():
        print("red")
        start_i = 33
    elif "blue" in basedir.lower():
        print("blue")
        start_i = 55
    else:
        raise ValueError("Unknown dataset")

    for frame_dict in frames:
        cam_name = frame_dict["file_path"][-1:]  # train0x -> x used to determine with train_views
        if cam_name not in target_cam_names:
            continue

        camera_angle_x = float(frame_dict["camera_angle_x"])
        camera_hw = frame_dict["camera_hw"]
        H, W = camera_hw

        Focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        if half_res:
            H = H // 2
            W = W // 2
            Focal = Focal / 2.0

        imgs = []

        for time_idx in range(start_i, start_i + frame_nums * 2, 2):

            frame_path = os.path.join(frame_dict["file_path"], f"{time_idx:03d}.png")
            frame = cv2.imread(os.path.join(basedir, frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if half_res:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

            imgs.append(frame)

        # print(f"video {train_video['file_name']} focal {Focal}")
        imgs = np.float32(imgs) / 255.0

        all_imgs.append(imgs)
        all_poses.append(np.array(frame_dict["transform_matrix"]).astype(np.float32))

    imgs = np.stack(all_imgs, 0)  # [V, T, H, W, 3]
    imgs = np.transpose(imgs, [1, 0, 2, 3, 4])  # [T, V, H, W, 3]
    poses = np.stack(all_poses, 0)  # [V, 4, 4]
    hwf = np.float32([H, W, Focal])

    print(f"real capture {split} {imgs.shape}, {poses.shape}, {hwf}")

    return imgs, poses, hwf, voxel_tran, voxel_scale, near, far
