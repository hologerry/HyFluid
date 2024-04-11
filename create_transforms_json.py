import json


info_json_path = "data/ScalarReal/info.json"

splits = ["train", "test", "train_test"]

with open(info_json_path, "r") as fp:
    meta = json.load(fp)

for split in splits:

    target_json_path = f"data/ScalarReal/transforms_{split}_hyfluid.json"

    target_dict = {}
    if split in ["train", "test"]:
        video_list = meta[split + "_videos"] if (split + "_videos") in meta else meta["train_videos"][0:1]
    else:
        video_list = meta["train_videos"] + meta["test_videos"]

    frames = []

    for video in video_list:
        cur_frame = {}
        cur_frame["file_path"] = "input/" + video["file_name"].replace(".mp4", "")
        cur_frame["transform_matrix"] = video["transform_matrix"]
        cur_frame["camera_angle_x"] = video["camera_angle_x"]
        cur_frame["camera_hw"] = video["camera_hw"]
        frames.append(cur_frame)

    target_dict["frames"] = frames
    target_dict["frame_bkg_color"] = meta["frame_bkg_color"]
    target_dict["voxel_scale"] = meta["voxel_scale"]
    target_dict["voxel_matrix"] = meta["voxel_matrix"]
    target_dict["render_center"] = meta["render_center"]
    target_dict["near"] = meta["near"]
    target_dict["far"] = meta["far"]
    target_dict["phi"] = meta["phi"]
    target_dict["rot"] = meta["rot"]

    with open(target_json_path, "w") as fp:
        json.dump(target_dict, fp, indent=4, sort_keys=True)
