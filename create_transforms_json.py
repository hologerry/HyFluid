import json


info_json_path = "data/ScalarReal/info.json"

splits = ["train", "test"]

with open(info_json_path, "r") as fp:
    meta = json.load(fp)

for split in splits:

    target_json_path = f"data/ScalarReal/transforms_{split}_hyfluid.json"

    target_dict = {}
    video_list = meta[split + "_videos"] if (split + "_videos") in meta else meta["train_videos"][0:1]

    frames = []

    for video in video_list:
        cur_frame = {}
        cur_frame["file_path"] = "input/" + video["file_name"].replace(".mp4", "")
        cur_frame["transform_matrix"] = video["transform_matrix"]
        cur_frame["camera_angle_x"] = video["camera_angle_x"]
        cur_frame["camera_hw"] = video["camera_hw"]
        cur_frame["near"] = meta["near"]
        cur_frame["far"] = meta["far"]
        frames.append(cur_frame)

    target_dict["frames"] = frames

    with open(target_json_path, "w") as fp:
        json.dump(target_dict, fp, indent=4, sort_keys=True)
