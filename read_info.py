import json


info_json_path = "data/ScalarReal/info.json"

splits = ["train", "test"]

with open(info_json_path, "r") as fp:
    meta = json.load(fp)
