import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project import calculate_iou, load_config, read_positions_from_file


def parse_args():
    parser = argparse.ArgumentParser(description="Object Tracking in images")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON config file"
    )
    return parser.parse_args()


args = parse_args()
config = load_config(args.config)

args = parse_args()
groundtruth_file = config["groundtruth_file"]
predictions_file = config["predictions_file"]

groundtruth = read_positions_from_file(groundtruth_file)
predictions = read_positions_from_file(predictions_file, [1, 2, 3, 4])

if len(groundtruth) > len(predictions):
    raise "Error - the ground truth and prediction files do not align. There are more groundtruths than predictions"
elif len(groundtruth) < len(predictions):
    raise "Error - the ground truth and prediction files do not align. There are more predictions than groundtruths"

ious = []
for i in range(len(groundtruth)):
    ious.append(calculate_iou(groundtruth[i], predictions[i]))

stats = {"max_iou": max(ious), "min_iou": min(ious), "avg_iou": sum(ious) / len(ious)}

savedir = os.path.basename(predictions_file)
savedir = predictions_file.split(savedir)[0]
savefile = os.path.join(savedir, "results.json")

print(stats)
print(f"stats saved at {savefile}")

with open(savefile, "w") as f:
    json.dump(stats, f)
