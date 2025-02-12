##Run this file to detect objects within a series of images
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project.pipeline_utils import VehicleTracker, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Object Tracking in images")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON config file"
    )
    return parser.parse_args()


args = parse_args()
config = load_config(args.config)
batch_size = config.get("batch_size", 10)
data_dir = config["datadir"]
output_dir = config["savedir"]
save_text = config.get("save_text", False)

tracker = VehicleTracker(batch_size=batch_size)

tracker(data_dir, output_dir, save_text)

print(f"Results saved at {output_dir}")
