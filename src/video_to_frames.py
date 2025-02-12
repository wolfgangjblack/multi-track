import os
import cv2
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project.pipeline_utils import load_config
from project.data_utils import extract_frames


def parse_args():
    parser = argparse.ArgumentParser(description='Object Tracking in images')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
    return parser.parse_args()


args = parse_args()
config = load_config(args.config)

video_path = config['videopath']
output_dir = config['savedir']
fps = config.get('fps', 10)

extract_frames(video_path, output_dir, fps)
