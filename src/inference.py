##Run this file to detect objects within a series of images
import os
import sys
import argparse
from ultralytics import YOLO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project.pipeline_utils import VehicleTracker

def parse_args():
    parser = argparse.ArgumentParser(description='Object Tracking in images')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output results')
    parser.add_argument('--save_text', type=bool, default=False, help='Flag to save bounding boxes as text')
    return parser.parse_args()

args = parse_args()
data_dir = args.data_dir
output_dir = args.output_dir
save_text = args.save_text

yolo_model = YOLO('yolo11s.pt')
    
tracker = VehicleTracker()

tracker(yolo_model, data_dir, output_dir, save_text)
