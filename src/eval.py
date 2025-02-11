import os
import sys
import json
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project import read_positions_from_file, calculate_iou

def parse_args():
    parser = argparse.ArgumentParser(description='Object Tracking in images')
    parser.add_argument('--groundtruth', type=str, required=True, help='Path to groundtruth.txt file - Single object detection supported only') 
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions.txt file - Single object detection supported only')
    return parser.parse_args()


args = parse_args()
groundtruth_file = args.groundtruth
predictions_file = args.predictions

groundtruth = read_positions_from_file(groundtruth_file)
predictions = read_positions_from_file(predictions_file, [1, 2, 3, 4])

if len(groundtruth) > len(predictions):
    raise 'Error - the ground truth and prediction files do not align. There are more groundtruths than predictions'
elif len(groundtruth) < len(predictions):
    raise 'Error - the ground truth and prediction files do not align. There are more predictions than groundtruths'

ious = []
for i in range(len(groundtruth)):
    ious.append(calculate_iou(groundtruth[i], predictions[i]))

stats = {'max_iou': max(ious),
         "min_iou": min(ious),
         "avg_iou": sum(ious)/len(ious)}

savedir = os.path.basename(predictions_file)
savedir = predictions_file.split(savedir)[0]
savefile = os.path.join(savedir, 'results.json')

print(stats)
print(f"stats saved at {savefile}")

with open(savefile, 'w') as f:
    json.dump(stats, f)

