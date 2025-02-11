import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from typing import List, Dict
import torch.nn.functional as F
from torchvision.models import resnet18
from project.motion import SimpleMotionModel
from scipy.optimize import linear_sum_assignment
from project.data_classes import Detection, Track

class FeatureExtractor(nn.Module):
    """
    A bounding box feature extractor. Here we utilize a resnet18 pertrained on ImageNet.
    As usage slips we can either
    1. Fine-tune the model on our dataset
    2. Replace the model with a more suitable one
        - this could be a transformer based model which can capture more information with attention
        - this could be a more efficient network like a mobilenet, but this would likely need to be trained on our dataset
    - note: embeddings should not exceed the last layer output dims
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        base_model = resnet18(pretrained=True) 
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.embedding = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize embeddings
    
class VehicleTracker:
    """
    A simple vehicle tracker using velocity prediction and feature similarity.
    A velocity prediction help us predict the next location of a vehicle based on its previous locations.
    Feature similarity helps us match detections to tracks based on appearance.
    
    Parameters:
    - feature_dim: Dimensionality of the features extracted from the image
    - max_missed_frames: Maximum number of frames a track can be missed before deletion
    - min_confidence: Minimum confidence required for a detection to be considered
    - iou_threshold: Minimum IoU required for a detection to be matched to a track
    - feature_threshold: Minimum feature similarity required for a detection to be matched to a track
    """
    def __init__(self, 
                 feature_dim: int = 128,
                 max_missed_frames: int = 30,
                 min_confidence: float = 0.6,
                 iou_threshold: float = 0.3,
                 feature_threshold: float = 0.7):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.max_missed_frames = max_missed_frames
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(embedding_dim=feature_dim)
        self.feature_extractor.eval()
        
    def extract_features(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract features from image region defined by bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (224, 224))
        
        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        crop = crop.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            features = self.feature_extractor(crop)
        
        return features.numpy().squeeze()
    
    def compute_similarity(self, detection: Detection, track: Track) -> float:
        """
        Enhanced similarity computation that accounts for motion uncertainty
        """
        predicted_bbox = track.motion_model.predict()
        uncertainty = track.motion_model.get_state_uncertainty()
        
        # Convert from [x, y, w, h] to [x1, y1, x2, y2] format
        predicted_bbox = np.array([
            predicted_bbox[0],
            predicted_bbox[1],
            predicted_bbox[0] + predicted_bbox[2],
            predicted_bbox[1] + predicted_bbox[3]
        ])
        
        iou = self._compute_iou(predicted_bbox, detection.bbox)
        feature_sim = np.dot(detection.features, track.features)
        
        # Adjust weights based on uncertainty
        # When uncertainty is high, trust features more than position
        motion_weight = 0.5 / (1 + uncertainty)
        feature_weight = 1 - motion_weight
        
        similarity = motion_weight * iou + feature_weight * feature_sim
        
        return similarity

    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def update(self,
               detections: List[Detection]) -> Dict[int, np.ndarray]:
        """Updates with new detections."""
        # Filter low confidence detections
        detections = [d for d in detections if d.confidence > self.min_confidence]
        
        for track in self.tracks.values():
            track.motion_model.predict()
        
        similarity_matrix = np.zeros((len(detections), len(self.tracks)))
        for i, detection in enumerate(detections):
            for j, track in enumerate(self.tracks.values()):
                similarity_matrix[i, j] = self.compute_similarity(detection, track)
        
        if len(detections) > 0 and len(self.tracks) > 0:
            detection_indices, track_indices = linear_sum_assignment(-similarity_matrix)
            
            # Filter out low similarity matches
            valid_matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))
            
            for d_idx, t_idx in zip(detection_indices, track_indices):
                if similarity_matrix[d_idx, t_idx] > self.iou_threshold:
                    valid_matches.append((d_idx, t_idx))
                    unmatched_detections.remove(d_idx)
                    unmatched_tracks.remove(t_idx)
        else:
            valid_matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))
        
        # Update matched tracks
        track_ids = list(self.tracks.keys())
        for detection_idx, track_idx in valid_matches:
            track_id = track_ids[track_idx]
            track = self.tracks[track_id]
            detection = detections[detection_idx]
            
            # Update the motion model with the new detection
            track.motion_model.update(detection.bbox)
            
            # Update features with moving average
            track.features = 0.9 * track.features + 0.1 * detection.features
            track.features /= np.linalg.norm(track.features)
            
            # Update history
            track.history.append(detection.bbox)
            track.missed_frames = 0
            track.age += 1
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.tracks[track_id].missed_frames += 1
        
        # Remove dead tracks
        self.tracks = {
            track_id: track 
            for track_id, track in self.tracks.items()
            if track.missed_frames <= self.max_missed_frames
        }
        
        # Initialize new tracks
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            new_mm = SimpleMotionModel()
            new_mm.update(detection.bbox)
            
            self.tracks[self.next_id] = Track(
                id=self.next_id,
                motion_model=new_mm,
                features=detection.features,
                history=[detection.bbox],
                age=1,
                missed_frames=0,
                cls = detection.cls,
                label = detection.label
            )
            self.next_id += 1
        
        # Return current positions of all tracks
        return {
            track_id: {'bbox': track.history[-1],
                       'cls': track.cls,
                       'label': track.label}
            for track_id, track in self.tracks.items()
        }
    
    def process_frame(self,
                    frame: np.ndarray,
                    yolo_model,
                    yolo_labels = {1:'bicycle',
                                    2:'car',
                                    3:'motorcycle',
                                    5:'bus',
                                    7:'truck'}) -> Dict[int, np.ndarray]:
        """
        Process a single frame for a dictionary of yolo classes. 
        Here we assume we're interested in vehicles.
        Updated for newer YOLO format that returns Results objects.
        """
        results = yolo_model(frame, classes=list(yolo_labels.keys()))
        detections = []

        # Process each detection in the frame
        for result in results:  # Iterate through results
            boxes = result.boxes.data  # Get all boxes from this result
            
            # Process each box in this result
            for box in boxes:
                # Extract data from tensor
                x1, y1, x2, y2, conf, cls_id = box.cpu().numpy()
                cls = int(cls_id)  # Convert to int for dictionary lookup
                
                if cls in yolo_labels:  # Check if this class is one we care about
                    bbox = np.array([x1, y1, x2, y2])
                    features = self.extract_features(frame, bbox)
                    label = yolo_labels[cls]
                    
                    detections.append(
                        Detection(
                            bbox=bbox,
                            confidence=float(conf),
                            features=features,
                            cls=cls,
                            label=label
                        )
                    )
        
        # Update tracker with all detections from this frame
        return self.update(detections)
    
    def __call__(self, 
                 yolo_model,
                 data_dir: str, 
                 output_dir: str,
                 save_txt:bool = False 
                 ):
        """
        Process a video and save the results to an output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        idx = 0
        for frame_name in sorted(os.listdir(data_dir)):
            #I choose not to do enumerate here because users may put
            #other files within their directory that are not images
            if not frame_name.endswith('.jpg'):
                #Can expand this with better file checking
                continue

            idx += 1

            frame_path = os.path.join(data_dir, frame_name)
            img = cv2.imread(frame_path)
            tracks = self.process_frame(img, yolo_model)
            
            for track_id, track_info in tracks.items():
                bbox = track_info['bbox']
                label = track_info['label']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{track_id}:{label}",
                             (x1, y1 - 10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
                if save_txt:
                    ##Right now we're assuming a single class single object
                    ##can expand this to handle multi case senarios but would
                    ##need to figure out how we want to compare output to groundtruth
                    ##ie. does each frame have its own ground truth file for all objs? 
                    with open(os.path.join(output_dir,'bbox.txt'), 'a') as f:
                            f.write(f'{track_id},{x1},{y1},{x2},{y2},{label}, \n')

            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, img)