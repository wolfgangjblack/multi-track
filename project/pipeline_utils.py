import os
import cv2
import json
import torch
import logging
import numpy as np
import torch.nn as nn
from ultralytics import YOLO
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
    def __init__(self,
                 embedding_dim=128,
                 device='auto'):
        super().__init__()
        base_model = resnet18(pretrained=True) 
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.embedding = nn.Linear(512, embedding_dim)
        
        ##I developed on a mac because my computer is still packed up
        ##But I wanted to make sure the code would run on a GPU or on metal
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available()
                                    else 'mps' if torch.backends.mps.is_available()
                                    else 'cpu')
        else:
            self.device = device

        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize embeddings
    
class VehicleTracker:
    """
    A simple vehicle tracking pipeline class using velocity prediction and feature similarity.
    A velocity prediction help us predict the next location of a vehicle based on its previous locations.
    This is based off the assumption that vehicles move smoothly between frames and loosely looks like 
    newton raphson interpolation if you don't think too hard about it. 
    The Feature similarity uses embeddings from a pretrained deep learning model and 
    helps us match detections to tracks based on appearance.
    
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
                 feature_threshold: float = 0.7,
                 device: str = 'auto',
                 yolo_model: str = 'yolo11s.pt',
                 yolo_labels: Dict[int,str] = {1:'bicycle',
                                    2:'car',
                                    3:'motorcycle',
                                    5:'bus',
                                    7:'truck'},
                 batch_size: int = 4):

        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.max_missed_frames = max_missed_frames
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.batch_size = batch_size
        


        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(embedding_dim=feature_dim, device = device)
        self.feature_extractor.eval()
        self.yolo_labels = yolo_labels
        self.device = self.feature_extractor.device

        # Initialize Yolo model
        self.yolo_model = YOLO(yolo_model).to(self.device)

    def extract_features(self,
                         images: List[np.ndarray],
                         bboxes: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from image region defined by bbox using resnet18. 
        In the future we need to consider
        1. updating to a more modern architecture (I recommend DeiT)
        2. finetuning on data of interest, with data augmentation
        3. distilling large DieT model down to a smaller model for inference
        """

        crops = []
        for img, bbox in zip(images, bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (224, 224))
            crops.append(crop)
        
        batch = torch.from_numpy(np.stack(crops)) \
            .permute(0, 3, 1, 2).float() / 255.0
        
        batch = batch.to(self.feature_extractor.device)
        
        with torch.no_grad():
            features = self.feature_extractor(batch)
        
        return features.cpu().numpy()
    
    def get_batch_detections(self, 
                             frames: List[np.ndarray],
                             results) -> List[List[Detection]]:
        """
        Processes output from YOLO model to get detections for each frame in a batch.
        Passes bounding boxes into feature extractor to get features.
        """
        detections = []
        for frame_idx, (frame, result) in enumerate(zip(frames, results)):
            frame_detections = []
            boxes = result.boxes.data.cpu()

            batch_boxes = []
            batch_images = []
            batch_info = []

            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box.numpy()
                cls = int(cls_id)

                if cls in self.yolo_labels:
                    bbox = np.array([x1, y1, x2, y2])
                    batch_boxes.append(bbox)
                    batch_images.append(frame)
                    batch_info.append((float(conf), cls))

                    if len(batch_boxes) == self.batch_size:
                        features = self.extract_features(batch_images, batch_boxes)
                        
                        for i, ((conf, cls), bbox, feat) in enumerate(zip(batch_info, batch_boxes, features)):
                            frame_detections.append(
                                Detection(
                                    bbox=bbox,
                                    confidence=conf,
                                    features=feat,
                                    cls=cls,
                                    label=self.yolo_labels[cls]
                                )
                            )

                        batch_boxes = []
                        batch_images = []
                        batch_info = []

            if batch_boxes:
                features = self.extract_features(batch_images, batch_boxes)
                
                for i, ((conf, cls), bbox, feat) in enumerate(zip(batch_info, batch_boxes, features)):
                    frame_detections.append(
                        Detection(
                            bbox=bbox,
                            confidence=conf,
                            features=feat,
                            cls=cls,
                            label=self.yolo_labels[cls]
                        )
                    )
            detections.append(frame_detections)

        return detections
    
    def compute_similarity(self, detection: Detection, track: Track) -> float:
        """
        Enhanced similarity computation that accounts for motion uncertainty
        """
        predicted_bbox = track.motion_model.predict()
        
        # Convert from [x, y, w, h] to [x1, y1, x2, y2] format
        predicted_bbox = np.array([
            predicted_bbox[0],
            predicted_bbox[1],
            predicted_bbox[0] + predicted_bbox[2],
            predicted_bbox[1] + predicted_bbox[3]
        ])
        
        iou = self._compute_iou(predicted_bbox, detection.bbox)
        feature_sim = np.dot(detection.features, track.features)
        
        # When uncertainty is high, trust features more than position
        motion_weight = 0.5
        feature_weight = 0.5
        
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
        """
        Updates the tracks with new detections using the bounding boxes and
        features from the YOLO model.
        First we use predict to add velocity to the current position
        then we compare the similarity of the predicted position to the actual (and the features)
        Using this we can match detections to tracks and update the motion model
        """
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
    
    def process_frames_batch(self,
                           frames: List[np.ndarray]) -> List[Dict[int, np.ndarray]]:
        """Process a batch of frames."""
        results = self.yolo_model(frames, classes=list(self.yolo_labels.keys()))
        
        # Process detections in batch
        detections = self.get_batch_detections(frames, results)
        
        # Update tracker for each frame's detections
        return [self.update(detections) for detections in detections]
      
    def __call__(self, 
                 data_dir: str, 
                 output_dir: str,
                 save_txt:bool = False, 
                 save_frames: bool = True):
        """
        Process a video and save the results to an output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
        
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i+self.batch_size]
            
            frame_paths = [os.path.join(data_dir, f) for f in batch_files]
            frames = [cv2.imread(f) for f in frame_paths]

            batch_tracks = self.process_frames_batch(frames)

            for frame_name, img, tracks in zip(batch_files, frames, batch_tracks):
                for track_id, track_info in tracks.items():
                    bbox = track_info['bbox']
                    label = track_info['label']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{track_id}:{label}",
                            (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                    if save_txt:
                        with open(os.path.join(output_dir, 'bbox.txt'), 'a') as f:
                            f.write(f'{track_id},{x1},{y1},{x2},{y2},{label},\n')
                if save_frames:
                    output_path = os.path.join(output_dir, frame_name)
                    cv2.imwrite(output_path, img)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
    

def setup_logger():
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
