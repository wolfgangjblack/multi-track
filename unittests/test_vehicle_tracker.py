import pytest
import numpy as np
from project.pipeline_utils import VehicleTracker
from project.data_classes import Detection, Track

@pytest.fixture
def tracker():
    return VehicleTracker(
        feature_dim = 128, 
        max_missed_frames=30,
        iou_threshold=0.3,
        min_confidence=0.61,
        feature_threshold=0.7
    )

@pytest.fixture
def sample_detection():
    return Detection(
        bbox = np.array([100, 100, 200, 200]),
        confidence=0.9,
        features = np.random.rand(128),
        cls = 2,
        label = 'car' 
    )

def test_tracker_init(tracker):
    assert tracker.max_missed_frames == 30
    assert tracker.iou_threshold == 0.3
    assert tracker.feature_threshold == 0.7
    assert len(tracker.tracks) == 0
    assert tracker.next_id == 0
    assert tracker.max_missed_frames == 30
    assert tracker.min_confidence == 0.61

def test_compute_iou(tracker):
    bbox1 = np.array([0, 0, 10, 10]) 
    bbox2 = np.array([5, 5, 15, 15])
    iou = tracker._compute_iou(bbox1, bbox2)

    intersection = 25
    union = 175
    expected_iou = intersection / union
    assert np.isclose(iou, expected_iou)

def test_track_creation(tracker, sample_detection):
    tracks = tracker.update([sample_detection])

    assert len(tracker.tracks) == 1
    assert 0 in tracker.tracks
    assert len(tracker.tracks[0].history) == 1
    assert tracker.tracks[0].missed_frames == 0

def test_track_update(tracker, sample_detection):
    tracks = tracker.update([sample_detection])


    moved_detection = Detection(
        bbox = np.array([110, 110, 210, 210]),
        confidence = 0.9,
        features = sample_detection.features,
        cls = 2,
        label = 'car'
    )
    tracks = tracker.update([moved_detection])
    assert len(tracker.tracks) == 1
    assert len(tracker.tracks[0].history) == 2
    assert np.allclose(
        tracker.tracks[0].history[-1],
        moved_detection.bbox,
    )