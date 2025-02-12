import cv2
import numpy as np
import pytest
from ultralytics import YOLO

from project.pipeline_utils import VehicleTracker


@pytest.fixture
def test_image():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
    return img


@pytest.fixture
def yolo_model():
    return YOLO("yolo11s.pt")


@pytest.fixture
def tracker(yolo_model):
    return VehicleTracker()


def test_end_to_end_pipeline(test_image, tracker):
    tracks = tracker.process_frames_batch([test_image])

    assert isinstance(tracks, list)
    for track_id, track_info in tracks[0].items():
        assert "bbox" in track_info
        assert "cls" in track_info
        assert "label" in track_info
        assert len(track_info["bbox"]) == 4


def test_tracker_id_uniqueness(test_image, tracker):
    tracks1 = tracker.process_frames_batch([test_image])
    tracks2 = tracker.process_frames_batch([test_image])

    track1_ids = [track_id for track_id in tracks1[0].keys()]
    track2_ids = [track_id for track_id in tracks2[0].keys()]

    assert len(track1_ids) == len(set(track2_ids))


def test_pipeline_performance_metric(test_image, tracker):
    import time

    start_time = time.time()
    tracker.process_frames_batch([test_image])
    end_time = time.time()

    assert end_time - start_time < 0.5
