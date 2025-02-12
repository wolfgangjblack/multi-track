import pytest
import numpy as np
from project.motion import SimpleMotionModel

@pytest.fixture
def motion_model():
    return SimpleMotionModel(state_dim = 8)

def test_motion_model_init(motion_model):
    assert motion_model.state_dim == 8
    assert np.all(motion_model.state == 0)
    assert motion_model.velocity_weight == 0.7


def test_motion_model_predict(motion_model):
    motion_model.state = np.array([10, 20, 30, 40, 1, 2, 3, 4])
    predicted = motion_model.predict()

    expected = np.array([11, 22, 33, 44])
    np.testing.assert_array_almost_equal(predicted, expected)

def test_motion_model_update(motion_model):
    start_pos = np.array([0, 0, 0, 0])
    first_pos = np.array([100, 100, 120, 120]) 
    motion_model.update(first_pos)
    np.testing.assert_array_almost_equal(motion_model.state[:4], [100, 100, 120, 120])
    np.testing.assert_array_almost_equal(motion_model.state[4:8], [30, 30, 6, 6])

    second_pos = np.array([110, 110, 130, 130])
    motion_model.update(second_pos)

    np.testing.assert_array_almost_equal(motion_model.state[:4], [110, 110, 20, 20])
    np.testing.assert_array_almost_equal(motion_model.state[4:8], [24, 24, 4.2, 4.2])

def test_motion_model_update_no_measurement(motion_model):
    start_pos = np.array([0, 0, 0, 0])
    motion_model.update(start_pos)
    motion_model.updaet(start_pos)
    assert np.all(motion_model.state[4:] == 0)

    neg_pos = np.array([-10, -10, -5, -5])
    motion_model.update(neg_pos)
    assert motion_model.state[0] == -10